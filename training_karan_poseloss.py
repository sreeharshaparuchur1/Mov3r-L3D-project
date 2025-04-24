import random
import torch
import torch.distributed as dist 
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import tensorboardX
import os
import argparse
import time
from torch.utils.data.distributed import DistributedSampler
import datetime
from torch.utils.data import Dataset, DataLoader
from dino.dino_embedder import DINOEmbedder
from cross_attention import PatchWiseCrossAttentionDecoder, DUSt3RAsymmetricCrossAttention
from heads.dpthead import DPTHead
from depth import DepthEmbedder
from heads.posehead import PoseHead
from scannet_dataset_v7 import BufferedSceneDataset, ScanNetPreprocessor, ScanNetMemoryDataset
from losses.losses import ConfAlignPointMapRegLoss, ConfAlignDepthRegLoss, PoseLoss
from heads.utils_pose import mat_to_quat
import logging
import sys
import gc
import matplotlib.pyplot as plt
from PIL import Image

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="train_embed_v1", help="Run name, default: pointmap_predict")
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--context_length', type=int, default=32)
    parser.add_argument('--max_scenes', type=int, default=10)
    parser.add_argument('--frame_skip', type=int, default=10)

    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--encoder_dim', type=int, default=768)
    parser.add_argument('--decoder_dim', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=8)
    
    parser.add_argument('--prefetch_factor', type=int, default=8)
    parser.add_argument('--dl_num_workers', type=int, default=4)
    parser.add_argument('--ds_num_workers', type=int, default=8)
    
    parser.add_argument('--ca_depth', type=int, default=4)
    parser.add_argument('--cls_pose', type=bool, default=False)
    parser.add_argument('--pos_embed', type=str, default='cosine', help='Position embedding type: cosine or RoPE')
    parser.add_argument('--pc_dec_depth', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clip_grad_norm', type=float, default=5.0)
    parser.add_argument('--clip_grad_val', type=float, default=10.0)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--alpha_pointmap', type=float, default=0.5)
    parser.add_argument('--alpha_depth', type=float, default=0.5)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--alpha_pose', type=float, default=0.5)
    parser.add_argument('--w_pose',  type=float, default=1.)
    parser.add_argument('--w_depth',  type=float, default=1.)
    parser.add_argument('--w_pm',  type=float, default=1.)
    parser.add_argument('--sim_th',  type=float, default=0.5)
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0, 1, 2], help='List of GPU device IDs')

    parser.add_argument('--eval_after', type=int, default=5)
    parser.add_argument('--load_after', type=int, default=5)
    parser.add_argument('--save_after', type=int, default=50)
    parser.add_argument('--log_dir', type=str, default='/data/kmirakho/git/Mov3r-L3D-project/logs/')
    parser.add_argument('--model_dir', type=str, default='/data/kmirakho/git/Mov3r-L3D-project/models/')
    parser.add_argument('--ckpt_dir', type=str, default='/data/kmirakho/l3d_proj/Mov3r-L3D-project/ckpt/')
    parser.add_argument('--load_from_ckpt', type=str, default=None)
    parser.add_argument('--load_model', action="store_true")
    parser.add_argument('--run_eval', action="store_true")
    parser.add_argument('--eval_model', type=str, default='/data/kmirakho/offline_data/model/data_10_40_640/model_20.pth')
    parser.add_argument('--dataset_path', type=str, default='/data/kmirakho/l3d_proj/scannetv2')
    parser.add_argument('--dino_encoder', type=str, default='/data/kmirakho/l3dProject/git/Mov3r-L3D-project/pretrained_weights/dinov2_vitb14_reg4_pretrain.pth')
    parser.add_argument('--depth_embedder', type=str, default='./pretrained_weights/align3r_depthanything.pth')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args

class UnifiedModel(torch.nn.Module):
    def __init__(self, args):
        super(UnifiedModel, self).__init__()
        self.args = args
        self.depth_embedder = DepthEmbedder(patch_embed_cls='PatchEmbedDust3R', img_size=args.img_size, patch_size=args.patch_size, dec_embed_dim=args.decoder_dim, pos_embed=args.pos_embed, pc_dec_depth=args.pc_dec_depth)

        self.dino_patch_embed = self.dino_patch_embed = DINOEmbedder(patch_embed='dinov2_vitb14_reg', img_size=args.img_size, patch_size=args.patch_size)

        # Initialize the CrossAttention model
        self.dust3r_cross = DUSt3RAsymmetricCrossAttention(
            encoder_dim=args.encoder_dim, 
            decoder_dim=args.decoder_dim, 
            depth=args.ca_depth, 
            num_heads=args.num_heads, 
            dropout=args.dropout
        )

        self.point_head = DPTHead(dim_in=args.embed_dim, patch_size=args.patch_size, output_dim=4, activation="inv_log", conf_activation="sigmoid")
        self.depth_head = DPTHead(dim_in=args.embed_dim, patch_size=args.patch_size, output_dim=2, activation="exp", conf_activation="sigmoid")
        self.pose_head = PoseHead(S=args.context_length, emb_dim=args.embed_dim, alpha=0.5)
    
    def encoder(self, rgb, pred_depth, intrinsic_depth):
        pc_embedding = self.depth_embedder(pred_depth, intrinsic_depth)

        rgb = rgb.permute(0, 3, 1, 2).contiguous()
        if self.dino_patch_embed.training:
            patch_embedding = self.dino_patch_embed(rgb)
        else:
            with torch.no_grad():
                patch_embedding = self.dino_patch_embed(rgb)
        
        features = self.dust3r_cross(patch_embedding, pc_embedding)
        return features
    
    def decoder(self, features, rgb):
        BS, H, W, C = rgb.shape
        BS, P, D = features.shape
        S = self.args.context_length
        assert BS % S == 0, "Batch size must be divisible by context length"
        B = BS // S

        rgb = rgb.permute(0,3,1,2)
        rgb = rgb.unsqueeze(1)
        # import pdb; pdb.set_trace()
        #predict value and confidence for depth and pointmap
        predict_depth = self.depth_head([features.unsqueeze(1)], rgb, patch_start_idx=1)
        predict_pointmap = self.point_head([features.unsqueeze(1)], rgb, patch_start_idx=1)
        predict_pose = self.pose_head(features.view(B, S, P, D).contiguous()[:, :,0,:])
        
        predict_depth = list(predict_depth)
        predict_pointmap = list(predict_pointmap)

        predict_depth[0] = predict_depth[0].view(B, S, H, W, 1).contiguous()
        predict_depth[1] = predict_depth[1].view(B, S, H, W, 1).contiguous()

        predict_pointmap[0] = predict_pointmap[0].view(B, S, H, W, C).contiguous()
        predict_pointmap[1] = predict_pointmap[1].view(B, S, H, W, 1).contiguous()

        assert predict_depth[0].shape[-1] == 1
        assert predict_pointmap[0].shape[-1] == 3
        assert predict_pose[0].shape[-1] == 7

        return predict_depth, predict_pointmap, predict_pose

    def forward(self, rgb, pred_depth, intrinsic_depth):
        features = self.encoder(rgb, pred_depth, intrinsic_depth)
        predict_depth, predict_pointmap, predict_pose = self.decoder(features, rgb)
        return predict_depth, predict_pointmap, predict_pose
    
    def load_depth_embedder(self, device):
        depth_state_dict = torch.load(args.depth_embedder, map_location=device)
        filtered_state_dict = {k: v for k, v in depth_state_dict['model'].items() if k.startswith('dec_blocks_pc')}
        self.depth_embedder.dec_blocks_pc.load_state_dict({k.replace('dec_blocks_pc.', ''): v for k, v in filtered_state_dict.items() if k.startswith('dec_blocks_pc.')}, strict=False)
        del depth_state_dict, filtered_state_dict
    
    def load_dino_patch_embed(self, device):
        dino_model_state_dict = torch.load(args.dino_encoder, map_location=device)
        # Remove the 'patch_embed.' prefix from the keys
        for k, _ in self.dino_patch_embed.state_dict().items():
            new_key = k.replace('patch_embed.', '', 1)
            self.dino_patch_embed.state_dict()[k] = dino_model_state_dict[new_key]

        del dino_model_state_dict

    def loss(self, rgb, pred_depth, depth, intrinsic_depth, pose, args, mask=False):
        BS, H, W, C = depth.shape
        S = self.args.context_length
        assert BS % S == 0, "Batch size must be divisible by context length"
        B = BS // S
        predict_depth, predict_pointmap, predict_pose = self.forward(rgb, pred_depth, intrinsic_depth)
        
        depth = depth.view(B, S, H, W, C).contiguous()
        pose = torch.cat([mat_to_quat(pose[..., :3,:3]), pose[..., -1, :3]], dim=-1)

        assert depth.shape[-1] == 1
        assert intrinsic_depth.shape[-1] == 4
        assert intrinsic_depth.shape[-2] == 4
        assert pose.shape[-1] == 7

        loss_pointmap = ConfAlignPointMapRegLoss(depth, predict_pointmap, intrinsic_depth, alpha = args.alpha_pointmap, eps=args.eps)
        loss_depth = ConfAlignDepthRegLoss(depth, predict_depth,  alpha = args.alpha_depth, eps=args.eps)
        loss_pose = PoseLoss(pose, predict_pose)
        return loss_pointmap, loss_depth, loss_pose

class Mov3r:
    def __init__(self, args, dist, local_rank, data_transforms=None):
        self.num_epochs = args.num_epochs
        self.dataset_path = args.dataset_path
        self.batch_size = args.batch_size
        self.model_dir = args.model_dir
        self.ckpt_dir = args.ckpt_dir
        self.run_name = args.run_name
        self.dist =  dist
        self.log_dir = args.log_dir + args.run_name
        self.pc_dec_depth = args.pc_dec_depth
        self.local_rank = local_rank
        self.transforms = data_transforms
        self.device = torch.device(f"cuda:{self.local_rank}")
        
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        
        self.ckpt_dir = os.path.join(self.ckpt_dir, self.run_name) #self.ckpt_dir + self.run_name
        if not os.path.exists(self.ckpt_dir) and local_rank == 0:
            os.makedirs(self.ckpt_dir)
        
        self.model_dir = os.path.join(self.model_dir, self.run_name)
        if not os.path.exists(self.model_dir) and local_rank == 0:
            os.makedirs(self.model_dir)
        
        self.log_dir = os.path.join(self.log_dir, f"{self.run_name}_{timestamp}")
    
        if not os.path.exists(self.log_dir) and local_rank == 0:
            os.makedirs(self.log_dir)
                
        # if self.local_rank == 0:
        #     self.log_file = os.path.join(self.log_dir, f"{self.run_name}_{timestamp}.log")

        #     print(f"Log file: {self.log_file}")
        #     self.log_file = open(self.log_file, 'w')
        #     sys.stdout = self.log_file
        #     sys.stderr = self.log_file
    
        #     logging.basicConfig(
        #         level=logging.INFO,
        #         format='%(asctime)s - %(levelname)s - %(message)s',
        #         handlers=[
        #             logging.StreamHandler(self.log_file)
        #         ]
        #     )
        #     logging.info("Starting training...")
        #     logging.info(f"Arguments: {args}")
        #     logging.info(f"World size: {dist.get_world_size()}")            

        #Preprocess and load dataset to memory
        self.buffer_scene = BufferedSceneDataset(
            root_dir=args.dataset_path,
            max_scenes=args.max_scenes,
            num_workers=args.ds_num_workers,
            num_frames=args.context_length,
            frame_skip=args.frame_skip,
            data_transforms=None
        )

        self.writer = tensorboardX.SummaryWriter(self.log_dir)

        #create the model
        self.model = UnifiedModel(args)
        if args.load_model:
            self.load_model()
        self.model.load_depth_embedder(self.device)
        self.model.load_dino_patch_embed(self.device)
        self.model.to(self.device)
        
        #distributed training
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        #optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        #scheduler 
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=args.gamma)

        #avg loss
        self.avg_loss = 0

        torch.cuda.empty_cache()
        gc.collect()

    def create_distributed_loader(self, dataset):
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.dist.get_world_size(),
            rank=self.dist.get_rank(),
            shuffle=True
            )

        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=args.dl_num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True
        )
        return data_loader, sampler   
    

    def train(self):
        #set model to train
        self.model.train()
        if args.load_from_ckpt is not None:
            epoch = self.load_checkpoint(args)
            train_dataset = self.buffer_scene.fetch_dataset()
            self.train_loader, self.train_sampler = self.create_distributed_loader(train_dataset)
        else :
            epoch = 0

        #set DINO to eval
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.module.dino_patch_embed.eval()
        else:
            self.model.dino_patch_embed.eval()

        while epoch < self.num_epochs:
            #new the next k scenes for training
            if epoch % args.load_after == 0:
                train_dataset = self.buffer_scene.fetch_dataset()
                self.train_loader, self.train_sampler = self.create_distributed_loader(train_dataset)
            self.train_sampler.set_epoch(epoch)

            #epoch bar for training each epoch       
            epoch_bar = tqdm(
                self.train_loader,
                desc=f'Epoch {epoch+1} (rank {self.local_rank})',
                bar_format='{l_bar}{bar:20}{r_bar}',
                leave=True,
                position=self.local_rank,
                disable=not (self.local_rank == 0),
                )

            #set avg loss to zero
            self.avg_loss = 0
            self.avg_loss_pointmap = 0
            self.avg_loss_depth = 0
            self.avg_loss_pose = 0

            for batch_idx, batch in enumerate(epoch_bar):
                rgb, depth, pred_depth, pose, intrinsic = (batch["rgb"].to(self.device, non_blocking=True), batch["depth"].to(self.device, non_blocking=True),
                                                    batch["pred_depth"].to(self.device, non_blocking=True), batch["pose"].to(self.device, non_blocking=True), batch["intrinsics"])

                rgb = rgb.to(torch.float32, non_blocking=True)/255.0
                depth = depth.to(torch.float32, non_blocking=True)/1000.0
                pred_depth = pred_depth.to(torch.float32, non_blocking=True)/255
                pose = pose.to(torch.float32, non_blocking=True)

                if self.transforms is not None:
                    # Apply the transformations
                    rgb = self.transforms(rgb)
                
                # rgb shape [B, S, C, H, W]
                _, _, C, H, W = rgb.shape 
                
                #Collapsed the batch and sequence dimension
                rgb = rgb.view(-1, C, H, W).contiguous() # BS, C, H, W

                #permuting the channel dimension to the front
                rgb = rgb.permute(0, 2, 3, 1).contiguous() # BS, H, W, C
                
                # debug rgb images
                # rg = self.float32_to_uint8(rgb[0].cpu().numpy())
                # self.save_image(rg, batch_idx, "debug_rgb", good=True)

                #collapsed the batch and sequence dimension
                depth = depth.view(-1, H, W, 1).contiguous() # BS, H, W, 1
                pred_depth = pred_depth.view(-1, H, W, 1).contiguous() # BS, H, W, 1

                # debug depth and pred_depth images
                # dg = self.float32_to_uint8(depth[0].cpu().numpy())
                # self.save_image(dg, batch_idx, "debug_depth", good=True)

                # pdg = self.float32_to_uint8(pred_depth[0].cpu().numpy())
                # self.save_image(pdg, batch_idx, "debug_pred_depth", good=True)

                assert rgb.shape[-1] == 3
                assert depth.shape[-1] == 1
                assert pred_depth.shape[-1] == 1

                # Check for NaN/Inf in inputs
                assert not torch.isnan(rgb).any(), "NaN in RGB inputs"
                assert torch.isfinite(depth).all(), "Non-finite depth values"

                try:
                    intrinsic = {k: v.to(torch.float32, non_blocking=True) for k, v in intrinsic.items()}
                    intrinsic_depth = intrinsic['intrinsic_depth'].to(self.device, non_blocking=True)
                    intrinsic_depth = intrinsic_depth.repeat_interleave(repeats=args.context_length, dim=0)
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    #calculate loss
                    loss_pointmap, loss_depth, loss_pose = self.model.module.loss(rgb, pred_depth, depth, intrinsic_depth, pose, args)
                    
                    loss = args.w_pm*loss_pointmap + args.w_depth*loss_depth + args.w_pose*loss_pose
                    loss.backward()
                    
                    # Clip gradients after synchronization
                    self.g_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=args.clip_grad_norm)

                    # #Clip gradient values
                    # self.g_val = torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=args.clip_grad_val)
                    
                    self.optimizer.step()
                    
                    #sync losses across all GPUs
                    loss_tensor = loss.detach().clone()
                    dist.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
                    self.avg_loss = (loss_tensor.item() / dist.get_world_size() + self.avg_loss * batch_idx) / (batch_idx + 1)
                    
                    self.g_norm = (self.g_norm / dist.get_world_size() + self.g_norm * batch_idx) / (batch_idx + 1)
                    epoch_bar.set_postfix(
                        avg_loss=f'{self.avg_loss:.2f}',
                        loss_pm=f'{loss_pointmap:.2f}',
                        loss_depth=f'{loss_depth:.2f}',
                        loss_pose=f'{loss_pose:.2f}',
                        # lr=f'{self.optimizer.param_groups[0]["lr"]:.5f}',
                        grad_norm=f'{self.g_norm:.2f}',
                    )
            
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    print(f"Skipping batch {batch_idx} due to error.")
            if epoch % args.save_after == 0 and self.local_rank == 0:
                self.save_checkpoint(epoch)
            
            if self.writer is not None and self.local_rank == 0:
                self.log_data(epoch)
            
            # if epoch % args.eval_after == 0 and self.local_rank == 0:
            #     self.eval(epoch)
            
            self.scheduler.step()
            self.dist.barrier()
            epoch += 1

        # if self.local_rank == 0:
        #     logging.info("Training completed.")    
        #     self.log_file.close()
            
    def log_data(self, epoch):
        self.writer.add_scalar('Avg_loss', self.avg_loss, epoch)
        self.writer.add_scalar('grad_norm', self.g_norm, epoch)
        #optimizer
        self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)
        for name, param in self.model.module.dust3r_cross.named_parameters():
            self.writer.add_histogram("dust3r_cross/"+name, param.clone().cpu().data.numpy(), epoch)
        for name, param in self.model.module.depth_embedder.named_parameters():
            self.writer.add_histogram("depth_embedder/"+name, param.clone().cpu().data.numpy(), epoch)
        for name, param in self.model.module.point_head.named_parameters():
            self.writer.add_histogram("point_head/"+name, param.clone().cpu().data.numpy(), epoch)            
        for name, param in self.model.module.depth_head.named_parameters():
            self.writer.add_histogram("depth_head/"+name, param.clone().cpu().data.numpy(), epoch)            

    def load_model(self, model_path):
        if os.path.exists(model_path):
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"Model path {model_path} does not exist")

    def save_model(self, epoch):        
        #save the model
        model_path = self.model_dir + '/model_'+str(epoch)+'.pth'
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            torch.save(self.model.module.state_dict(), model_path)
        else:
            torch.save(self.model.state_dict(), model_path)
        # print(f"Model saved to {model_path}")

    def save_checkpoint(self, epoch):
        fname = str(epoch)
        checkpoint_path = os.path.join(self.ckpt_dir, 'checkpoint-%s.pth' % fname)
        to_save = {
            'model': self.model.module.state_dict() if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
            'avg_loss': self.avg_loss,
        }

        torch.save(to_save, checkpoint_path)

    def load_checkpoint(self, args):
        checkpoint = torch.load(args.load_from_ckpt, map_location='cpu')
        self.model.module.load_state_dict(checkpoint['model'], strict=False)
        epoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        avg_loss = checkpoint['avg_loss']
        print(f'Loaded checkpoint from {args.load_from_ckpt} at epoch {epoch} with avg_loss {avg_loss}')
        return epoch
                
if __name__ == "__main__":
    args = get_config()

    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set the number of threads for PyTorch
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    # Set NCCL environment variables
    os.environ['NCCL_ALGO'] = 'Ring'
    os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'
    os.environ['NCCL_SOCKET_NTHREADS'] = '2'
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])

    # Initialize process group first
    dist.init_process_group(
        backend='nccl',  # Use 'gloo' for CPU
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=10000)
        )
    
    torch.cuda.set_device(local_rank)

    # Define transforms
    data_transforms = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # dataset = buffer_scene.fetch_dataset()
    mov3r = Mov3r(args, dist, local_rank, data_transforms=data_transforms)
    try:
        # Training loop
        mov3r.train()
    finally:
        dist.destroy_process_group()