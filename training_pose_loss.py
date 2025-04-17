
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
from scannet_dataset_v4 import ScanNetPreprocessor, ScanNetMemoryDataset
from losses.losses import ConfAlignPointMapRegLoss, ConfAlignDepthRegLoss

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="pointmap_predict", help="Run name, default: pointmap_predict")
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--context_length', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--encoder_dim', type=int, default=512)
    parser.add_argument('--decoder_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--ca_depth', type=int, default=4)
    parser.add_argument('--pos_embed', type=str, default='cosine', help='Position embedding type: cosine or RoPE')
    parser.add_argument('--pc_dec_depth', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0, 1, 2], help='List of GPU device IDs')
    parser.add_argument('--log_dir', type=str, default='/data/kmirakho/git/Mov3r-L3D-project/logs/')
    parser.add_argument('--model_dir', type=str, default='/data/kmirakho/git/Mov3r-L3D-project/models/')
    parser.add_argument('--load_model', action="store_true")
    parser.add_argument('--run_eval', action="store_true")
    parser.add_argument('--eval_model', type=str, default='/data/kmirakho/offline_data/model/data_10_40_640/model_20.pth')
    parser.add_argument('--dataset_path', type=str, default='/data/kmirakho/git/Mov3r-L3D-project/data/')
    parser.add_argument('--dino_encoder', type=str, default='/data/kmirakho/l3dProject/git/Mov3r-L3D-project/pretrained_weights/dinov2_vitb14_reg4_pretrain.pth')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args



class Mov3r:
    def __init__(self, args, dist, local_rank):
        self.num_epochs = args.num_epochs
        self.dataset_path = args.dataset_path
        self.batch_size = args.batch_size
        self.model_dir = args.model_dir
        self.run_name = args.run_name
        self.dist =  dist
        self.log_dir = args.log_dir + args.run_name
        self.pc_dec_depth = args.pc_dec_depth
        self.local_rank = local_rank
        torch.cuda.set_device(local_rank)
        self.device = torch.device(f"cuda:{local_rank}")
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
        self.writer = tensorboardX.SummaryWriter(self.log_dir)

        #generate list of 100 lists of 700 dictionaries with keys 'color', 'depth', 'pose', 'extrinsics'
        train_dataset, val_dataset = dataset, dataset #load dataset here
        
        #create train dataloader and sampler
        self.train_loader, self.train_sampler = self.create_distributed_loader(train_dataset)
        
        #create val dataloader and sampler
        self.val_loader, self.val_sampler = self.create_distributed_loader(val_dataset)

        #  Depth Image Self attention Module Takes depth image and projects to point map and then takes to 
        # Initialize ViT model for self attention
        self.depth_embedder = DepthEmbedder(patch_embed_cls='PatchEmbedDust3R', img_size=224, patch_size=16, dec_embed_dim=768, pos_embed=args.pos_embed, pc_dec_depth=self.pc_dec_depth)

        self.dino_patch_embed = DINOEmbedder(patch_embed='dinov2_vitb14_reg', img_size=224, patch_size=16)
        # self.dino_patch_embed.load_state_dict(torch.load(args.dino_encoder, map_location=self.device))
        dino_model_state_dict = torch.load(args.dino_encoder, map_location=self.device)

        # Remove the 'patch_embed.' prefix from the keys
        for k, _ in self.dino_patch_embed.state_dict().items():
            new_key = k.replace('patch_embed.', '', 1)
            self.dino_patch_embed.state_dict()[k] = dino_model_state_dict[new_key]

        # Initialize the CrossAttention model
        self.dust3r_cross = DUSt3RAsymmetricCrossAttention(
            encoder_dim=args.encoder_dim, 
            decoder_dim=args.decoder_dim, 
            depth=args.ca_depth, 
            num_heads=args.num_heads, 
            dropout=args.dropout
        )
        # Initialize the PointMap & Depth Predictor model
        self.point_head = DPTHead(dim_in=args.embed_dim, patch_size=args.patch_size, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=args.embed_dim, patch_size=args.patch_size, output_dim=2, activation="exp", conf_activation="expp1")

        self.depth_embedder = self.depth_embedder.to(self.device)
        self.dino_patch_embed =  self.dino_patch_embed.to(self.device)
        self.dust3r_cross = self.dust3r_cross.to(self.device)
        self.point_head = self.point_head.to(self.device)
        self.depth_head = self.depth_head.to(self.device)


        if torch.cuda.device_count() > 1:
            #data parallel for ViT Model selfAttention
            self.depth_embedder = torch.nn.parallel.DistributedDataParallel(self.depth_embedder, device_ids=[local_rank])
            
            self.dino_patch_embed = torch.nn.parallel.DistributedDataParallel(self.dino_patch_embed, device_ids=[local_rank])
            
            #data parallel for CrossAttention Model
            self.dust3r_cross = torch.nn.parallel.DistributedDataParallel(self.dust3r_cross, device_ids=[local_rank])
            
            #data parallel for PointMap and Depth Head
            self.point_head = torch.nn.parallel.DistributedDataParallel(self.point_head, device_ids=[local_rank])

            self.depth_head = torch.nn.parallel.DistributedDataParallel(self.depth_head, device_ids=[local_rank])

        
        self.optimizer = torch.optim.AdamW([
            #parameters for ViT model to be added
            {'params': self.depth_embedder.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5},
            {'params': self.dino_patch_embed.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5},
            {'params': self.dust3r_cross.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5},
            {'params': self.point_head.parameters(), 'lr': 2e-4, 'weight_decay': 1e-6},
            {'params': self.depth_head.parameters(), 'lr': 2e-4, 'weight_decay': 1e-6}
            ], lr=1e-4, weight_decay=1e-5)
        

        #scheduler 
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=args.gamma)
        # Mixed precision setup
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        #avg loss
        self.avg_loss = 0

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
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        return data_loader, sampler
    
    def get_loss(self, predict_pointmap, predict_depth, depth , intrinsic_depth, weights = [0.01,0.01,1], mask=False):
        # Image [B,S,C,H,W ]
        # Depth [B,S, H,W,C]
        # features (B, S, patch_tokens, dim_in)
        # instrinsic shape [B,S,4,4]
        
        depth = depth.permute(0, 1, 3, 4, 2)
        assert depth.shape[-1] == 1
        assert intrinsic_depth.shape[-1] == 4
        assert intrinsic_depth.shape[-2] == 4

        loss_pointmap = ConfAlignPointMapRegLoss(depth, predict_pointmap, intrinsic_depth, weights[0])
        loss_depth = ConfAlignDepthRegLoss(depth, predict_depth,  weights[1])
        loss_pose = ConfAlignPoseLoss(pose, predicted_pose)

        return loss_pointmap + loss_depth

    
    def dpt_head(self,features, images):
        predict_depth = self.depth_head([features], images, patch_start_idx=0)
        predict_pointmap = self.point_head(
            [features], images, patch_start_idx=0
        )

        return predict_depth, predict_pointmap

    def train(self):
        #set DINO to eval
        self.dino_patch_embed.eval()

        #set the depth embedder to train
        self.depth_embedder.train()

        #set the crossAttention to train
        self.dust3r_cross.train()
        
        #set the point head to train
        self.point_head.train()

        #set the depth head to train
        self.depth_head.train()
        
        for epoch in range(self.num_epochs):
            
            self.train_sampler.set_epoch(epoch)

            #epoch bar for training each epoch
            local_rank = self.local_rank          
            epoch_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}', disable=local_rank != 0, bar_format='{l_bar}{bar:20}{r_bar}', leave=True)

            #set avg loss to zero
            self.avg_loss = 0

            #start time
            start_time = time.time()

            for batch_idx, batch in enumerate(epoch_bar):
                rgb, depth, _, intrinsic = (batch["rgb"].to(self.device, non_blocking=True), batch["depth"].to(self.device, non_blocking=True), batch["pose"].to(self.device, non_blocking=True), batch
                ["intrinsics"])
                B, S, C, H, W = rgb.shape
                rgb = rgb.view(-1, H, W, C)

                _, _, H, W = depth.shape
                depth = depth.view(-1, H, W, 1)
                intrinsic_depth = intrinsic['intrinsic_depth'].to(self.device, non_blocking=True)
                intrinsic_depth = intrinsic_depth.repeat_interleave(repeats=S, dim=0)

                assert rgb.shape[-1] == 3
                assert depth.shape[-1] == 1
                
                pc_embedding = self.depth_embedder(depth, intrinsic_depth)

                rgb = rgb.permute(0, 3, 1, 2)
                with torch.no_grad():
                    patch_embedding = self.dino_patch_embed(rgb)
                
                features = self.dust3r_cross(patch_embedding, pc_embedding)
                features = features.view(B*S, 1, features.shape[-2], features.shape[-1])
                rgb = rgb.view(B*S, 1, C, H, W)
                depth = depth.view(B, S, 1, H, W)

                predict_depth, predict_pointmap = self.dpt_head(features,rgb)
                predict_depth = list(predict_depth)
                predict_pointmap = list(predict_pointmap)
                predict_depth[0] = predict_depth[0].view(B, S, H, W, 1)
                predict_depth[1] = predict_depth[1].view(B, S, H, W, 1)

                predict_pointmap[0] = predict_pointmap[0].view(B, S, H, W, C)
                predict_pointmap[1] = predict_pointmap[1].view(B, S, H, W, 1)
                assert predict_depth[0].shape[-1] == 1
                assert predict_pointmap[0].shape[-1] == 3

                self.optimizer.zero_grad(set_to_none=True)
                
                with torch.cuda.amp.autocast():
                    #calculate loss
                    loss = self.get_loss(predict_pointmap, predict_depth, depth , intrinsic_depth)
            
                # self.scaler.scale(loss)
                # self.scaler.step(self.optimizer)
                # self.scaler.update()
                loss.backward()
                self.optimizer.step()

                #sync losses across all GPUs
                loss_tensor = loss.detach().clone()
                dist.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
                self.avg_loss = (loss_tensor.item() / dist.get_world_size() + self.avg_loss * batch_idx) / (batch_idx + 1)

                if local_rank == 0:
                    epoch_bar.set_postfix(
                        avg_loss=f'{self.avg_loss:.5f}',
                        batch_rate=f'{(batch_idx+1)/(time.time()-start_time):.2f}'
                    )
                
                # global features
                # local features
                # key frame selection
                # pose prediction
        if epoch % 10 == 0 and local_rank == 0:
            self.save_model(epoch)

        if self.writer is not None and local_rank == 0:
            self.log_data(epoch)

    
    def log_data(self, epoch):
        self.writer.add_scalar('Avg_loss', self.avg_loss, epoch)
        for name, param in self.dust3r_cross.named_parameters():
            self.writer.add_histogram("dust3r_cross/"+name, param.clone().cpu().data.numpy(), epoch)
        for name, param in self.point_head.named_parameters():
            self.writer.add_histogram("point_head/"+name, param.clone().cpu().data.numpy(), epoch)            
        for name, param in self.depth_head.named_parameters():
            self.writer.add_histogram("depth_head/"+name, param.clone().cpu().data.numpy(), epoch)            

    
    def save_model(self, epoch):
        model_dir = self.model_dir + self.run_name
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        #save the ViT encoder mode
        model_path = model_dir + '/dino_patch_embed_'+str(epoch)+'.pth'
        if isinstance(self.dino_patch_embed, torch.nn.DataParallel):
            torch.save(self.dino_patch_embed.module.state_dict(), model_path)
        else:
            torch.save(self.dino_patch_embed.state_dict(), model_path)

        #save the depth embedder model
        model_path = model_dir + '/depth_embedder_'+str(epoch)+'.pth'
        if isinstance(self.depth_embedder, torch.nn.DataParallel):
            torch.save(self.depth_embedder.module.state_dict(), model_path)
        else:
            torch.save(self.depth_embedder.state_dict(), model_path)
     
        #save the crossAttention Model
        model_path = model_dir +'/dust3r_cross_'+str(epoch)+'.pth'
        if isinstance(self.dust3r_cross, torch.nn.DataParallel):
            torch.save(self.dust3r_cross.module.state_dict(), model_path)
        else:
            torch.save(self.dust3r_cross.state_dict(), model_path)

        #save the point head
        model_path = model_dir +'/point_head_'+str(epoch)+'.pth'
        if isinstance(self.point_head, torch.nn.DataParallel):
            torch.save(self.point_head.module.state_dict(), model_path)
        else:
            torch.save(self.point_head.state_dict(), model_path)

        #save the depth head Model
        model_path = model_dir +'/depth_head_'+str(epoch)+'.pth'
        if isinstance(self.depth_head, torch.nn.DataParallel):
            torch.save(self.depth_head.module.state_dict(), model_path)
        else:
            torch.save(self.depth_head.state_dict(), model_path)




if __name__ == "__main__":
    args = get_config()
    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set the device IDs for distributed training
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.device_ids))
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
        rank=local_rank,
        timeout=datetime.timedelta(seconds=30)
        )
    

    # Define transforms
    data_transforms = transforms.Compose([
        # transforms.ToTensor(),
        # transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Step 1: Preprocess and load dataset to memory
    preprocessor = ScanNetPreprocessor(
        root_dir=args.dataset_path,
        output_h5_path='scannet_preprocessed.h5',
        max_scenes=1,  # Limit to 100 scenes for memory efficiency
        num_workers=8,   # Adjust based on your system
        rgb_only=False,  # Set to True to only load RGB images
        compression="lzf" # Fast compression
    )
    
    # Option 2: Load directly to memory
    memory_dataset = preprocessor.load_dataset(save_to_disk=False, return_data=True)

    # From memory (fastest, but requires most RAM)
    dataset = ScanNetMemoryDataset(
        dataset_source=memory_dataset,
        num_frames=args.context_length,
        transforms=data_transforms,
        frame_skip=1,
        rgb_only=False
    )


    mov3r = Mov3r(args, dist, local_rank)
    mov3r.train()
    # Initialize the PointMap & Depth Predictor model