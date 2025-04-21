import random
import torch
# import torch.distributed as dist 
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
from scannet_dataset_v7 import BufferedSceneDataset, ScanNetPreprocessor, ScanNetMemoryDataset
from losses.losses import ConfAlignPointMapRegLoss, ConfAlignDepthRegLoss
import logging
import sys
import gc
from PIL import Image
import matplotlib.pyplot as plt

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
    parser.add_argument('--save_evals', type=str)
    parser.add_argument('--return_sum', action='store_true')
    parser.add_argument('--visualize_results', action='store_true')

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
        S = self.args.context_length
        assert BS % S == 0, "Batch size must be divisible by context length"
        B = BS // S
        features = features.view(BS, 1, features.shape[-2], features.shape[-1]).contiguous()
        rgb = rgb.view(BS, 1, C, H, W).contiguous()
        predict_depth = self.depth_head([features], rgb, patch_start_idx=0)
        predict_pointmap = self.point_head([features], rgb, patch_start_idx=0)
        predict_depth = self.depth_head([features], rgb, patch_start_idx=0)
        predict_pointmap = self.point_head(
            [features], rgb, patch_start_idx=0
        )
        predict_depth = list(predict_depth)
        predict_pointmap = list(predict_pointmap)
        predict_depth[0] = predict_depth[0].view(B, S, H, W, 1).contiguous()
        predict_depth[1] = predict_depth[1].view(B, S, H, W, 1).contiguous()

        predict_pointmap[0] = predict_pointmap[0].view(B, S, H, W, C).contiguous()
        predict_pointmap[1] = predict_pointmap[1].view(B, S, H, W, 1).contiguous()
        assert predict_depth[0].shape[-1] == 1
        assert predict_pointmap[0].shape[-1] == 3
        return predict_depth, predict_pointmap

    def forward(self, rgb, pred_depth, intrinsic_depth):
        features = self.encoder(rgb, pred_depth, intrinsic_depth)
        predict_depth, predict_pointmap = self.decoder(features, rgb)
        return predict_depth, predict_pointmap
    
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

    def loss(self, rgb, pred_depth, depth, intrinsic_depth, weights = [0.01,0.01,1], mask=False):
        BS, H, W, C = depth.shape
        S = self.args.context_length
        assert BS % S == 0, "Batch size must be divisible by context length"
        B = BS // S
        predict_depth, predict_pointmap = self.forward(rgb, pred_depth, intrinsic_depth)
        depth = depth.view(B, S, H, W, C).contiguous()
        assert depth.shape[-1] == 1
        assert intrinsic_depth.shape[-1] == 4
        assert intrinsic_depth.shape[-2] == 4

        loss_pointmap = ConfAlignPointMapRegLoss(depth, predict_pointmap, intrinsic_depth, weights[0])
        loss_depth = ConfAlignDepthRegLoss(depth, predict_depth,  weights[1])

        return loss_pointmap + loss_depth
    
    def eval_loss(self, predict_depth, predict_pointmap, depth, intrinsic_depth, mask=False):
        BS, H, W, C = depth.shape
        S = self.args.context_length
        assert BS % S == 0, "Batch size must be divisible by context length"
        B = BS // S
        depth = depth.view(B, S, H, W, C).contiguous()
        assert depth.shape[-1] == 1
        assert intrinsic_depth.shape[-1] == 4
        assert intrinsic_depth.shape[-2] == 4
        loss_pointmap = ConfAlignPointMapRegLoss(depth, predict_pointmap, intrinsic_depth, alpha = args.alpha_pointmap, eps=args.eps)
        loss_depth = ConfAlignDepthRegLoss(depth, predict_depth,  alpha = args.alpha_depth, eps=args.eps)
        return loss_pointmap, loss_depth

def get_pointmap(depth):
    B, K, H, W, _ = depth.shape # as C is 1
    Z = depth[:,:,:,:, 0]  # Use ground truth depth for point map calculation

    prediction_pm = depth #.squeeze(1)
    u = torch.arange(W, device=prediction_pm.device).view(1, 1, 1, W).expand(B, K, H, W).contiguous()
    v = torch.arange(H, device=prediction_pm.device).view(1, 1, H, 1).expand(B, K, H, W).contiguous()

    intrinsics = intrinsics.view(B, K, 4, 4).contiguous()  # Reshape to [B, K, 4, 4]
    fx = intrinsics[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
    fy = intrinsics[:, :, 1, 1].unsqueeze(-1).unsqueeze(-1)
    cx = intrinsics[:, :, 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = intrinsics[:, :, 1, 2].unsqueeze(-1).unsqueeze(-1)

    # Broadcast to shape (B, K, H, W)
    fx = fx.expand(B, K, H, W)
    fy = fy.expand(B, K, H, W)
    cx = cx.expand(B, K, H, W)
    cy = cy.expand(B, K, H, W)

    # Compute X, Y
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    pointmap = torch.stack([X, Y, Z], dim=-1)

class Mov3r:
    def __init__(self, args):
        self.num_epochs = args.num_epochs
        self.dataset_path = args.dataset_path
        self.batch_size = args.batch_size
        self.model_dir = args.model_dir
        self.ckpt_dir = args.ckpt_dir
        self.run_name = args.run_name
        self.eval_dir = args.save_evals
        self.dist =  None
        self.log_dir = args.log_dir #+ args.run_name
        self.pc_dec_depth = args.pc_dec_depth
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join(self.log_dir, self.eval_dir + "_" + self.run_name )
        self.log_file = os.path.join(self.log_dir, f"{timestamp}.log")
        self.transforms = data_transforms

        if not os.path.exists(self.log_dir):
            os.makedirs(os.path.join(self.log_dir))
            
        print(f"Log file: {self.log_file}")
        # self.log_file = open(self.log_file, 'w')
        # sys.stdout = self.log_file
        # sys.stderr = self.log_file

        # logging.basicConfig(
        #     level=logging.INFO,
        #     format='%(asctime)s - %(levelname)s - %(message)s',
        #     handlers=[
        #         logging.StreamHandler(self.log_file)
        #     ]
        # )
        # logging.info("Starting evaluating...")
        # logging.info(f"Arguments: {args}")

        #Preprocess and load dataset to memory
        self.buffer_scene = BufferedSceneDataset(
            root_dir=args.dataset_path,
            max_scenes=args.max_scenes,
            num_workers=args.ds_num_workers,
            num_frames=args.context_length,
            frame_skip=args.frame_skip,
            data_transforms=None
        )
        # now = datetime.datetime.now()
        # timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        # self.log_dir = os.path.join(self.log_dir, f"{self.run_name}_{timestamp}")

        if not os.path.exists(os.path.join("evals_vis", self.run_name)):
            os.makedirs(os.path.join("evals_vis", self.run_name))

        if not os.path.exists(self.log_dir): #and local_rank == 0:
            os.makedirs(self.log_dir)
    
        self.ckpt_dir = os.path.join(self.ckpt_dir, self.run_name) #self.ckpt_dir + self.run_name
        if not os.path.exists(self.ckpt_dir):# and local_rank == 0:
            os.makedirs(self.ckpt_dir)
        
        self.model_dir = os.path.join(self.model_dir, self.run_name)
        if not os.path.exists(self.model_dir): # and local_rank == 0:
            os.makedirs(self.model_dir)

        self.writer = tensorboardX.SummaryWriter(self.log_dir)

        #create the model
        self.model = UnifiedModel(args)
        if args.load_model:
            self.load_model()
        self.model.load_depth_embedder(self.device)
        self.model.load_dino_patch_embed(self.device)
        self.model.to(self.device)
        
        #distributed training
        # self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        #optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        #scheduler 
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=args.gamma)

        #avg loss
        self.avg_loss = 0

        torch.cuda.empty_cache()
        gc.collect()

    def create_eval_loader(self, dataset):
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=args.dl_num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True
        )
        return data_loader
   
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


    def load_checkpoint(self, args):
        checkpoint = torch.load(args.load_from_ckpt, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'], strict=False)
        epoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        avg_loss = checkpoint['avg_loss']
        print(f'Loaded checkpoint from {args.load_from_ckpt} at epoch {epoch} with avg_loss {avg_loss}')
        return epoch

    def float32_to_uint8(self, img):
        img_min = np.min(img)
        img_max = np.max(img)
        img_norm = (img - img_min) / (img_max - img_min + 1e-8)  # Avoid division by zero
        img_uint8 = (img_norm * 255).astype(np.uint8)
        return img_uint8

    def save_image(self, img, folder_name, img_name, idx, is_depth=False):
        file_name = os.path.join(folder_name, img_name + "_" + str(idx).zfill(4) + ".png")
        if is_depth:
            plt.imshow(img[:,:,0], aspect='auto', cmap='viridis')
            plt.colorbar(label='Measured Value')
            plt.xlabel('Time')
            plt.ylabel('Depth')
            plt.title('Depth Heatmap')    
            im = Image.fromarray(img[:,:,0], mode="L")
            plt.savefig(file_name, bbox_inches='tight')
            plt.close()
        else:
            # img=img[:, :, ::-1] # bgr to rgb
            im = Image.fromarray(img)
            im.save(file_name)


    # def save_image(self, img, folder_name, idx, modality, img_type, good=True):
    
    #     if good:
    #         name = "good_" + modality + "_" + str(idx).zfill(4) + "_" + str(img_type) +".png"
    #     else:
    #         name = "bad_" + modality + "_" + str(idx).zfill(4) + "_" + str(img_type) + ".png"

    #     file_name = os.path.join(folder_name, name)
    #     if "depth" in modality or "depth" in img_type:
    #         plt.imshow(img[:,:,0], aspect='auto', cmap='viridis')
    #         plt.colorbar(label='Measured Value')
    #         plt.xlabel('Time')
    #         plt.ylabel('Depth')
    #         plt.title('Depth Heatmap')    
    #         im = Image.fromarray(img[:,:,0], mode="L")
    #         plt.savefig(file_name, bbox_inches='tight')
    #         plt.close()
    #     else:
    #         img=img[:, :, ::-1]
    #         im = Image.fromarray(img)
    #         im.save(file_name)


    def eval(self, return_sum=True, visualize_results=False):
        '''
        eval_batch: if metrics for the entire batch is needed at once. This is the preferred option as it's quick.
        eval_batch=False: returns the frame indices with the 5 highest and 5 lowest depth/posemap losses and saves
        them to ./evals_vis.

        This returns the sum of the depth and pointmap loss.
        '''
        print(f"return sum: {return_sum} and visualize results: {visualize_results}")
        # Set the model to evaluation mode
        self.model.eval()

        # Load the evaluation dataset
        # eval_dataset = self.buffer_scene.fetch_eval_dataset()
        eval_dataset = self.buffer_scene.fetch_dataset()
        # eval_loader, _ = self.create_distributed_loader(eval_dataset)
        eval_loader = self.create_eval_loader(eval_dataset)


        # Evaluation loop
        list_of_loss = []
        # list_of
        # list_of_depth_loss=[]
        # list_of_pointmap_loss=[]
        dict_of_depth = {}
        dict_of_pointmap = {}
        dict_of_depth_gt ={}
        dict_of_rgb = {}

        return_loss = 0
        return_loss_pointmap = 0
        return_loss_depth = 0

    # Evaluation loop
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(eval_loader), desc='Evaluating...', bar_format='{l_bar}{bar:20}{r_bar}', leave=True, disable=False):
                rgb, depth, pred_depth, pose, intrinsic = (batch["rgb"].to(self.device, non_blocking=True), batch["depth"].to(self.device, non_blocking=True),
                                                    batch["pred_depth"].to(self.device, non_blocking=True), batch["pose"].to(self.device, non_blocking=True), batch["intrinsics"])
                # Perform evaluation here
                rgb = rgb.to(torch.float32, non_blocking=True)/255.0
                depth = depth.to(torch.float32, non_blocking=True)/1000.0
                pred_depth = pred_depth.to(torch.float32, non_blocking=True)/255
                pose = pose.to(torch.float32, non_blocking=True)
                intrinsic = {k: v.to(torch.float32, non_blocking=True) for k, v in intrinsic.items()}

                if self.transforms is not None:
                    # Apply the transformations
                    rgb = self.transforms(rgb)
                
                # rgb shape [B, S, C, H, W]
                _, _, C, H, W = rgb.shape 
                
                #Collapsed the batch and sequence dimension
                rgb = rgb.view(-1, C, H, W).contiguous() # BS, C, H, W

                #permuting the channel dimension to the front
                rgb = rgb.permute(0, 2, 3, 1).contiguous() # BS, H, W, C
                
                #collapsed the batch and sequence dimension
                depth = depth.view(-1, H, W, 1).contiguous() # BS, H, W, 1
                pred_depth = pred_depth.view(-1, H, W, 1).contiguous() # BS, H, W, 1

                assert rgb.shape[-1] == 3
                assert depth.shape[-1] == 1
                assert pred_depth.shape[-1] == 1

                intrinsic_depth = intrinsic['intrinsic_depth'].to(self.device, non_blocking=True)
                intrinsic_depth = intrinsic_depth.repeat_interleave(repeats=args.context_length, dim=0)

                # Forward pass
                predict_depth, predict_pointmap = self.model(rgb, pred_depth, intrinsic_depth)
                pointmap_loss, depth_loss = self.model.eval_loss(predict_depth, predict_pointmap, depth, intrinsic_depth)

                return_loss_pointmap+=pointmap_loss
                return_loss_depth+=depth_loss
                return_loss+=(pointmap_loss+depth_loss)

            
                # return pointmap_loss, depth_loss
                if visualize_results:
                    list_of_loss.append([pointmap_loss, depth_loss, batch_idx])
                    dict_of_depth[batch_idx] = predict_depth[0].squeeze(0).squeeze(0).cpu().numpy()
                    dict_of_pointmap[batch_idx] = predict_pointmap[0].squeeze(0).squeeze(0).cpu().numpy()
                    dict_of_depth_gt[batch_idx] = depth[0].squeeze(0).cpu().numpy()
                    dict_of_rgb[batch_idx] = rgb.squeeze(0).squeeze(0).cpu().numpy()

            if return_sum:
                return return_loss_pointmap + return_loss_depth
            # create separate folders for rgb, depth, gt depth, pointmap
            if visualize_results: 
                list_of_sorted_loss_pointmap = sorted(list_of_loss, key=lambda x: x[0])
                list_of_sorted_loss_depth = sorted(list_of_loss, key=lambda x: x[1])
                # should save the following files
                # run_name/good,bad_depth_idx_rgb
                # run_name/good,bad_depth_idx_pointmap
                # run_name/good,bad_depth_idx_gt_depth
                # run_name/good,bad_pointmap_idx_rgb
                # run_name/good,bad_pointmap_idx_depth
                # run_name/good,bad_pointmap_idx_gt_depth
                # visualizations = ["depth", "rgb", "pointmap", "gt_depth"]

                folder_name = os.path.join("evals_vis", self.run_name)
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name) 

                with open(os.path.join(folder_name, 'metrics.txt'), 'w') as f:

                    for idx in range(5): 
                        # import pdb
                        # pdb.set_trace()
                        good_pointmap_loss = list_of_sorted_loss_pointmap[idx][0]

                        good_pointmap_img = self.float32_to_uint8(dict_of_pointmap[list_of_sorted_loss_pointmap[idx][-1]])
                        good_pointmap_depth = self.float32_to_uint8(dict_of_depth[list_of_sorted_loss_pointmap[idx][-1]])
                        good_pointmap_depth_gt = self.float32_to_uint8(dict_of_depth_gt[list_of_sorted_loss_pointmap[idx][-1]])
                        good_pointmap_rgb = self.float32_to_uint8(dict_of_rgb[list_of_sorted_loss_pointmap[idx][-1]])
                        
                        # def save_image(self, img, folder_name, idx, img_name, is_depth=False):

                        self.save_image(good_pointmap_img, folder_name, "good_pointmap_img", idx, is_depth=False)
                        self.save_image(good_pointmap_depth, folder_name, "good_pointmap_depth", idx, is_depth=True)
                        self.save_image(good_pointmap_depth_gt, folder_name, "good_pointmap_depth_gt", idx, is_depth=True)
                        self.save_image(good_pointmap_rgb, folder_name, "good_pointmap_rgb", idx, is_depth=False)


                        bad_pointmap_loss = list_of_sorted_loss_pointmap[-idx - 1][0]

                        bad_pointmap_img = self.float32_to_uint8(dict_of_pointmap[list_of_sorted_loss_pointmap[-idx - 1][-1]])
                        bad_pointmap_depth = self.float32_to_uint8(dict_of_depth[list_of_sorted_loss_pointmap[-idx - 1][-1]])
                        bad_pointmap_depth_gt = self.float32_to_uint8(dict_of_depth_gt[list_of_sorted_loss_pointmap[-idx - 1][-1]])
                        bad_pointmap_rgb = self.float32_to_uint8(dict_of_rgb[list_of_sorted_loss_pointmap[-idx - 1][-1]])


                        self.save_image(bad_pointmap_img, folder_name, "bad_pointmap_img", idx, is_depth=False)
                        self.save_image(bad_pointmap_depth, folder_name, "bad_pointmap_depth", idx, is_depth=True)
                        self.save_image(bad_pointmap_depth_gt, folder_name, "bad_pointmap_depth_gt", idx, is_depth=True)
                        self.save_image(bad_pointmap_rgb, folder_name, "bad_pointmap_rgb", idx, is_depth=False)


                        # self.save_image(bad_pointmap_img, folder_name, idx, "pointmap", "img", good=False)
                        # self.save_image(bad_pointmap_depth, folder_name, idx, "pointmap", "depth", good=False)
                        # self.save_image(bad_pointmap_depth_gt, folder_name, idx, "pointmap", "depth_gt", good=False)
                        # self.save_image(bad_pointmap_rgb, folder_name, idx, "pointmap", "rgb", good=False)


                        good_depth_loss = list_of_sorted_loss_depth[idx][0]

                        good_depth_img = self.float32_to_uint8(dict_of_pointmap[list_of_sorted_loss_depth[idx][-1]])
                        good_depth_pointmap = self.float32_to_uint8(dict_of_pointmap[list_of_sorted_loss_depth[idx][-1]])
                        good_depth_depth_gt = self.float32_to_uint8(dict_of_depth_gt[list_of_sorted_loss_depth[idx][-1]])
                        good_depth_rgb = self.float32_to_uint8(dict_of_rgb[list_of_sorted_loss_depth[idx][-1]])


                        self.save_image(good_depth_img, folder_name, "good_depth_img", idx, is_depth=True)
                        self.save_image(good_depth_pointmap, folder_name, "good_depth_pointmap", idx, is_depth=False)
                        self.save_image(good_depth_depth_gt, folder_name, "good_depth_depth_gt", idx, is_depth=True)
                        self.save_image(good_depth_rgb, folder_name, "good_depth_rgb", idx, is_depth=False)

                        bad_depth_loss = list_of_sorted_loss_depth[-idx - 1][0]

                        bad_depth_img = self.float32_to_uint8(dict_of_depth[list_of_sorted_loss_depth[-idx - 1][-1]])
                        bad_depth_pointmap = self.float32_to_uint8(dict_of_pointmap[list_of_sorted_loss_depth[-idx - 1][-1]])
                        bad_depth_depth_gt = self.float32_to_uint8(dict_of_depth_gt[list_of_sorted_loss_depth[-idx - 1][-1]])
                        bad_depth_rgb = self.float32_to_uint8(dict_of_rgb[list_of_sorted_loss_depth[-idx - 1][-1]])


                        self.save_image(bad_depth_img, folder_name, "bad_depth_img", idx, is_depth=True)
                        self.save_image(bad_depth_pointmap, folder_name, "bad_depth_pointmap", idx, is_depth=False)
                        self.save_image(bad_depth_depth_gt, folder_name, "bad_depth_depth_gt", idx, is_depth=True)
                        self.save_image(bad_depth_rgb, folder_name, "bad_depth_rgb", idx, is_depth=False)
                        # bad_pointmap_img = self.float32_to_uint8(dict_of_pointmap[list_of_sorted_loss_pointmap[-idx][-1]])
                        # bad_gt_pointmap_depth = self.float32_to_uint8(dict_of_gt_depth[list_of_sorted_loss_pointmap[-idx][-1]])

                        # good_depth_loss = list_of_sorted_loss_depth[idx][0]
                        # good_depth_img = self.float32_to_uint8(dict_of_depth[list_of_sorted_loss_depth[idx][-1]])
                        # good_gt_depth = self.float32_to_uint8(dict_of_gt_depth[list_of_sorted_loss_depth[idx][-1]])

                        # bad_depth_loss = list_of_sorted_loss_depth[-idx - 1][0]
                        # bad_depth_img = self.float32_to_uint8(dict_of_depth[list_of_sorted_loss_depth[-idx][-1]])
                        # bad_gt_depth = self.float32_to_uint8(dict_of_gt_depth[list_of_sorted_loss_depth[-idx][-1]])



                        # self.save_image(good_pointmap_img, folder_name, idx, "pointmap", good=True)
                        # self.save_image(bad_pointmap_img, folder_name, idx, "pointmap", good=False)

                        # self.save_image(good_depth_img, folder_name, idx, "depth", good=True)
                        # self.save_image(bad_depth_img, folder_name, idx, "depth", good=False)

                        # self.save_image(good_gt_pointmap_depth, folder_name, idx, "gt_pointmap_depth", good=False)
                        # self.save_image(bad_gt_pointmap_depth, folder_name, idx, "gt_pointmap_depth", good=False)

                        # self.save_image(good_gt_depth, folder_name, idx, "gt_depth", good=True)
                        # self.save_image(bad_gt_depth, folder_name, idx, "gt_depth", good=False)

                        print(f"For idx {idx}, the good pointmap loss is: {good_pointmap_loss}, the bad pointmap loss: {bad_pointmap_loss} \nthe good depth loss is: {good_depth_loss}, the bad depth loss is: {bad_depth_loss}", file=f)
                    f.close()
                print("\n=============== Evaluation Metrics ========================")
                print("Average PointMap Loss : ", return_loss_pointmap/len(eval_loader))
                print("Average Depth Loss : ", return_loss_depth/len(eval_loader))
                print("Average Loss: ", return_loss/len(eval_loader))
                return (return_loss_pointmap, return_loss_depth, return_loss)
                
                
if __name__ == "__main__":
    args = get_config()

    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    # Use first available GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    # Define transforms (same as original)
    data_transforms = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Initialize model
    mov3r = Mov3r(args)  # Remove 'dist' parameter if not needed

    # Load checkpoint (handle DDP -> single GPU conversion if needed)
    # checkpoint = torch.load(args.load_from_ckpt, map_location=device)
    # if any(k.startswith('module.') for k in checkpoint["model"].keys()):
    #     checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    mov3r.load_checkpoint(args)
    
    # evaluate
    print(f"ret sum: {args.return_sum} and vis res: {args.visualize_results}")
    loss = mov3r.eval(args.return_sum, args.visualize_results)