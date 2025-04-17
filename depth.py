# read image 
# compute depth by depth anything v2 
# save depth

# encode depth by vit point encoder

import numpy as np
from tqdm import tqdm
import cv2
import torch
import torch.nn.functional as F
import matplotlib
from functools import partial

from depth_anything_v2.dpt import DepthAnythingV2

from croco.models.blocks import PatchEmbed, Block
from croco.models.pos_embed import RoPE2D

def generate_monocular_depth_map(imgs, encoder='vits', input_size=512):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(f'depth_anything_v2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    imgs, (h,w) = depth_anything.image2tensor(imgs, input_size) # image passing rawimages
    
    depth = depth_anything(imgs)
    depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]

    return depth

class PatchEmbedDust3R(PatchEmbed):
    def forward(self, x, **kw):
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({self.patch_size[0]})."
        assert W % self.patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({self.patch_size[1]})."
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2).contiguous()  # BCHW -> BNC
        x = self.norm(x)
        return x, pos

class DepthEmbedder(torch.nn.Module):
    def __init__(self, patch_embed_cls, img_size, patch_size, dec_embed_dim, pos_embed, pc_dec_depth):
        super().__init__()

        self.patch_embed_cls = patch_embed_cls
        self.patch_embed_point_cloud = self.get_patch_embed(patch_embed_cls, img_size, patch_size, dec_embed_dim)
        self.pc_dec_depth=pc_dec_depth

        self.pos_embed = pos_embed
        if self.pos_embed=='cosine':
            self.rope = None
        elif self.pos_embed.startswith('RoPE'):
            freq = float(self.pos_embed[len('RoPE'):])
            self.rope = RoPE2D(freq=freq)

        self.dec_blocks_pc = torch.nn.ModuleList([
            Block(dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), rope=self.rope)
            for i in range(self.pc_dec_depth//2-2)])

    def get_patch_embed(self, patch_embed_cls, img_size, patch_size, dec_embed_dim):
        assert patch_embed_cls in ['PatchEmbedDust3R', 'ManyAR_PatchEmbed']
        patch_embed = eval(patch_embed_cls)(img_size, patch_size, 3, dec_embed_dim)
        return patch_embed

    def pixel_to_pointcloud(self, depth_map, focal_length_px):
        """
        Convert batched depth maps to 3D point clouds.

        Args:
            depth_map (torch.Tensor): Depth maps of shape (B, H, W)
            focal_length_px (torch.Tensor): Focal lengths of shape (B,)

        Returns:
            torch.Tensor: Point clouds of shape (B, H, W, 3)
        """
        B, _, H, W = depth_map.shape
        device = depth_map.device

        # Create pixel coordinate grid
        u = torch.arange(W, device=device).view(1, 1, W).expand(B, H, W)
        v = torch.arange(H, device=device).view(1, H, 1).expand(B, H, W)

        # Compute principal point (center of image)
        cx = W / 2.0
        cy = H / 2.0

        # Reshape focal_length to broadcast with image grid
        f = focal_length_px.view(B, 1, 1)

        # Convert to camera coordinates
        Z = depth_map
        X = (u - cx) * Z / f
        Y = (v - cy) * Z / f
        
        point_cloud = torch.cat((X, Y, Z), dim=-1)  # (B, H, W, 3)
        # Optional: Normalize per-point cloud or apply further filtering
        # point_cloud = normalize_pointcloud(point_cloud)

        return point_cloud

    def normalize_pointcloud(self, point_cloud):
        min_vals = np.min(point_cloud, axis=(0, 1))
        max_vals = np.max(point_cloud, axis=(0, 1))
        #print(min_vals, max_vals)
        normalized_point_cloud = (point_cloud - min_vals) / (max_vals - min_vals)
        return normalized_point_cloud

    def forward(self, x, intrinsic_depth):
        focal_length_px = intrinsic_depth[:, 0, 0]
        #unproject depth to point cloud
        point_cloud = self.pixel_to_pointcloud(x, focal_length_px)
        point_cloud = point_cloud.permute(0,3,1,2)
        x, pos = self.patch_embed_point_cloud(point_cloud)

        for i in range(len(self.dec_blocks_pc)):
            x = self.dec_blocks_pc[i](x, pos)
        return x

if __name__ == "__main__":
    import os
    import glob
    import argparse

    test_depth_estimation = False

    test_depth_embedder = True

    if test_depth_estimation:
        parser = argparse.ArgumentParser(description="Depth Estimation")
        parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing images")
        parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory to save depth maps")
        args = parser.parse_args()

        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Get list of images in the input directory
        img_list = glob.glob(os.path.join(args.input_dir, '*.png')) + glob.glob(os.path.join(args.input_dir, '*.jpg'))
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        for image_path in tqdm(img_list):
            path_depthanything = os.path.join(args.output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.png')
            image = cv2.imread(image_path)
            depth_map = generate_monocular_depth_map(image)
            depth = depth.cpu().detach().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth_map.astype(np.uint8)
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            # import pdb;pdb.set_trace()
            cv2.imwrite(path_depthanything, depth)
            print(f"Depth map saved to {path_depthanything}")

    if test_depth_embedder:
        depth= torch.randn(2, 224, 224, 1).cuda()
        flen = torch.randn(2, 4, 4).cuda()
        
        depth_embedder = DepthEmbedder(patch_embed_cls='PatchEmbedDust3R', img_size=224, patch_size=16, dec_embed_dim=768, pos_embed='cosine', pc_dec_depth=8)
        depth_embedder = depth_embedder.cuda()
        
        depth_embedding = depth_embedder(depth, flen)
        print(depth_embedding.shape)