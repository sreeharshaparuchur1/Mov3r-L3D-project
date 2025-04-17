from depth import generate_monocular_depth_map
import os
import numpy as np
import cv2

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
from tqdm import tqdm

def generate_monocular_depth_map(imgs, depth_anything, input_size=512):    
    imgs, (h,w) = depth_anything.image2tensor(imgs, input_size) # image passing rawimages
    depth = depth_anything(imgs)
    depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
    return depth


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder='vits'
depth_anything = DepthAnythingV2(**model_configs[encoder])
depth_anything.load_state_dict(torch.load(f'depth_anything_v2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
depth_anything = depth_anything.to(DEVICE).eval()

root_dir = '/data/kmirakho/l3dProject/scannetv2'
scene_after = 707

for scene_dir in tqdm(os.listdir(root_dir), desc="Processing scenes"):
    for i, c in enumerate(scene_dir):
        if c.isdigit():
            break
    scene = scene_dir[:i]                 # 'scene'
    num_part = scene_dir[i:]              # '0316_00'

    # Split numbers and convert to int
    nums = list(map(int, num_part.split('_')))  # [316, 0
    if nums[0] < scene_after:
        continue
    
    print(f"Processing scene: {scene_dir}")
    for frame in os.listdir(os.path.join(root_dir, scene_dir, 'color')):
        filename = os.path.splitext(frame)[0]
        output_path = os.path.join('output_depth_maps', scene_dir, f'{filename}.npz')
        if os.path.exists(output_path):
            # print(f"Skipping {output_path} as it already exists.")
            continue
        # Load RGB image
        rgb_image = cv2.imread(os.path.join(root_dir, scene_dir, 'color' ,frame))
        depth = generate_monocular_depth_map(rgb_image,depth_anything)
        depth = depth.cpu().detach().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, depth=depth)

