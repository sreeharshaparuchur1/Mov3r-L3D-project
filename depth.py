# read image 
# compute depth by depth anything v2 
# save depth

import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
import matplotlib
from depth_anything_v2.dpt import DepthAnythingV2

def generate_monocular_depth_map(imgs, depth_prior_name='depth_anything_v2', encoder='vits', input_size=512, grayscale=False):
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(f'depth_anything_v2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(imgs.device).eval()

    depth = depth_anything.infer_image(imgs, input_size)
        
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    
    if grayscale:
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    else:
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    # pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf",device='cuda')
    # depth_map = pipe(imgs)["predicted_depth"].numpy()
    return depth_map


if __name__ == "__main__":
    import os
    import glob
    import argparse

    parser = argparse.ArgumentParser(description="Depth Estimation")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory to save depth maps")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Get list of images in the input directory
    img_list = glob.glob(os.path.join(args.input_dir, '*.png')) + glob.glob(os.path.join(args.input_dir, '*.jpg'))

    for image_path in tqdm(img_list):
        path_depthanything = os.path.join(args.output_dir, os.path.basename(image_path))
        image = Image.open(image_path)
        image = transforms.ToTensor()(image).unsqueeze(0)
        image = image.to(device=DEVICE)
        depth_map = generate_monocular_depth_map(image)
        depth_map = Image.fromarray(depth_map)
        depth_map.save(path_depthanything)