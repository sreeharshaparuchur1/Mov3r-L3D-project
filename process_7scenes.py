'''
scene-num
    - frame-num.color.png
    - frame-num.depth.png
    - frame-num.pose.txt

    
scene-num
    - color
        - num.jpg
    - depth
        - num.png
    - pose
        - num.txt
    - intrinsic
        - intrinsic_depth.txt
'''

import os
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
from depth_anything_v2.dpt import DepthAnythingV2
import dust3r.datasets.utils.cropping as cropping
import pdb
input_dir = "/data/kmirakho/l3d_proj/7scenes/stairs"
output_dir = "/data/kmirakho/l3d_proj/7scenes_processed/stairs"

output_resolution=(224, 224)
intrinsic_depth = np.array([[ 585.0, 0.0, 320.0, 0.0],
                            [ 0.0, 585.0, 240.0, 0.0],
                            [ 0.0, 0.0, 1.0, 0.0],
                            [ 0.0, 0.0, 0.0, 1.0]])
def load_depth_anything():
    # Load the model
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder='vits'
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(f'depth_anything_v2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(device).eval()
    return depth_anything

def generate_monocular_depth_map(depth_anything, imgs, input_size=512):    
    imgs, (h,w) = depth_anything.image2tensor(imgs, input_size) # image passing rawimages
    depth = depth_anything(imgs)
    depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
    return depth


def crop_resize_if_necessary(image, depthmap, pred_depth, intrinsics, resolution, rng=None, info=None):
    """ This function:
        - first downsizes the image with LANCZOS inteprolation,
            which is better than bilinear interpolation in
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # downscale with lanczos interpolation so that image.size == resolution
    # cropping centered on the principal point
    W, H = image.size
    cx, cy = intrinsics[:2, 2].round().astype(int)
    min_margin_x = min(cx, W - cx)
    min_margin_y = min(cy, H - cy)
    # assert min_margin_x > W/5, f'Bad principal point in view={info}'
    # assert min_margin_y > H/5, f'Bad principal point in view={info}'
    
    # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
    # l, t = cx - min_margin_x, cy - min_margin_y
    # r, b = cx + min_margin_x, cy + min_margin_y
    # crop_bbox = (l, t, r, b)
    # image, depthmap, pred_depth, intrinsics = cropping.crop_image_depthmap(image, depthmap, pred_depth, intrinsics, crop_bbox)

    # transpose the resolution if necessary
    W, H = image.size  # new size
    assert resolution[0] >= resolution[1]
    if H > 1.1 * W:
        # image is portrait mode
        resolution = resolution[::-1]
    elif 0.9 < H / W < 1.1 and resolution[0] != resolution[1]:
        # image is square, so we chose (portrait, landscape) randomly
        if rng.integers(2):
            resolution = resolution[::-1]

    # high-quality Lanczos down-scaling
    target_resolution = np.array(resolution)
    # if self.aug_crop > 1:
    #     target_resolution += rng.integers(0, self.aug_crop)
    image, depthmap, pred_depth, intrinsics = cropping.rescale_image_depthmap(image, depthmap, pred_depth, intrinsics, target_resolution)

    # actual cropping (if necessary) with bilinear interpolation
    intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
    crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
    image, depthmap, pred_depth, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, pred_depth[:, :, None], intrinsics, crop_bbox)

    return image, depthmap, pred_depth, intrinsics2


if __name__ == "__main__":
    # check if output_dir exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # check if input_dir exists
    if not os.path.exists(input_dir):
        raise Exception(f"Input directory {input_dir} does not exist")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    depth_anything = load_depth_anything()

    # loop through all scenes in the input directory
    for scenes in tqdm(os.listdir(input_dir)):
        if not os.path.isdir(os.path.join(input_dir, scenes)):
            continue
        
        scene_num = scenes.split("-")[1]
        if not os.path.exists(os.path.join(output_dir, scene_num)):
            os.makedirs(os.path.join(output_dir, scene_num))
            os.makedirs(os.path.join(output_dir, scene_num, "color"))
            os.makedirs(os.path.join(output_dir, scene_num, "depth"))
            os.makedirs(os.path.join(output_dir, scene_num, "pose"))
            os.makedirs(os.path.join(output_dir, scene_num, "intrinsics"))
            os.makedirs(os.path.join(output_dir, scene_num, "depth_any"))

        for frames in os.listdir(os.path.join(input_dir, scenes)):
            
            if frames.endswith("color.png"):    
                frame_num = frames.split(".")[0]
                frame_num = frame_num.split("-")[1]
                # rgb frame
                rgb_path = os.path.join(input_dir, scenes, "frame-" + frame_num + ".color.png")
                rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)               
            
                # depth frame
                depth_path = os.path.join(input_dir, scenes, "frame-" + frame_num + ".depth.png")
                depthmap = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)#.astype(np.float32)
                depthmap[~np.isfinite(depthmap)] = 0
                depthmap[depthmap>50000] = 0 #max depth in 7scenes is 65535
                # import pdb; pdb.set_trace()

                estimate_depth = generate_monocular_depth_map(depth_anything, rgb_img).cpu().detach().numpy()
                estimate_depth = (estimate_depth - estimate_depth.min()) / (estimate_depth.max() - estimate_depth.min())* 255.0
                estimate_depth = estimate_depth.astype(np.uint8)

                estimate_depth = 255.0 - estimate_depth
                estimate_depth = estimate_depth[:, :, None]
                estimate_depth[~np.isfinite(estimate_depth)] = 0
                # resize
                
                rgb_img_resized, depthmap_resized, estimate_depth_resized, intrinsic_depth_resized = crop_resize_if_necessary(rgb_img, depthmap, pred_depth=estimate_depth, intrinsics=intrinsic_depth, resolution=output_resolution)

                #save rgb as jog
                rgb_img_resized.save(os.path.join(output_dir, scene_num, "color", frame_num + ".jpg"))
                # cv2.imwrite(os.path.join(output_dir, scene_num, "color", frame_num, ".jpg"), rgb_img_resized)
                #save depth as png
                # depthmap_resized = depthmap_resized
                # depthmap_resized = depthmap_resized.astype(np.uint8)
                # Image.fromarray(depthmap_resized).save(os.path.join(output_dir, scene_num, "depth", frame_num + ".png"))
                
                cv2.imwrite(os.path.join(output_dir, scene_num, "depth", frame_num + ".png"), depthmap_resized)
                #save intrinsic depth as txt
                np.savetxt(os.path.join(output_dir, scene_num, "intrinsics", "intrinsic_depth.txt"), intrinsic_depth_resized)

                #save as npz
                np.savez_compressed(os.path.join(output_dir, scene_num, "depth_any", frame_num), depth=estimate_depth_resized)
                # pose frame
                pose_path = os.path.join(input_dir, scenes, "frame-" + frame_num + ".pose.txt")
                pose = np.loadtxt(pose_path)
                np.savetxt(os.path.join(output_dir, scene_num, "pose", frame_num + ".txt"), pose)
