
# from scannet_dataset import ScanNetV2Dataset
# from scannet_dataset import ScannetFromMemory
# from torch.utils.data import DataLoader
# from torchvision import transforms
from depth import generate_monocular_depth_map
import os
import numpy as np
import cv2


# Example transformation (you can customize this as needed)
# data_transforms = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Create dataset
# dataset = ScanNetV2Dataset(root_dir='/data/kmirakho/l3dProject/scannetv2', transforms=data_transforms, rgb_only=True)
# memory = dataset._load_data_into_memory()

# print(memory[0][0]['rgb'].shape)
# scannet_memory = ScannetFromMemory(memory)
# # Create DataLoader
# dataloader = DataLoader(scannet_memory, batch_size=5, shuffle=True, num_workers=4)

# for i in len(dataloader):
#     batch = next(iter(dataloader))
#     for i in range(len(batch)):
#         for j in range(len(batch[i])):
#             # Save RGB image
#             rgb = batch[i][j]['rgb']
        
#             # Generate depth map
#             depth_map = generate_monocular_depth_map(rgb)
            
#             # Save depth map
#             # output_path = os.path.join('output_depth_maps', f'depth_map_{i}.png')
#             # os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             # depth_map.save(output_path)

#             #save depth map as npz
#             output_path = os.path.join('output_depth_maps', f'depth_map_{i}_{j}.npz')
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             np.savez_compressed(output_path, depth_map=depth_map)
            
#             # print(f"Saved depth map to {output_path}")
#     break

root_dir = '/data/kmirakho/l3dProject/scannetv2'
for scene_dir in os.listdir(root_dir):
    print(f"Processing scene: {scene_dir}")
    for frame in os.listdir(os.path.join(root_dir, scene_dir, 'color')):
        filename = os.path.splitext(frame)[0]
        output_path = os.path.join('output_depth_maps', scene_dir, f'{filename}.npz')
        if os.path.exists(output_path):
            print(f"Skipping {output_path} as it already exists.")
            continue
        # Load RGB image
        rgb_image = cv2.imread(os.path.join(root_dir, scene_dir, 'color' ,frame))
        depth = generate_monocular_depth_map(rgb_image)

        depth = depth.cpu().detach().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, depth=depth)

