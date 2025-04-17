import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from glob import glob
from functools import partial
import multiprocessing as mp
from tqdm import tqdm
import gc

class ScanNetV2Dataset(Dataset):
    def __init__(self, root_dir, num_frames=20, transforms=None, train = True, rgb_only=False):
        """
        Args:
            root_dir (string): Directory with all the ScanNet V2 data.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.num_frames = num_frames
        self.rgb_only = rgb_only
        self.to_tensor = transforms.ToTensor()
        # List all scene directories
        self.scene_dirs = glob(os.path.join(root_dir,'*'))[:1]
        self.num_scenes = len(self.scene_dirs) 
        print(f"Number of scenes: {self.num_scenes}")

    def __len__(self):
        return self.num_scenes
    
    def _load_data_into_memory(self):
        num_workers = max(1, mp.cpu_count() - 1)  # Use all CPUs except one

        print(f"Loading dataset using {num_workers} processes...")
        try:
            with mp.Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.imap(self.__getitem__, self.scene_dirs[:1]),
                    total=len(self.scene_dirs[:1]),
                    desc="Loading files in parallel"
                ))
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
        finally:
            gc.collect()
        
        return results

    
    def get_frame(self, idx, scene_dir):
        frame_id = idx
        # Load RGB images, and depth images
        rgb_path = os.path.join(scene_dir, 'color', f'{frame_id}.jpg')
        depth_path = os.path.join(scene_dir, 'depth', f'{frame_id}.png')
        pose_path = os.path.join(scene_dir, 'pose', f'{frame_id}.txt')
        
        if self.rgb_only:
            rgb_image = Image.open(rgb_path)
            if self.transforms:
                rgb_image = self.transforms(rgb_image)

            sample = {'rgb': rgb_image}
            return sample
        
        rgb_image = Image.open(rgb_path)
        depth_image = self.to_tensor(np.array(Image.open(depth_path)).astype(np.float32) / 1000) #TODO: need to handle NaN?
        pose = torch.from_numpy(np.loadtxt(pose_path))  # Load camera pose

        # Apply transformations if provided
        if self.transforms:
            rgb_image = self.transforms(rgb_image)

        # Return as a dictionary (or your preferred format)
        sample = {
            'rgb': rgb_image,
            'depth': depth_image,
            'pose': pose,
        }

        return sample

    def __getitem__(self, scene_dir):

        total_frames = len(glob(os.path.join(scene_dir, 'color', '*.jpg')))
        total_frames = 50

        frames = []
        intrinsic_color = np.loadtxt(os.path.join(scene_dir, 'intrinsic', 'intrinsic_color.txt'))
        intrinsic_depth = np.loadtxt(os.path.join(scene_dir, 'intrinsic', 'intrinsic_depth.txt'))
        extrinsics_color = np.loadtxt(os.path.join(scene_dir, 'intrinsic', 'extrinsic_color.txt'))
        extrinsic_depth = np.loadtxt(os.path.join(scene_dir, 'intrinsic', 'extrinsic_depth.txt'))

        instrinsics = {'extrinsics_color': extrinsics_color , 
                       'extrinsic_depth': extrinsic_depth, 
                       'intrinsic_color': intrinsic_color , 
                       'intrinsic_depth': intrinsic_depth }  # Load intrinsic parameters from scene_info
            
        for i in range(total_frames):
            # print(i)
            sample = self.get_frame(i, scene_dir)
            sample['instrinsics'] = instrinsics
            frames.append(sample)
        
        return frames

class ScannetFromMemory(Dataset):
    def __init__(self, memory, num_frames=5):
        self.memory = memory
        self.num_frames = num_frames

    def __len__(self):
        return len(self.memory)
    
    def __getitem__(self, idx):
        item = self.memory[idx]
        sample = []
        total_frames = len(item)

        print(f"Total frames: {total_frames}")

        start_frame_idx = np.random.randint(0, total_frames - self.num_frames )

        for i in range(start_frame_idx, start_frame_idx + self.num_frames):
            sample.append(item[i])
        return sample

if __name__ == "__main__":
    # Example transformation (you can customize this as needed)
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset
    train_dataset = ScanNetV2Dataset(root_dir='/data/kmirakho/l3dProject/scannetv2', transforms=data_transforms, train = True)
    train_memory = train_dataset._load_data_into_memory()

    print(train_memory[0][0]['rgb'].shape)
    scannet_memory = ScannetFromMemory(train_memory)
    # Create DataLoader
    dataloader = DataLoader(scannet_memory, batch_size=20, shuffle=True, num_workers=4)

    # Example usage of the dataloader
    for batch in dataloader:
        print(batch[0]['rgb'].shape, batch[0]['depth'].shape)
        break
