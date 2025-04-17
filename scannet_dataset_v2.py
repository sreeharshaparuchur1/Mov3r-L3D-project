# import torch
# from torch.utils.data import Dataset, DataLoader
# import os
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# from glob import glob
# from functools import partial
# import multiprocessing as mp
# from tqdm import tqdm
# import gc

# class ScanNetV2Dataset(Dataset):
#     def __init__(self, root_dir, num_frames=20, transforms=None, train = True, rgb_only=False):
#         """
#         Args:
#             root_dir (string): Directory with all the ScanNet V2 data.
#             transforms (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.root_dir = root_dir
#         self.transforms = transforms
#         self.num_frames = num_frames
#         self.rgb_only = rgb_only

#         # List all scene directories
#         self.scene_dirs = glob(os.path.join(root_dir,'*'))
#         self.num_scenes = len(self.scene_dirs) 
#         print(f"Number of scenes: {self.num_scenes}")

#     def __len__(self):
#         return self.num_scenes
    
#     def _load_data_into_memory(self):
#         num_workers = max(1, mp.cpu_count() - 1)  # Use all CPUs except one

#         print(f"Loading dataset using {num_workers} processes...")
#         try:
#             with mp.Pool(processes=num_workers) as pool:
#                 results = list(tqdm(
#                     pool.imap(self.__getitem__, self.scene_dirs),
#                     total=len(self.scene_dirs),
#                     desc="Loading files in parallel"
#                 ))
#         except Exception as e:
#             print(f"Error loading data: {e}")
#             return []
#         finally:
#             gc.collect()
        
#         return results

    
#     def get_frame(self, idx, scene_dir):
#         frame_id = idx
#         # Load RGB images, and depth images
#         rgb_path = os.path.join(scene_dir, 'color', f'{frame_id}.jpg')
#         depth_path = os.path.join(scene_dir, 'depth', f'{frame_id}.png')
#         pose_path = os.path.join(scene_dir, 'pose', f'{frame_id}.txt')
        
#         if self.rgb_only:
#             rgb_image = Image.open(rgb_path)
#             if self.transforms:
#                 rgb_image = self.transforms(rgb_image)

#             sample = {'rgb': rgb_image}
#             return sample
        
#         rgb_image = Image.open(rgb_path)
#         depth_image = np.array(Image.open(depth_path)).astype(np.float32) / 1000 #TODO: need to handle NaN?
#         pose = np.loadtxt(pose_path)  # Load camera pose

#         # Apply transformations if provided
#         if self.transforms:
#             rgb_image = self.transforms(rgb_image)

#         # Return as a dictionary (or your preferred format)
#         sample = {
#             'rgb': rgb_image,
#             'depth': depth_image,
#             'pose': pose,
#         }

#         return sample

#     def __getitem__(self, scene_dir):

#         total_frames = len(glob(os.path.join(scene_dir, 'color', '*.jpg')))
#         total_frames = 50

#         frames = []
#         intrinsic_color = np.loadtxt(os.path.join(scene_dir, 'intrinsic', 'intrinsic_color.txt'))
#         intrinsic_depth = np.loadtxt(os.path.join(scene_dir, 'intrinsic', 'intrinsic_depth.txt'))
#         extrinsics_color = np.loadtxt(os.path.join(scene_dir, 'intrinsic', 'extrinsic_color.txt'))
#         extrinsic_depth = np.loadtxt(os.path.join(scene_dir, 'intrinsic', 'extrinsic_depth.txt'))

#         instrinsics = {'extrinsics_color': extrinsics_color , 
#                        'extrinsic_depth': extrinsic_depth, 
#                        'intrinsic_color': intrinsic_color , 
#                        'intrinsic_depth': intrinsic_depth }  # Load intrinsic parameters from scene_info
            
#         for i in range(total_frames):
#             # print(i)
#             sample = self.get_frame(i, scene_dir)
#             sample['instrinsics'] = instrinsics
#             frames.append(sample)
        
#         return frames

# class ScannetFromMemory(Dataset):
#     def __init__(self, memory, num_frames=5):
#         self.memory = memory
#         self.num_frames = num_frames

#     def __len__(self):
#         return len(self.memory)
    
#     def __getitem__(self, idx):
#         item = self.memory[idx]
#         sample = []
#         total_frames = len(item)

#         print(f"Total frames: {total_frames}")

#         start_frame_idx = np.random.randint(0, total_frames - self.num_frames )

#         for i in range(start_frame_idx, start_frame_idx + self.num_frames):
#             sample.append(item[i])
#         return sample


# # Example transformation (you can customize this as needed)
# data_transforms = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Create dataset
# train_dataset = ScanNetV2Dataset(root_dir='/data/kmirakho/l3dProject/scannetv2', transforms=data_transforms, train = True)
# train_memory = train_dataset._load_data_into_memory()

# print(train_memory[0][0]['rgb'].shape)
# scannet_memory = ScannetFromMemory(train_memory)
# # Create DataLoader
# dataloader = DataLoader(scannet_memory, batch_size=20, shuffle=True, num_workers=4)

# # Example usage of the dataloader
# for batch in dataloader:
#     print(batch[0]['rgb'].shape, batch[0]['depth'].shape)
#     break


import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from PIL import Image
import cv2
from functools import lru_cache
from torchvision import transforms
import multiprocessing as mp
from tqdm import tqdm
import gc
from threading import Lock
import time

# Cache for faster image loading
@lru_cache(maxsize=1024)
def cached_imread_rgb(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

@lru_cache(maxsize=1024)
def cached_imread_depth(path):
    return cv2.imread(path, cv2.IMREAD_ANYDEPTH)

class ScanNetV2Dataset(Dataset):
    def __init__(self, root_dir, num_frames=20, transforms=None, train=True, 
                 rgb_only=False, frame_skip=1, cache_size=20):
        """
        Args:
            root_dir: Directory with all the ScanNet V2 data
            num_frames: Number of frames per sequence
            transforms: Optional transform for RGB images
            train: Whether this is training set
            rgb_only: Whether to load only RGB images
            frame_skip: Number of frames to skip between consecutive frames
            cache_size: Number of scenes to cache in memory
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.num_frames = num_frames
        self.rgb_only = rgb_only
        self.frame_skip = frame_skip
        self.cache_size = cache_size
        
        # List all scene directories
        self.scene_dirs = sorted(glob(os.path.join(root_dir, '*')))
        self.num_scenes = len(self.scene_dirs)
        print(f"Number of scenes: {self.num_scenes}")
        
        # Precompute metadata without loading everything
        self.scene_metadata = {}
        self.samples = []
        
        print("Preprocessing dataset metadata...")
        for scene_idx, scene_dir in enumerate(tqdm(self.scene_dirs)):
            # Count frames in this scene
            frame_count = len(glob(os.path.join(scene_dir, 'color', '*.jpg')))
            # frame_count = min(frame_count, 50)  # Cap at 50 as in original code
            
            # Store metadata
            self.scene_metadata[scene_dir] = {
                'frame_count': frame_count,
                'intrinsic_paths': {
                    'color': os.path.join(scene_dir, 'intrinsic', 'intrinsic_color.txt'),
                    'depth': os.path.join(scene_dir, 'intrinsic', 'intrinsic_depth.txt'),
                    'extrinsic_color': os.path.join(scene_dir, 'intrinsic', 'extrinsic_color.txt'),
                    'extrinsic_depth': os.path.join(scene_dir, 'intrinsic', 'extrinsic_depth.txt'),
                }
            }
            
            # Create samples (scene_idx, start_frame)
            max_start_idx = frame_count - num_frames*frame_skip
            if max_start_idx > 0:
                for start_idx in range(0, max_start_idx + 1, frame_skip):
                    self.samples.append((scene_idx, start_idx))
        
        # Scene caching
        self.cache = {}
        self.cache_lock = Lock()
        self.lru_scenes = []  # Track least recently used scenes
    
    def __len__(self):
        return len(self.samples)
    
    def _get_intrinsics(self, scene_dir):
        """Load or fetch camera intrinsics from cache"""
        if scene_dir in self.cache and 'intrinsics' in self.cache[scene_dir]:
            return self.cache[scene_dir]['intrinsics']
        
        paths = self.scene_metadata[scene_dir]['intrinsic_paths']
        intrinsics = {
            'intrinsic_color': np.loadtxt(paths['color']),
            'intrinsic_depth': np.loadtxt(paths['depth']),
            'extrinsics_color': np.loadtxt(paths['extrinsic_color']),
            'extrinsic_depth': np.loadtxt(paths['extrinsic_depth'])
        }
        
        with self.cache_lock:
            if scene_dir not in self.cache:
                self.cache[scene_dir] = {}
            self.cache[scene_dir]['intrinsics'] = intrinsics
            
        return intrinsics
    
    def get_frame(self, frame_id, scene_dir):
        """Load a single frame data"""
        # Check if frame is in cache
        if scene_dir in self.cache and 'frames' in self.cache[scene_dir] and frame_id in self.cache[scene_dir]['frames']:
            return self.cache[scene_dir]['frames'][frame_id]
        
        # Load RGB image
        rgb_path = os.path.join(scene_dir, 'color', f'{frame_id}.jpg')
        rgb_image = cached_imread_rgb(rgb_path)
        
        if self.rgb_only:
            if self.transforms:
                rgb_image = self.transforms(Image.fromarray(rgb_image))
            sample = {'rgb': rgb_image}
            return sample
        
        # Load depth and pose
        depth_path = os.path.join(scene_dir, 'depth', f'{frame_id}.png')
        pose_path = os.path.join(scene_dir, 'pose', f'{frame_id}.txt')
        
        depth_image = cached_imread_depth(depth_path).astype(np.float32) / 1000
        pose = np.loadtxt(pose_path)
        
        # Apply transformations to RGB if provided
        if self.transforms:
            rgb_image = self.transforms(Image.fromarray(rgb_image))
        
        # Create sample
        sample = {
            'rgb': rgb_image,
            'depth': depth_image,
            'pose': pose,
        }
        
        # Cache the frame
        with self.cache_lock:
            if scene_dir not in self.cache:
                # Add new scene to cache
                self.cache[scene_dir] = {'frames': {}}
                self.lru_scenes.append(scene_dir)
                
                # Check cache size
                if len(self.lru_scenes) > self.cache_size:
                    # Remove least recently used scene
                    lru_scene = self.lru_scenes.pop(0)
                    del self.cache[lru_scene]
            elif scene_dir in self.lru_scenes:
                # Move to end of LRU list (most recently used)
                self.lru_scenes.remove(scene_dir)
                self.lru_scenes.append(scene_dir)
                
            # Add frame to scene cache
            if 'frames' not in self.cache[scene_dir]:
                self.cache[scene_dir]['frames'] = {}
            self.cache[scene_dir]['frames'][frame_id] = sample
        
        return sample

    # def get_frame(self, frame_id, scene_dir):
    #     # Check cache first
    #     if scene_dir in self.cache and frame_id in self.cache[scene_dir].get('frames', {}):
    #         return self.cache[scene_dir]['frames'][frame_id]

    #     # Load files with proper context managers
    #     with open(os.path.join(scene_dir, 'color', f'{frame_id}.jpg'), 'rb') as f:
    #         rgb_image = Image.open(f)
    #         rgb_image.load()  # Load image data before closing file
    #         if self.transforms:
    #             rgb_image = self.transforms(rgb_image)
        
    #     if self.rgb_only:
    #         sample = {'rgb': rgb_image}
    #     else:
    #         with open(os.path.join(scene_dir, 'depth', f'{frame_id}.png'), 'rb') as f:
    #             depth_image = np.array(Image.open(f).convert('I')).astype(np.float32) / 1000
            
    #         with open(os.path.join(scene_dir, 'pose', f'{frame_id}.txt'), 'r') as f:
    #             pose = np.loadtxt(f)
                
    #         sample = {
    #             'rgb': rgb_image,
    #             'depth': depth_image,
    #             'pose': pose
    #         }
        
    #     # Cache the frame
    #     with self.cache_lock:
    #         if scene_dir not in self.cache:
    #             # Add new scene to cache
    #             self.cache[scene_dir] = {'frames': {}}
    #             self.lru_scenes.append(scene_dir)
                
    #             # Check cache size
    #             if len(self.lru_scenes) > self.cache_size:
    #                 # Remove least recently used scene
    #                 lru_scene = self.lru_scenes.pop(0)
    #                 del self.cache[lru_scene]
    #         elif scene_dir in self.lru_scenes:
    #             # Move to end of LRU list (most recently used)
    #             self.lru_scenes.remove(scene_dir)
    #             self.lru_scenes.append(scene_dir)
                
    #         # Add frame to scene cache
    #         if 'frames' not in self.cache[scene_dir]:
    #             self.cache[scene_dir]['frames'] = {}
    #         self.cache[scene_dir]['frames'][frame_id] = sample
    #     return sample

    
    def __getitem__(self, idx):
        scene_idx, start_frame = self.samples[idx]
        scene_dir = self.scene_dirs[scene_idx]
        
        # Load intrinsics
        intrinsics = self._get_intrinsics(scene_dir)
        
        # Load frames
        frames = []
        for i in range(self.num_frames):
            # print(idx)
            frame_id = start_frame + (i * self.frame_skip)
            sample = self.get_frame(frame_id, scene_dir)
            sample['intrinsics'] = intrinsics
            frames.append(sample)
        
        return frames
    

if __name__ == "__main__":

    #data transforms
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Create dataset with optimizations
    dataset = ScanNetV2Dataset(
        root_dir='/data/kmirakho/l3dProject/scannetv2',
        transforms=data_transforms,
        num_frames=32,
        frame_skip=1,
        cache_size=20  # Cache 20 scenes in memory
    )

    # Create optimized DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,  # Adjust based on CPU cores
        pin_memory=True,  # Faster CPU to GPU transfers
        prefetch_factor=8,  # Prefetch 2 samples per worker
        persistent_workers=True  # Keep workers alive between epochs
    )
    epoch = 0
    epoch_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}', bar_format='{l_bar}{bar:20}{r_bar}', leave=True)
    for batch_idx, batch in enumerate(epoch_bar):
        time.sleep(0.01)