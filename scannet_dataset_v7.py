import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from PIL import Image
import cv2
from glob import glob
from tqdm import tqdm
import gc
import psutil
import h5py
from functools import lru_cache
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor
import time

# dont convert anything to float32. Pose and intrinsics to np.float32
# remove the 224 reshaping
# rgb should be np.uint8
# ground truth depth is to be divided by 1000 
# output is everything in torch.uint8

class ScanNetPreprocessor:
    """Optimized preprocessor to load entire ScanNet dataset into memory"""
    
    def __init__(self, scene_dirs, output_h5_path=None, max_scenes=None, 
                 num_workers=None, scene_indices=None, rgb_only=False, compression="lzf"):
        """
        Args:
            root_dir: Path to ScanNet dataset directory
            output_h5_path: Path to save preprocessed dataset (optional)
            max_scenes: Maximum number of scenes to process (None=all)
            num_workers: Number of parallel workers (None=auto)
            rgb_only: Whether to only load RGB images
            compression: HDF5 compression filter ("lzf" for speed, "gzip" for size)
        """
        self.scene_dirs = scene_dirs[scene_indices]

        self.num_scenes = len(self.scene_dirs)
        self.rgb_only = rgb_only
        self.compression = compression
        
        # Determine optimal number of workers based on system
        if num_workers is None:
            self.num_workers = min(mp.cpu_count() - 1, 8)  # Leave 1 core free
        else:
            self.num_workers = num_workers
            
        # Calculate memory requirements and check availability
        self._check_memory_requirements()
        
    def _check_memory_requirements(self):
        """Estimate memory requirements and check if sufficient memory is available"""
        # Sample a few scenes to estimate size per scene
        sample_size = min(5, self.num_scenes)
        test_scenes = self.scene_dirs[:sample_size]
        
        total_frames = 0
        for scene_dir in test_scenes:
            frames = len(glob(os.path.join(scene_dir, 'color', '*.jpg')))
            total_frames += frames
        
        # Rough estimate: RGB (3 channels, uint8) + Depth (1 channel, float32) + Pose (4x4, float32)
        # RGB: 3 * height * width * 1 byte
        # Depth: height * width * 4 bytes
        # Pose: 16 * 4 bytes = 64 bytes
        avg_frames_per_scene = total_frames / sample_size
        est_frame_size_mb = (3 * 640 * 480 + 640 * 480 * 4 + 64) / (1024 * 1024)  # Assuming 640x480 resolution
        
        est_total_size_gb = (self.num_scenes * avg_frames_per_scene * est_frame_size_mb) / 1024
        
        # Get available system memory
        available_memory_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
        
        print(f"Estimated dataset size: {est_total_size_gb:.2f} GB")
        print(f"Available system memory: {available_memory_gb:.2f} GB")
        
        if est_total_size_gb > available_memory_gb * 0.9:  # Allow 90% memory usage
            print(f"WARNING: Dataset may not fit in memory!")
            print(f"Consider reducing number of scenes or using disk-based storage.")

    def process_frame(self, frame_id, scene_dir, rgb_only, intrinsic_depth):
        """Process a single frame and return its data
        
        Args:
            frame_id: ID of the frame to process
            scene_dir: Directory containing the scene data
            rgb_only: Whether to only load RGB images
            
        Returns:
            Dictionary containing the frame data
        """
        frame_data = {}
        
        # Load RGB image and convert to uniform format (uint8 array)
        rgb_path = os.path.join(scene_dir, 'color', f'{frame_id}.jpg')
        # Use cv2 for faster image loading
        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        frame_data['rgb'] = rgb_img
        
        if not rgb_only:
            # Load depth image
            depth_path = os.path.join(scene_dir, 'depth', f'{frame_id}.png')
            depthmap = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) # verified it's UINT16 - Sreeharsha
            # the groundtruth depth map is a metric depthmap and has a scale of 1000
            # and the predicted_depthmap is uint8

            # .astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0
            frame_data['depth'] = depthmap.astype(np.int16) # to allow for float conversion later
            
            pred_depth_path = os.path.join(scene_dir, 'depth_any', f'{frame_id}.npz')
            # pred_depth_path = os.path.join(os.path.basename(scene_dir), f'{frame_id}.npz')

            pred_depthmap = np.load(pred_depth_path)
            pred_depthmap = pred_depthmap['depth'] # verified it's UINT8 - Sreeharsha
            # print(f"Here check the pred depthmap dtype")
            #.astype(np.float32) / 255
            pred_depthmap = pred_depthmap[:, :, None]
            pred_depthmap[~np.isfinite(pred_depthmap)] = 0 # this is of size (968, 1296, 1)

            frame_data['pred_depth'] = pred_depthmap

            # Load pose
            pose_path = os.path.join(scene_dir, 'pose', f'{frame_id}.txt')
            pose = np.loadtxt(pose_path)
            frame_data['pose'] = pose.astype(np.float32) # this is of float64
        return frame_data
        
    def process_scene(self, scene_dir):
        """Process a single scene and return its data"""
        scene_id = os.path.basename(scene_dir)
        print(f"Processing scene {scene_id}")
        
        # Get all frame IDs
        frame_files = sorted(glob(os.path.join(scene_dir, 'color', '*.jpg')))
        frame_ids = [os.path.splitext(os.path.basename(f))[0] for f in frame_files]
        
        # Preallocate data structures for this scene
        frames_data = {
            'rgb': [],
            'scene_id': scene_id,
            'frame_ids': frame_ids
        }
        
        if not self.rgb_only:
            frames_data['depth'] = []
            frames_data['pred_depth'] = []
            frames_data['pose'] = []
        
        # Load camera intrinsics once per scene
        # intrinsic_color_path = os.path.join(scene_dir, 'intrinsic', 'intrinsic_color.txt')
        intrinsic_depth_path = os.path.join(scene_dir, 'intrinsics', 'intrinsic_depth.txt')
        # extrinsic_color_path = os.path.join(scene_dir, 'intrinsic', 'extrinsic_color.txt')
        # extrinsic_depth_path = os.path.join(scene_dir, 'intrinsic', 'extrinsic_depth.txt')
        
        frames_data['intrinsics'] = {
            # 'intrinsic_color': np.loadtxt(intrinsic_color_path),
            'intrinsic_depth': np.loadtxt(intrinsic_depth_path)
            # 'extrinsic_color': np.loadtxt(extrinsic_color_path),
            # 'extrinsic_depth': np.loadtxt(extrinsic_depth_path)
        }
        intrinsic_depth = frames_data['intrinsics']['intrinsic_depth']
        
        # Process all frames in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Create a list of arguments for each frame
            frame_args = [(frame_id, scene_dir, self.rgb_only, intrinsic_depth) for frame_id in frame_ids]
            
            frame_results = list(executor.map(lambda args: self.process_frame(*args), frame_args))

        # Organize results into the frames_data structure
        for frame_data in frame_results:
            frames_data['rgb'].append(frame_data['rgb'])
            
            if not self.rgb_only:
                frames_data['depth'].append(frame_data['depth'])
                frames_data['pred_depth'].append(frame_data['pred_depth'])
                frames_data['pose'].append(frame_data['pose'])
        frames_data['intrinsics']['intrinsic_depth'] = frames_data['intrinsics']['intrinsic_depth'].astype(np.float32)
        return frames_data
    
    def load_dataset(self, save_to_disk=True, return_data=False):
        """Load all scenes into memory with parallel processing"""
        print(f"Loading {self.num_scenes} scenes using {self.num_workers} workers...")
        
        # Prepare for parallel processing
        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                all_scenes_data = list(tqdm(
                    executor.map(self.process_scene, self.scene_dirs),
                    total=self.num_scenes,
                    desc="Loading scenes"
                ))
        else:
            # Single process loading
            all_scenes_data = [self.process_scene(scene_dir) for scene_dir in tqdm(self.scene_dirs)]
        
        # Convert to more efficient structure for memory usage
        memory_dataset = {}
        for scene_data in all_scenes_data:
            scene_id = scene_data['scene_id']
            rgb = np.stack(scene_data['rgb'])
            # rgb = torch.from_numpy(rgb).permute(0,3,1,2).to(torch.float32, non_blocking=True)/255
            rgb = torch.from_numpy(rgb).permute(0,3,1,2).to(torch.uint8, non_blocking=True) #changed to uint8
            memory_dataset[scene_id] = {
                'rgb': rgb,
                'frame_ids': scene_data['frame_ids'],
                'intrinsics': scene_data['intrinsics'],
            }
            
            if not self.rgb_only:
                memory_dataset[scene_id]['depth'] = torch.from_numpy(np.stack(scene_data['depth']))#.float()
                memory_dataset[scene_id]['pred_depth'] = torch.from_numpy(np.stack(scene_data['pred_depth']))#.float()
                memory_dataset[scene_id]['pose'] = torch.from_numpy(np.stack(scene_data['pose']))#.float()
        
        del all_scenes_data
        gc.collect()        
        return memory_dataset



class ScanNetMemoryDataset(Dataset):
    """Dataset for efficient loading from in-memory preprocessed ScanNet data"""
    
    def __init__(self, dataset_source, num_frames=20, transforms=None, 
                 frame_skip=1, rgb_only=False):
        """
        Args:
            dataset_source: Either path to h5 file or loaded dataset dict
            num_frames: Number of consecutive frames to return
            transforms: Transforms to apply to RGB images
            frame_skip: Number of frames to skip between consecutive frames
            rgb_only: Whether to only use RGB images
        """
        self.num_frames = num_frames
        self.transforms = transforms
        self.frame_skip = frame_skip
        self.rgb_only = rgb_only
        
        # Dataset is already in memory
        self.memory_dataset = dataset_source
        self.scene_ids = list(self.memory_dataset.keys())
        
        # Apply transforms once to all scenes if in-memory dataset
        # This avoids applying transforms repeatedly during training
        if self.transforms is not None:
            # cannot apply transformations on an unsigned8 int.
            print("Applying transforms to all scenes up front...")
            for scene_id in tqdm(self.scene_ids, desc="Applying transforms"):
                self.memory_dataset[scene_id]['rgb'] = self.transforms(self.memory_dataset[scene_id]['rgb'])
            # Set transforms to None since data is already transformed
            self.transforms = None

        # Create samples list
        self.samples = []
        print("Processing Frames...")
        for scene_id in tqdm(self.scene_ids):
            scene_data = self.memory_dataset[scene_id]
            num_frames_in_scene = len(scene_data['rgb'])
            
            # Generate valid frame sequences
            max_start_idx = num_frames_in_scene - self.num_frames * self.frame_skip
            if max_start_idx > 0:
                for start_idx in range(0, max_start_idx + 1, self.frame_skip):
                    self.samples.append((scene_id, start_idx))
    
        print(f"Dataset initialized with {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        scene_id, start_frame = self.samples[idx]
    
        # Get data from memory
        scene_data = self.memory_dataset[scene_id]

        sample = {}
        
        # Get RGB images for all frames in sequence at once
        frame_indices = np.arange(self.num_frames) * self.frame_skip + start_frame
        frame_indices = frame_indices.astype(int)
        # add frame skip - start_frame + (i * self.frame_skip) for i in range(self.num_frames)]
        rgb = scene_data['rgb'][frame_indices]

        sample['rgb'] = rgb
        
        if not self.rgb_only:
            # Get depth maps and poses for all frames at once
            sample['depth'] = scene_data['depth'][frame_indices]
            sample['pred_depth'] = scene_data['pred_depth'][frame_indices]
            sample['pose'] = scene_data['pose'][frame_indices]
        
        # Add intrinsics (same for all frames in scene)
        intrinsics = {}
        for k, v in scene_data['intrinsics'].items():
            intrinsics[k] = torch.from_numpy(v) #.float()
        sample['intrinsics'] = intrinsics
        
        return sample

class BufferedSceneDataset(Dataset):
    def __init__(self, root_dir, max_scenes=10, num_workers=8,num_frames=32, frame_skip=1, 
                 data_transforms=None):
        self.max_scenes = max_scenes
        self.num_workers = num_workers
        self.data_transforms = data_transforms
        self.num_frames=num_frames
        self.frame_skip=frame_skip

        # Scene management
        self.all_scene_dirs = sorted(glob(os.path.join(root_dir, '*')))
        self.all_scene_dirs = np.array(self.all_scene_dirs)
        self.total_scenes = len(self.all_scene_dirs)
        self.current_idx = 0
    
    def _load_scene_batch(self, scene_indices):
        """Load batch of scenes into memory"""
        preprocessor = ScanNetPreprocessor(
            scene_dirs=self.all_scene_dirs,
            max_scenes=len(scene_indices),
            scene_indices=scene_indices,
            num_workers=self.num_workers,
            rgb_only=False
        )
        return preprocessor.load_dataset(save_to_disk=False, return_data=True)
    
    def fetch_dataset(self):
        # Load scenes and create dataset
        next_indices = np.arange(self.current_idx, min(self.current_idx + self.max_scenes, self.total_scenes))
        new_data = self._load_scene_batch(next_indices)
        dataset = ScanNetMemoryDataset(
            dataset_source=new_data,
            num_frames=self.num_frames,
            transforms=self.data_transforms,
            frame_skip=self.frame_skip,
            rgb_only=False
            )
        if self.current_idx+len(next_indices) > self.total_scenes:
            np.random.shuffle(self.all_scene_dirs)
        self.current_idx = (self.current_idx + len(next_indices))%self.total_scenes
    
        return dataset
    
    def fetch_eval_dataset(self):
        # Load scenes and create dataset
        next_indices = np.arange(self.current_idx, min(self.current_idx + self.max_scenes, self.total_scenes))
        next_indices = np.random.choice(next_indices, 2, replace=False)
        new_data = self._load_scene_batch(next_indices)
        dataset = ScanNetMemoryDataset(
            dataset_source=new_data,
            num_frames=1,
            transforms=self.data_transforms,
            frame_skip=self.frame_skip,
            rgb_only=False
            )    
        return dataset

# Example usage
def main():
    # Define transforms
    # data_transforms = transforms.Compose([
    #     # transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    data_transforms = None
    buffer_scene = BufferedSceneDataset(
        root_dir='/data/kmirakho/l3d_proj/scannetv4',
        max_scenes=10,
        num_workers=8,
        num_frames=4,
        frame_skip=1,
        data_transforms=data_transforms
    )

    for epoch in range(10):
        dataset = buffer_scene.fetch_dataset()
        
        # Step 3: Create optimized DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=4,  # For dataset in memory, fewer workers needed
            pin_memory=True,
            prefetch_factor=8,
            persistent_workers=True
        )
        epoch_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
        for batch_idx, batch in enumerate(epoch_bar):
            rgb = batch['rgb'][0, 0, :, : ,:] # shape is batch x ??? x c x h x w
            #save rgb image
            rgb = rgb.permute(1, 2, 0).numpy()
            rgb = (rgb - rgb.min())/(rgb.max() - rgb.min()) * 255
            rgb = rgb.astype(np.uint8)
            cv2.imwrite("lanja_rgb.png", rgb)

            depth = batch['depth'][0, 0, :, :]
            depth = depth.numpy()
            depth = (depth - depth.min())/(depth.max() - depth.min()) * 255
            depth = depth.astype(np.uint8)
            cv2.imwrite("lanja_depth.png", depth)

            pred_depth = batch['pred_depth'][0, 0, : ,:]
            pred_depth = pred_depth.numpy()
            pred_depth = (pred_depth - pred_depth.min())/(pred_depth.max() - pred_depth.min()) * 255
            pred_depth = pred_depth.astype(np.uint8)
            cv2.imwrite("lanja_pred_depth.png", pred_depth)


            print(f"Batch {batch_idx} - RGB shape: {rgb.shape}, RGB Datatype: {rgb.dtype}")
            print(f"Batch {batch_idx} - Depth shape: {batch['depth'].shape}, Depth Datatype: {batch['depth'].dtype}")
            print(f"Batch {batch_idx} - Pred Depth shape: {batch['pred_depth'].shape}, Pred Depth Datatype: {batch['pred_depth'].dtype}")
            print(f"Batch {batch_idx} - Pose shape: {batch['pose'].shape}, Pose Datatype: {batch['pose'].dtype}")
            print(f"Batch {batch_idx} - Intrinsics shape: {batch['intrinsics']['intrinsic_depth'].shape}, Intrinsics Datatype: {batch['intrinsics']['intrinsic_depth'].dtype}")
            break
        break
if __name__ == "__main__":
    main()