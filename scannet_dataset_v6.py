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
import threading
import time
import dust3r.datasets.utils.cropping as cropping
import random
from collections import deque

class ScanNetPreprocessor:
    """Optimized preprocessor to load entire ScanNet dataset into memory"""
    
    def __init__(self, root_dir, output_h5_path=None, max_scenes=None, 
                 num_workers=None, rgb_only=False, compression="lzf"):
        """
        Args:
            root_dir: Path to ScanNet dataset directory
            output_h5_path: Path to save preprocessed dataset (optional)
            max_scenes: Maximum number of scenes to process (None=all)
            num_workers: Number of parallel workers (None=auto)
            rgb_only: Whether to only load RGB images
            compression: HDF5 compression filter ("lzf" for speed, "gzip" for size)
        """
        self.root_dir = root_dir
        self.output_h5_path = output_h5_path or "scannet_preprocessed.h5"
        self.scene_dirs = sorted(glob(os.path.join(root_dir, '*')))
        
        if max_scenes:
            self.scene_dirs = self.scene_dirs[:max_scenes]

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
    
    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None):
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
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

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
        image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

        return image, depthmap, intrinsics2


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
            depthmap = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0
            frame_data['depth'] = depthmap
            
            # Load pose
            pose_path = os.path.join(scene_dir, 'pose', f'{frame_id}.txt')
            pose = np.loadtxt(pose_path)
            frame_data['pose'] = pose    
            frame_data['rgb'], frame_data['depth'], frame_data['intrinsics'] = self._crop_resize_if_necessary(frame_data['rgb'], frame_data['depth'], intrinsic_depth, resolution=(224, 224)) #hardcode to 224
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
            frames_data['pose'] = []
        
        # Load camera intrinsics once per scene
        intrinsic_color_path = os.path.join(scene_dir, 'intrinsic', 'intrinsic_color.txt')
        intrinsic_depth_path = os.path.join(scene_dir, 'intrinsic', 'intrinsic_depth.txt')
        extrinsic_color_path = os.path.join(scene_dir, 'intrinsic', 'extrinsic_color.txt')
        extrinsic_depth_path = os.path.join(scene_dir, 'intrinsic', 'extrinsic_depth.txt')
        
        frames_data['intrinsics'] = {
            'intrinsic_color': np.loadtxt(intrinsic_color_path),
            'intrinsic_depth': np.loadtxt(intrinsic_depth_path),
            'extrinsic_color': np.loadtxt(extrinsic_color_path),
            'extrinsic_depth': np.loadtxt(extrinsic_depth_path)
        }
        intrinsic_depth = frames_data['intrinsics']['intrinsic_depth']
        
        # Process all frames in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Create a list of arguments for each frame
            frame_args = [(frame_id, scene_dir, self.rgb_only, intrinsic_depth) for frame_id in frame_ids]
            
            # Process frames in parallel and collect results
            # frame_results = list(tqdm(
            #     executor.map(lambda args: self.process_frame(*args), frame_args),
            #     total=len(frame_ids),
            #     desc=f"Processing frames for scene {scene_id}"
            # ))

            frame_results = list(executor.map(lambda args: self.process_frame(*args), frame_args))

        # Organize results into the frames_data structure
        for frame_data in frame_results:
            frames_data['rgb'].append(frame_data['rgb'])
            
            if not self.rgb_only:
                frames_data['depth'].append(frame_data['depth'])
                frames_data['pose'].append(frame_data['pose'])
        frames_data['intrinsics']['intrinsic_depth'] = frames_data['intrinsics']['intrinsic_depth']
        return frames_data
    
    def load_dataset(self, save_to_disk=True, return_data=False):
        """Load all scenes into memory with parallel processing"""
        print(f"Loading {self.num_scenes} scenes using {self.num_workers} workers...")
        
        # Prepare for parallel processing
        if self.num_workers > 1:
            # pool = mp.Pool(processes=self.num_workers)
            # # Process scenes in parallel
            # all_scenes_data = list(tqdm(
            #     pool.imap(self.process_scene, self.scene_dirs),
            #     total=self.num_scenes,
            #     desc="Loading scenes"
            # ))
            # pool.close()
            # pool.join()
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
            rgb = torch.from_numpy(rgb).permute(0,3,1,2).to(torch.float32, non_blocking=True)
            memory_dataset[scene_id] = {
                'rgb': rgb,
                'frame_ids': scene_data['frame_ids'],
                'intrinsics': scene_data['intrinsics'],
            }
            
            if not self.rgb_only:
                memory_dataset[scene_id]['depth'] = torch.from_numpy(np.stack(scene_data['depth'])).float()
                memory_dataset[scene_id]['pose'] = torch.from_numpy(np.stack(scene_data['pose'])).float()
        
        # Save to disk if requested
        if save_to_disk:
            self.save_to_h5(memory_dataset)
        
        # Clean up to release memory if not returning data
        if not return_data:
            del all_scenes_data
            gc.collect()
            return self.output_h5_path
        
        return memory_dataset
    
    def save_to_h5(self, dataset):
        """Save preprocessed dataset to HDF5 file for future use"""
        print(f"Saving dataset to {self.output_h5_path}...")
        
        with h5py.File(self.output_h5_path, 'w') as f:
            for scene_id, scene_data in tqdm(dataset.items(), desc="Saving scenes"):
                # Create scene group
                scene_group = f.create_group(scene_id)
                
                # Save RGB images with compression
                scene_group.create_dataset(
                    'rgb', 
                    data=scene_data['rgb'],
                    compression=self.compression,
                    chunks=True
                )
                
                # Save frame IDs
                scene_group.create_dataset(
                    'frame_ids', 
                    data=np.array(scene_data['frame_ids'], dtype='S10')
                )
                
                # Save intrinsics
                intrinsics_group = scene_group.create_group('intrinsics')
                for k, v in scene_data['intrinsics'].items():
                    intrinsics_group.create_dataset(k, data=v)
                
                if not self.rgb_only:
                    # Save depth maps with compression
                    scene_group.create_dataset(
                        'depth', 
                        data=scene_data['depth'],
                        compression=self.compression,
                        chunks=True
                    )
                    
                    # Save poses
                    scene_group.create_dataset(
                        'pose', 
                        data=scene_data['pose'],
                        compression=self.compression
                    )
        
        print(f"Dataset saved to {self.output_h5_path}")

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
        # frame_indices = [start_frame + (i * self.frame_skip) for i in range(self.num_frames)]
        rgb = scene_data['rgb'][start_frame:start_frame+self.num_frames]

        sample['rgb'] = rgb
        
        if not self.rgb_only:
            # Get depth maps and poses for all frames at once
            sample['depth'] = scene_data['depth'][start_frame:start_frame+self.num_frames]
            sample['pose'] = scene_data['pose'][start_frame:start_frame+self.num_frames]
        
        # Add intrinsics (same for all frames in scene)
        intrinsics = {}
        for k, v in scene_data['intrinsics'].items():
            intrinsics[k] = torch.from_numpy(v).float()
        sample['intrinsics'] = intrinsics
        
        return sample

class BufferedSceneDataset(Dataset):
    def __init__(self, root_dir, max_scenes=50, num_workers=8, prefetch_scenes=2, num_frames=32, frame_skip=1, data_transforms=None):
        """
        Args:
            root_dir: Path to dataset directory
            max_scenes: Maximum scenes per buffer
            num_workers: Workers for scene loading
            prefetch_scenes: Number of buffers to prefetch
        """
        self.root_dir = root_dir
        self.max_scenes = max_scenes
        self.num_workers = num_workers
        self.prefetch_scenes = prefetch_scenes
        self.data_transforms = data_transforms
        self.num_frames=num_frames
        self.frame_skip=frame_skip

        # Scene management
        self.all_scene_dirs = sorted(glob(os.path.join(root_dir, '*')))
        self.total_scenes = len(self.all_scene_dirs)
        self.current_idx = 0
        
        # Buffer management
        self.active_buffer = []
        self.prefetch_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Initialize buffers
        self._fill_buffers()

    def _load_scene_batch(self, scene_indices):
        """Load batch of scenes into memory"""
        preprocessor = ScanNetPreprocessor(
            root_dir=self.root_dir,
            max_scenes=len(scene_indices),
            num_workers=self.num_workers,
            rgb_only=False
        )
        return preprocessor.load_dataset(save_to_disk=False, return_data=True)

    def _fill_buffers(self):
        """Asynchronously fill prefetch buffers"""
        def prefetch_task():
            next_indices = range(self.current_idx, 
                               min(self.current_idx + self.max_scenes, self.total_scenes))
            
            new_data = self._load_scene_batch(next_indices)
            dataset = ScanNetMemoryDataset(
                dataset_source=new_data,
                num_frames=self.num_frames,
                transforms=self.data_transforms,
                frame_skip=self.frame_skip,
                rgb_only=False
            )
            
            with self.buffer_lock:
                if not self.active_buffer:
                    self.active_buffer = dataset
                else:
                    self.prefetch_buffer = dataset

        threading.Thread(target=prefetch_task, daemon=True).start()
    
    def _swap_buffers(self):
        """Swap prefetch buffer to active and start prefetching next batch"""
        with self.buffer_lock:
            # If no prefetch buffer is available, wait until one is ready
            if not self.prefetch_buffer:
                print("Waiting for prefetch buffer to be ready...")
                self.buffer_lock.release()
                # Simple polling with sleep to avoid busy waiting
                while True:
                    time.sleep(0.1)
                    with self.buffer_lock:
                        if self.prefetch_buffer:
                            break
                        self.buffer_lock.release()
                self.buffer_lock.acquire()
            
            # Swap buffers
            self.active_buffer = self.prefetch_buffer
            self.prefetch_buffer = []
        
        # Update current index and start prefetching next batch
        self.current_idx = min(self.current_idx + self.max_scenes, self.total_scenes)
        
        # Only prefetch if there are more scenes to load
        if self.current_idx < self.total_scenes:
            self._fill_buffers()
    
    def __len__(self):
        """Return the total number of samples across all scenes"""
        # Wait until active buffer is available
        while not self.active_buffer:
            time.sleep(0.1)
        
        # Use the length of active buffer for current batch
        total_len = len(self.active_buffer)
        
        # Account for wrapped-around scenes if at the end
        if self.current_idx + self.max_scenes >= self.total_scenes:
            # At final batch
            return total_len
        
        # Return length of current active buffer
        return total_len
    
    def __getitem__(self, idx):
        """Get a sample from the active buffer, swapping if necessary"""
        # Wait for initial buffer to be ready
        while not self.active_buffer:
            time.sleep(0.1)
        
        # Check if we've exhausted the current buffer
        if idx >= len(self.active_buffer):
            # If we're at the end of the dataset, wrap around
            if self.current_idx + self.max_scenes >= self.total_scenes:
                self.current_idx = 0
                #randomly shuffle the  scene dirs
                random.shuffle(self.all_scene_dirs)
                self._fill_buffers()
                while not self.active_buffer:
                    time.sleep(0.1)
            else:
                # Otherwise swap to the next buffer
                self._swap_buffers()
            
            # Adjust index for the new buffer
            idx = idx % len(self.active_buffer)
        
        # Get the item from the active buffer
        return self.active_buffer


class BufferedSceneDataset(Dataset):
    def __init__(self, root_dir, max_scenes=50, num_workers=8, 
                 prefetch_scenes=2, num_frames=32, frame_skip=1, 
                 data_transforms=None):
        self.root_dir = root_dir
        self.max_scenes = max_scenes
        self.num_workers = num_workers
        self.prefetch_scenes = prefetch_scenes
        self.data_transforms = data_transforms
        self.num_frames=num_frames
        self.frame_skip=frame_skip

        # Scene management
        self.all_scene_dirs = sorted(glob(os.path.join(root_dir, '*')))
        self.total_scenes = len(self.all_scene_dirs)
        self.current_idx = 0
    
    def _load_scene_batch(self, scene_indices):
        """Load batch of scenes into memory"""
        preprocessor = ScanNetPreprocessor(
            root_dir=self.root_dir,
            max_scenes=len(scene_indices),
            num_workers=self.num_workers,
            rgb_only=False
        )
        return preprocessor.load_dataset(save_to_disk=False, return_data=True)
    
    def fetch_dataset(self):
        # Load scenes and create dataset
        next_indices = range(self.current_idx, min(self.current_idx + self.max_scenes, self.total_scenes))
        new_data = self._load_scene_batch(next_indices)
        dataset = ScanNetMemoryDataset(
            dataset_source=new_data,
            num_frames=self.num_frames,
            transforms=self.data_transforms,
            frame_skip=self.frame_skip,
            rgb_only=False
            )                
        self.current_idx = (self.current_idx + len(next_indices))%self.total_scenes
    
        return dataset


# Example usage
def main():
    # Define transforms
    data_transforms = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Usage
    buffer_scene = BufferedSceneDataset(
        root_dir='/data/kmirakho/l3d_proj/scannetv2',
        max_scenes=2,
        num_workers=8,
        prefetch_scenes=2,
        num_frames=32,
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
            time.sleep(0.1)
            pass

    # Step 1: Preprocess and load dataset to memory
    # preprocessor = ScanNetPreprocessor(
    #     root_dir='/data/kmirakho/l3d_proj/scannetv2',
    #     output_h5_path='scannet_preprocessed.h5',
    #     max_scenes=10,  # Limit to 100 scenes for memory efficiency
    #     num_workers=8,   # Adjust based on your system
    #     rgb_only=False,  # Set to True to only load RGB images
    #     compression="lzf" # Fast compression
    # )
    
    # Option 1: Save to disk for future use
    # h5_path = preprocessor.load_dataset(save_to_disk=False, return_data=False)
    
    # Option 2: Load directly to memory
    # memory_dataset = preprocessor.load_dataset(save_to_disk=False, return_data=True)
    
    # Step 2: Create memory dataset
    # From disk (more memory efficient)
    # h5_path='scannet_preprocessed.h5'
    # dataset = ScanNetMemoryDataset(
    #     dataset_source=h5_path,
    #     num_frames=5,
    #     transforms=data_transforms,
    #     frame_skip=1,
    #     rgb_only=False
    # )
    
    # From memory (fastest, but requires most RAM)
    # dataset = ScanNetMemoryDataset(
    #     dataset_source=memory_dataset,
    #     num_frames=32,
    #     transforms=data_transforms,
    #     frame_skip=1,
    #     rgb_only=False
    # )
    
    # # Step 3: Create optimized DataLoader
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=8,
    #     shuffle=True,
    #     num_workers=4,  # For dataset in memory, fewer workers needed
    #     pin_memory=True,
    #     prefetch_factor=8,
    #     persistent_workers=True
    # )
    
    # # Example training loop
    # for epoch in range(10):
    #     epoch_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
    #     for batch_idx, batch in enumerate(epoch_bar):
    #         # Process batch here
    #         # import pdb
    #         # pdb.set_trace()
    #         time.sleep(0.1)
    #         pass


if __name__ == "__main__":
    main()