"""
3D Dataset for volumetric microscopy data.
Loads and processes full 3D volumes for true volumetric training.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Try to import tifffile, fall back to PIL if not available
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    print("Warning: tifffile not available. Some 3D functionality may be limited.")


class Volume3DUnpairedDataset(Dataset):
    """
    Dataset for unpaired 3D volumetric microscopy data.
    
    Loads full Z-stacks as 3D volumes for true volumetric processing.
    Can either load entire volumes or random 3D crops for memory-efficient training.
    
    Expected file format:
    - TIFF files with shape (D, H, W) where D is the full Z-stack depth
    - All volumes should have the same depth for batch processing
    
    Args:
        dir_A (str or Path): Directory containing domain A volumes (clean)
        dir_B (str or Path): Directory containing domain B volumes (noisy)
        volume_depth (int): Target depth for training volumes
        img_height (int): Target height for training
        img_width (int): Target width for training
        normalize_mean (float): Mean for normalization
        normalize_std (float): Standard deviation for normalization
        use_random_crop (bool): Use random 3D crops instead of resizing
        use_random_flip (bool): Apply random flips in all axes
        use_random_rotation (bool): Apply random 90° rotations
        clip_min (float): Minimum value for clipping
        clip_max (float): Maximum value for clipping
        use_percentile_norm (bool): Use percentile-based normalization
        percentile_low (float): Lower percentile for normalization
        percentile_high (float): Upper percentile for normalization
        extensions (list): List of valid image extensions
    """
    def __init__(self, dir_A, dir_B, 
                 volume_depth=32, img_height=128, img_width=128,
                 normalize_mean=0.5, normalize_std=0.5,
                 use_random_crop=True,
                 use_random_flip=True, use_random_rotation=False,
                 clip_min=None, clip_max=None,
                 use_percentile_norm=True, percentile_low=0.0, percentile_high=99.0,
                 extensions=['.tif', '.tiff']):
        super(Volume3DUnpairedDataset, self).__init__()
        
        if not HAS_TIFFFILE:
            raise ImportError("tifffile is required for 3D volume loading. Please install: pip install tifffile")
        
        self.dir_A = Path(dir_A)
        self.dir_B = Path(dir_B)
        self.volume_depth = volume_depth
        self.img_height = img_height
        self.img_width = img_width
        self.use_random_crop = use_random_crop
        self.use_random_flip = use_random_flip
        self.use_random_rotation = use_random_rotation
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.use_percentile_norm = use_percentile_norm
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        
        # Discover volumes
        self.volumes_A = self._discover_volumes(self.dir_A, extensions)
        self.volumes_B = self._discover_volumes(self.dir_B, extensions)
        
        if len(self.volumes_A) == 0:
            raise ValueError(f"No volumes found in {dir_A}")
        if len(self.volumes_B) == 0:
            raise ValueError(f"No volumes found in {dir_B}")
        
        print(f"Found {len(self.volumes_A)} volumes in domain A")
        print(f"Found {len(self.volumes_B)} volumes in domain B")
        print(f"Target volume size: {volume_depth}×{img_height}×{img_width}")
        print(f"Random crop: {use_random_crop}")
        
        # Normalization (applied after cropping)
        self.normalize = transforms.Normalize(mean=[normalize_mean], std=[normalize_std])
    
    def _discover_volumes(self, directory, extensions):
        """
        Discover 3D volume TIFF files in directory.
        
        Returns list of dicts with 'path'.
        """
        volumes = []
        
        for ext in extensions:
            for path in sorted(directory.glob(f'*{ext}')):
                try:
                    # Check if file is readable
                    img = tifffile.imread(path)
                    if len(img.shape) >= 2:  # At least 2D
                        volumes.append({'path': path})
                except Exception as e:
                    print(f"Warning: Could not read {path.name}: {e}")
        
        return volumes
    
    def __len__(self):
        """Return max length to iterate through all volumes in both domains."""
        return max(len(self.volumes_A), len(self.volumes_B))
    
    def __getitem__(self, index):
        """
        Get a pair of 3D volumes from domain A and domain B.
        
        Returns:
            dict: Dictionary containing:
                - 'A': [1, D, H, W] tensor (volume from domain A)
                - 'B': [1, D, H, W] tensor (volume from domain B)
                - 'A_path': path to source file
                - 'B_path': path to source file
        """
        # Get domain A volume (sequential)
        idx_A = index % len(self.volumes_A)
        volume_A = self._load_volume(self.volumes_A[idx_A]['path'], idx_A)
        
        # Get domain B volume (random for unpaired)
        idx_B = np.random.randint(0, len(self.volumes_B))
        volume_B = self._load_volume(self.volumes_B[idx_B]['path'], idx_B)
        
        return {
            'A': volume_A,
            'B': volume_B,
            'A_path': str(self.volumes_A[idx_A]['path']),
            'B_path': str(self.volumes_B[idx_B]['path'])
        }
    
    def _load_volume(self, file_path, idx=99999):
        """
        Load and process a 3D volume.
        
        Args:
            file_path: Path to TIFF file
            idx: Volume index for debugging (default 99999 = no debug)
        
        Returns:
            torch.Tensor: [1, D, H, W] normalized tensor
        """
        # Load the volume
        volume = tifffile.imread(file_path).astype(np.float32)
        
        # Handle different input shapes
        if len(volume.shape) == 2:
            # Single 2D image - replicate to create fake volume
            volume = np.stack([volume] * self.volume_depth, axis=0)
        elif len(volume.shape) == 3:
            # Already 3D (D, H, W)
            pass
        else:
            raise ValueError(f"Unexpected volume shape {volume.shape} for {file_path}")
        
        # Crop or resize to target size
        if self.use_random_crop:
            volume = self._random_crop_3d(volume)
        else:
            volume = self._resize_volume(volume)
        
        # DEBUG: Print raw data statistics (only for first few volumes)
        # Save raw stats and a copy for potential debug output later
        raw_min, raw_max, raw_mean = volume.min(), volume.max(), volume.mean()
        
        # Normalize to [0, 1] range
        # First check if already in [0, 1], otherwise normalize
        if volume.max() > 1.0 or volume.min() < 0.0:
            # Float data outside [0, 1] range (e.g., 0-6000 microscopy data)
            if self.use_percentile_norm:
                p_min = np.percentile(volume, self.percentile_low)
                p_max = np.percentile(volume, self.percentile_high)
                
                if p_max - p_min > 1e-8:
                    volume = (volume - p_min) / (p_max - p_min)
                    volume = np.clip(volume, 0.0, 1.0)
                else:
                    # Data has no dynamic range - use min-max normalization
                    v_min, v_max = volume.min(), volume.max()
                    if v_max > 1e-8:
                        volume = (volume - v_min) / (v_max - v_min + 1e-8)
                    else:
                        # Truly empty volume
                        volume = np.zeros_like(volume)
            else:
                # Min-max normalization
                v_min, v_max = volume.min(), volume.max()
                if v_max - v_min > 1e-8:
                    volume = (volume - v_min) / (v_max - v_min)
                else:
                    volume = np.zeros_like(volume)
        elif self.use_percentile_norm:
            # Already in [0, 1], but apply percentile norm for consistency
            p_min = np.percentile(volume, self.percentile_low)
            p_max = np.percentile(volume, self.percentile_high)
            if p_max - p_min > 1e-8:
                volume = (volume - p_min) / (p_max - p_min)
                volume = np.clip(volume, 0.0, 1.0)
        
        # Apply clipping
        if self.clip_min is not None or self.clip_max is not None:
            if self.clip_min is not None:
                volume = np.maximum(volume, self.clip_min)
            if self.clip_max is not None:
                volume = np.minimum(volume, self.clip_max)
        
        # Apply 3D augmentations
        if self.use_random_flip:
            volume = self._random_flip_3d(volume)
        
        if self.use_random_rotation:
            volume = self._random_rotation_3d(volume)
        
        # Convert to tensor [1, D, H, W]
        volume_tensor = torch.from_numpy(volume).unsqueeze(0)  # Add channel dimension
        
        # Apply normalization to [-1, 1]
        volume_tensor = self.normalize(volume_tensor)
        
        return volume_tensor
    
    def _random_crop_3d(self, volume):
        """
        Randomly crop a 3D sub-volume.
        
        Args:
            volume: numpy array of shape (D, H, W)
        
        Returns:
            Cropped volume of shape (volume_depth, img_height, img_width)
        """
        D, H, W = volume.shape
        
        # If volume is smaller than target, pad it
        if D < self.volume_depth:
            pad_d = self.volume_depth - D
            volume = np.pad(volume, ((0, pad_d), (0, 0), (0, 0)), mode='reflect')
            D = self.volume_depth
        
        if H < self.img_height:
            pad_h = self.img_height - H
            volume = np.pad(volume, ((0, 0), (0, pad_h), (0, 0)), mode='reflect')
            H = self.img_height
        
        if W < self.img_width:
            pad_w = self.img_width - W
            volume = np.pad(volume, ((0, 0), (0, 0), (0, pad_w)), mode='reflect')
            W = self.img_width
        
        # Random crop
        d_start = np.random.randint(0, D - self.volume_depth + 1) if D > self.volume_depth else 0
        h_start = np.random.randint(0, H - self.img_height + 1) if H > self.img_height else 0
        w_start = np.random.randint(0, W - self.img_width + 1) if W > self.img_width else 0
        
        cropped = volume[
            d_start:d_start + self.volume_depth,
            h_start:h_start + self.img_height,
            w_start:w_start + self.img_width
        ]
        
        return cropped
    
    def _resize_volume(self, volume):
        """
        Resize volume to target size (simple approach - could use proper 3D interpolation).
        
        Args:
            volume: numpy array of shape (D, H, W)
        
        Returns:
            Resized volume of shape (volume_depth, img_height, img_width)
        """
        # This is a simple implementation - for production, consider using scipy.ndimage.zoom
        from scipy import ndimage
        
        D, H, W = volume.shape
        zoom_factors = (self.volume_depth / D, self.img_height / H, self.img_width / W)
        resized = ndimage.zoom(volume, zoom_factors, order=1)
        
        return resized
    
    def _random_flip_3d(self, volume):
        """
        Randomly flip volume along each axis.
        
        Args:
            volume: numpy array of shape (D, H, W)
        
        Returns:
            Flipped volume
        """
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=0).copy()  # Flip depth
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=1).copy()  # Flip height
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=2).copy()  # Flip width
        
        return volume
    
    def _random_rotation_3d(self, volume):
        """
        Randomly rotate volume by 90° around each axis.
        
        Args:
            volume: numpy array of shape (D, H, W)
        
        Returns:
            Rotated volume
        """
        # Random 90° rotations in each plane
        k = np.random.randint(0, 4)  # 0, 90, 180, or 270 degrees
        if k > 0:
            # Rotate in XY plane (along D axis)
            volume = np.rot90(volume, k=k, axes=(1, 2)).copy()
        
        return volume


def split_dataset_indices(dataset, val_split=0.2, random_seed=42):
    """
    Split dataset indices into training and validation sets.
    
    Args:
        dataset: PyTorch dataset
        val_split (float): Fraction for validation
        random_seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_indices, val_indices)
    """
    np.random.seed(random_seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    
    split_idx = int(len(indices) * (1 - val_split))
    train_indices = indices[:split_idx].tolist()
    val_indices = indices[split_idx:].tolist()
    
    return train_indices, val_indices


def get_dataloaders_3d(config, auto_split=False, val_split=0.2):
    """
    Create 3D training and validation dataloaders.
    
    Supports two modes:
    1. Pre-split: Separate directories for train and validation (default)
    2. Auto-split: Single directory that gets split automatically
    
    Args:
        config: Configuration object with 3D settings
        auto_split (bool): If True, automatically split data from single directories
        val_split (float): Fraction of data for validation when auto_split=True
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader, Subset
    
    print(f"\n{'='*60}")
    print("Creating 3D Volume Dataloaders")
    print(f"{'='*60}\n")
    
    if auto_split:
        # ===== AUTO-SPLIT MODE =====
        # Load all data from a single directory and split automatically
        print(f"\n{'='*60}")
        print("AUTO-SPLIT MODE: Splitting data automatically")
        print(f"Validation split: {val_split*100:.1f}%")
        print(f"{'='*60}\n")
        
        # Create full dataset with augmentation
        full_dataset = Volume3DUnpairedDataset(
            dir_A=config.TRAIN_A_DIR,
            dir_B=config.TRAIN_B_DIR,
            volume_depth=config.VOLUME_DEPTH,
            img_height=config.IMG_HEIGHT,
            img_width=config.IMG_WIDTH,
            normalize_mean=config.NORMALIZE_MEAN,
            normalize_std=config.NORMALIZE_STD,
            use_random_crop=config.USE_RANDOM_CROP,
            use_random_flip=config.USE_RANDOM_FLIP,
            use_random_rotation=config.USE_RANDOM_ROTATION,
            clip_min=config.CLIP_MIN,
            clip_max=config.CLIP_MAX,
            use_percentile_norm=config.USE_PERCENTILE_NORM,
            percentile_low=config.PERCENTILE_LOW,
            percentile_high=config.PERCENTILE_HIGH
        )
        
        # Split indices
        train_indices, val_indices = split_dataset_indices(
            full_dataset,
            val_split=val_split,
            random_seed=config.RANDOM_SEED
        )
        
        print(f"Total volumes: {len(full_dataset)}")
        print(f"Training volumes: {len(train_indices)}")
        print(f"Validation volumes: {len(val_indices)}\n")
        
        # Create training subset
        train_dataset = Subset(full_dataset, train_indices)
        
        # Create validation dataset (no augmentation)
        val_dataset_full = Volume3DUnpairedDataset(
            dir_A=config.TRAIN_A_DIR,  # Same directory
            dir_B=config.TRAIN_B_DIR,  # Same directory
            volume_depth=config.VOLUME_DEPTH,
            img_height=config.IMG_HEIGHT,
            img_width=config.IMG_WIDTH,
            normalize_mean=config.NORMALIZE_MEAN,
            normalize_std=config.NORMALIZE_STD,
            use_random_crop=False,  # No augmentation for validation
            use_random_flip=False,
            use_random_rotation=False,
            clip_min=config.CLIP_MIN,
            clip_max=config.CLIP_MAX,
            use_percentile_norm=config.USE_PERCENTILE_NORM,
            percentile_low=config.PERCENTILE_LOW,
            percentile_high=config.PERCENTILE_HIGH
        )
        val_dataset = Subset(val_dataset_full, val_indices)
        
    else:
        # ===== PRE-SPLIT MODE =====
        # Use separate directories for train and validation
        print(f"\n{'='*60}")
        print("PRE-SPLIT MODE: Using separate train/val directories")
        print(f"{'='*60}\n")
        
        # Training dataset
        train_dataset = Volume3DUnpairedDataset(
            dir_A=config.TRAIN_A_DIR,
            dir_B=config.TRAIN_B_DIR,
            volume_depth=config.VOLUME_DEPTH,
            img_height=config.IMG_HEIGHT,
            img_width=config.IMG_WIDTH,
            normalize_mean=config.NORMALIZE_MEAN,
            normalize_std=config.NORMALIZE_STD,
            use_random_crop=config.USE_RANDOM_CROP,
            use_random_flip=config.USE_RANDOM_FLIP,
            use_random_rotation=config.USE_RANDOM_ROTATION,
            clip_min=config.CLIP_MIN,
            clip_max=config.CLIP_MAX,
            use_percentile_norm=config.USE_PERCENTILE_NORM,
            percentile_low=config.PERCENTILE_LOW,
            percentile_high=config.PERCENTILE_HIGH
        )
        
        # Validation dataset (no augmentation)
        val_dataset = Volume3DUnpairedDataset(
            dir_A=config.VAL_A_DIR,
            dir_B=config.VAL_B_DIR,
            volume_depth=config.VOLUME_DEPTH,
            img_height=config.IMG_HEIGHT,
            img_width=config.IMG_WIDTH,
            normalize_mean=config.NORMALIZE_MEAN,
            normalize_std=config.NORMALIZE_STD,
            use_random_crop=False,  # No random crop for validation
            use_random_flip=False,
            use_random_rotation=False,
            clip_min=config.CLIP_MIN,
            clip_max=config.CLIP_MAX,
            use_percentile_norm=config.USE_PERCENTILE_NORM,
            percentile_low=config.PERCENTILE_LOW,
            percentile_high=config.PERCENTILE_HIGH
        )
    
    # Create dataloaders with HPC optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY if hasattr(config, 'PIN_MEMORY') else (True if config.DEVICE == 'cuda' else False),
        drop_last=True,  # Drop last incomplete batch
        prefetch_factor=config.PREFETCH_FACTOR if hasattr(config, 'PREFETCH_FACTOR') else 2,
        persistent_workers=config.PERSISTENT_WORKERS if hasattr(config, 'PERSISTENT_WORKERS') and config.NUM_WORKERS > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY if hasattr(config, 'PIN_MEMORY') else (True if config.DEVICE == 'cuda' else False),
        prefetch_factor=config.PREFETCH_FACTOR if hasattr(config, 'PREFETCH_FACTOR') else 2,
        persistent_workers=config.PERSISTENT_WORKERS if hasattr(config, 'PERSISTENT_WORKERS') and config.NUM_WORKERS > 0 else False
    )
    
    return train_loader, val_loader


def test_dataset_3d():
    """Test the 3D dataset."""
    print("Testing Volume3DUnpairedDataset...")
    print("\nNote: This test requires actual 3D volume directories to run.")
    print("Create data/3d/trainA and data/3d/trainB with TIFF volumes.")


if __name__ == "__main__":
    test_dataset_3d()
