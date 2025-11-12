"""
Dataset loader for unpaired microscopy images.
Supports both pre-split and automatic train/val splitting.
Supports Z-stack context for multi-plane denoising.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    print("Warning: tifffile not available. Z-stack support will be limited.")


class UnpairedMicroscopyDataset(Dataset):
    """
    Dataset for unpaired microscopy images.
    
    Loads images from two separate directories (domain A and domain B).
    Images are not paired - they can be sampled independently.
    
    Args:
        dir_A (str or Path): Directory containing domain A images (clean)
        dir_B (str or Path): Directory containing domain B images (noisy)
        img_size (int): Size to resize images to
        normalize_mean (float): Mean for normalization
        normalize_std (float): Standard deviation for normalization
        use_random_flip (bool): Apply random horizontal flip
        use_random_rotation (bool): Apply random rotation
        clip_min (float): Minimum value for clipping (None for no clipping)
        clip_max (float): Maximum value for clipping (None for no clipping)
        use_percentile_norm (bool): Use percentile-based normalization (default True for fluorescence)
        percentile_low (float): Lower percentile for normalization (default 1.0)
        percentile_high (float): Upper percentile for normalization (default 99.0)
        extensions (list): List of valid image extensions
    """
    def __init__(self, dir_A, dir_B, img_size=256, 
                 normalize_mean=0.5, normalize_std=0.5,
                 use_random_flip=True, use_random_rotation=False,
                 clip_min=None, clip_max=None,
                 use_percentile_norm=True, percentile_low=0.0, percentile_high=99.0,
                 extensions=['.tif', '.tiff', '.png', '.jpg', '.jpeg']):
        super(UnpairedMicroscopyDataset, self).__init__()
        
        self.dir_A = Path(dir_A)
        self.dir_B = Path(dir_B)
        self.img_size = img_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.use_percentile_norm = use_percentile_norm
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        
        # Get list of image files
        self.files_A = self._get_image_files(self.dir_A, extensions)
        self.files_B = self._get_image_files(self.dir_B, extensions)
        
        if len(self.files_A) == 0:
            raise ValueError(f"No images found in {dir_A}")
        if len(self.files_B) == 0:
            raise ValueError(f"No images found in {dir_B}")
        
        print(f"Found {len(self.files_A)} images in domain A")
        print(f"Found {len(self.files_B)} images in domain B")
        
        # Define transforms
        transform_list = [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        ]
        
        if use_random_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        
        if use_random_rotation:
            transform_list.append(transforms.RandomRotation(10))
        
        transform_list.extend([
            transforms.ToTensor(),  # Converts to [0, 1] range
        ])
        
        # Add normalization to [-1, 1] range (expected by Tanh output)
        transform_list.append(transforms.Normalize(mean=[normalize_mean], std=[normalize_std]))
        
        self.transform = transforms.Compose(transform_list)
    
    def _get_image_files(self, directory, extensions):
        """Get all image files in a directory with valid extensions."""
        image_files = []
        for ext in extensions:
            image_files.extend(sorted(directory.glob(f'*{ext}')))
            image_files.extend(sorted(directory.glob(f'*{ext.upper()}')))
        return image_files
    
    def __len__(self):
        """
        Return the maximum length of the two domains.
        This ensures we iterate through all images in both domains.
        """
        return max(len(self.files_A), len(self.files_B))
    
    def __getitem__(self, index):
        """
        Get a pair of images from domain A and domain B.
        
        Since the dataset is unpaired, we sample randomly from each domain.
        Domain A is accessed sequentially, while domain B is accessed randomly
        to ensure proper unpaired training.
        
        Args:
            index (int): Index
        
        Returns:
            dict: Dictionary containing 'A' and 'B' images
        """
        # Get domain A image (sequential access)
        idx_A = index % len(self.files_A)
        img_A = self._load_image(self.files_A[idx_A])
        
        # Get domain B image (random access for unpaired training)
        idx_B = np.random.randint(0, len(self.files_B))
        img_B = self._load_image(self.files_B[idx_B])
        
        return {
            'A': img_A,
            'B': img_B,
            'A_path': str(self.files_A[idx_A]),
            'B_path': str(self.files_B[idx_B])
        }
    
    def _load_image(self, path):
        """
        Load an image and apply transformations.
        
        Handles multiple image formats:
        - Pre-normalized float32/float64 images (0-1 range) - TIFF microscopy
        - 16-bit TIFF images (0-65535 range) - normalized to [0, 1]
        - 8-bit uint8 images (0-255 range) - standard images
        
        Args:
            path (Path): Path to image file
        
        Returns:
            torch.Tensor: Transformed image tensor
        """
        # Load image
        img = Image.open(path)
        
        # Convert to numpy for processing
        img_array = np.array(img)
        
        # Convert to float for normalization
        if img_array.dtype == np.uint16:
            img_array = img_array.astype(np.float32)
        elif img_array.dtype == np.uint8:
            img_array = img_array.astype(np.float32)
        elif img_array.dtype not in [np.float32, np.float64]:
            img_array = img_array.astype(np.float32)
        else:
            img_array = img_array.astype(np.float32)
        
        # Normalize to [0, 1] range
        if self.use_percentile_norm:
            # Percentile-based normalization (robust to outliers in fluorescence)
            p_min = np.percentile(img_array, self.percentile_low)
            p_max = np.percentile(img_array, self.percentile_high)
            
            if p_max - p_min < 1e-8:
                # Very low contrast - avoid division by zero
                img_array = np.zeros_like(img_array)
            else:
                # Normalize based on percentiles and clip
                img_array = (img_array - p_min) / (p_max - p_min)
                img_array = np.clip(img_array, 0.0, 1.0)
        else:
            # Simple min-max normalization (legacy)
            img_min = img_array.min()
            img_max = img_array.max()
            
            if img_max - img_min < 1e-8:
                img_array = np.zeros_like(img_array)
            else:
                img_array = (img_array - img_min) / (img_max - img_min)
        
        # Apply clipping if specified
        if self.clip_min is not None or self.clip_max is not None:
            if img_array.dtype != np.uint8:  # Already normalized
                if self.clip_min is not None:
                    img_array = np.maximum(img_array, self.clip_min)
                if self.clip_max is not None:
                    img_array = np.minimum(img_array, self.clip_max)
        
        # Convert to PIL Image format for transforms
        if img_array.dtype in [np.float32, np.float64]:
            # Convert float [0, 1] to uint8 [0, 255] for PIL
            img_array = (img_array * 255).astype(np.uint8)
        
        # Convert to grayscale PIL Image
        if len(img_array.shape) == 3:
            # If multi-dimensional, take first channel or mean
            img_array = img_array[..., 0] if img_array.shape[2] == 1 else np.mean(img_array, axis=2).astype(np.uint8)
        
        img = Image.fromarray(img_array, mode='L')
        
        # Apply transforms (ToTensor converts to [0, 1], then Normalize converts to [-1, 1])
        img = self.transform(img)
        
        return img


class ZStackUnpairedDataset(Dataset):
    """
    Dataset for unpaired Z-stack microscopy images with multi-plane context.
    
    Expects PRE-STACKED TIFF files where each file contains N adjacent Z-planes
    already grouped together. Each file should have shape (N, H, W) where N = zstack_context.
    
    This format is more efficient than loading multi-page stacks and extracting windows,
    as the data preparation step (creating N-plane stacks) is done once beforehand.
    
    Expected file format:
    - TIFF files with shape (zstack_context, H, W), e.g., (5, 128, 128)
    - Example: sample001.tif contains 5 adjacent planes pre-stacked
    
    Args:
        dir_A (str or Path): Directory containing domain A Z-stacks (clean)
        dir_B (str or Path): Directory containing domain B Z-stacks (noisy)
        zstack_context (int): Number of adjacent planes to use (3, 5, 7, etc.)
        img_size (int): Size to resize images to
        normalize_mean (float): Mean for normalization
        normalize_std (float): Standard deviation for normalization
        use_random_flip (bool): Apply random horizontal flip
        use_random_rotation (bool): Apply random rotation
        clip_min (float): Minimum value for clipping
        clip_max (float): Maximum value for clipping
        use_percentile_norm (bool): Use percentile-based normalization instead of min-max (default: True)
        percentile_low (float): Lower percentile for normalization (default: 0.0)
        percentile_high (float): Upper percentile for normalization (default: 99.0)
        extensions (list): List of valid image extensions
    """
    def __init__(self, dir_A, dir_B, zstack_context=5, img_size=256,
                 normalize_mean=0.5, normalize_std=0.5,
                 use_random_flip=True, use_random_rotation=False,
                 clip_min=None, clip_max=None,
                 use_percentile_norm=True, percentile_low=0.0, percentile_high=99.0,
                 extensions=['.tif', '.tiff']):
        super(ZStackUnpairedDataset, self).__init__()
        
        self.dir_A = Path(dir_A)
        self.dir_B = Path(dir_B)
        self.zstack_context = zstack_context
        self.half_context = zstack_context // 2
        self.img_size = img_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.use_percentile_norm = use_percentile_norm
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        
        # Load Z-stack metadata
        self.zstacks_A = self._discover_zstacks(self.dir_A, extensions)
        self.zstacks_B = self._discover_zstacks(self.dir_B, extensions)
        
        if len(self.zstacks_A) == 0:
            raise ValueError(f"No Z-stacks found in {dir_A}")
        if len(self.zstacks_B) == 0:
            raise ValueError(f"No Z-stacks found in {dir_B}")
        
        # Create index mapping (stack_id, z_index) for valid planes
        self.valid_indices_A = self._create_valid_indices(self.zstacks_A)
        self.valid_indices_B = self._create_valid_indices(self.zstacks_B)
        
        print(f"Found {len(self.zstacks_A)} Z-stacks in domain A ({len(self.valid_indices_A)} valid planes)")
        print(f"Found {len(self.zstacks_B)} Z-stacks in domain B ({len(self.valid_indices_B)} valid planes)")
        print(f"Using {zstack_context}-plane context window")
        
        # Define transforms for each plane
        transform_list = [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        ]
        
        if use_random_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        
        if use_random_rotation:
            transform_list.append(transforms.RandomRotation(10))
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[normalize_mean], std=[normalize_std])
        ])
        
        self.transform = transforms.Compose(transform_list)
    
    def _discover_zstacks(self, directory, extensions):
        """
        Discover pre-stacked Z-context TIFFs in directory.
        Expects TIFFs with shape (N, H, W) where N = zstack_context.
        
        Returns list of dicts with 'path'.
        """
        zstacks = []
        
        # Look for TIFF files with correct shape
        for ext in extensions:
            print('looking')
            for path in sorted(directory.glob(f'*{ext}')):
                try:
                    if HAS_TIFFFILE:
                        # Load to check shape
                        img = tifffile.imread(path)
                        print(img.shape)
                        if len(img.shape) == 3 and img.shape[0] == self.zstack_context:
                            # Correct shape: (zstack_context, H, W)
                            zstacks.append({'path': path})
                        elif len(img.shape) == 2:
                            print(f"Warning: Skipping {path.name} - single plane image (needs shape ({self.zstack_context}, H, W))")
                        else:
                            print(f"Warning: Skipping {path.name} - unexpected shape {img.shape} (needs ({self.zstack_context}, H, W))")
                    else:
                        print("Warning: tifffile not available, cannot verify shapes")
                        # Assume files are correct format
                        zstacks.append({'path': path})
                except Exception as e:
                    print(f"Warning: Could not read {path.name}: {e}")
        
        if len(zstacks) == 0:
            raise ValueError(f"No valid Z-stack TIFFs found in {directory}. "
                           f"Expected TIFF files with shape ({self.zstack_context}, H, W).")
        
        return zstacks
    
    def _create_valid_indices(self, zstacks):
        """
        Create list of indices for pre-stacked Z-context files.
        Each file is already a complete (N, H, W) stack.
        """
        # Simply return list of stack indices (0, 1, 2, ...)
        return list(range(len(zstacks)))
    
    def __len__(self):
        """Return max length to iterate through all stacks in both domains."""
        return max(len(self.valid_indices_A), len(self.valid_indices_B))
    
    def __getitem__(self, index):
        """
        Get a pair of pre-stacked Z-contexts from domain A and domain B.
        
        Returns:
            dict: Dictionary containing:
                - 'A': [zstack_context, H, W] tensor (multiple planes as channels)
                - 'B': [zstack_context, H, W] tensor
                - 'A_path': path to source file
                - 'B_path': path to source file
        """
        # Get domain A stack (sequential)
        idx_A = index % len(self.valid_indices_A)
        stack_idx_A = self.valid_indices_A[idx_A]
        context_A = self._load_prestacked_file(self.zstacks_A[stack_idx_A]['path'])
        
        # Get domain B stack (random for unpaired)
        idx_B = np.random.randint(0, len(self.valid_indices_B))
        stack_idx_B = self.valid_indices_B[idx_B]
        context_B = self._load_prestacked_file(self.zstacks_B[stack_idx_B]['path'])
        
        return {
            'A': context_A,
            'B': context_B,
            'A_path': str(self.zstacks_A[stack_idx_A]['path']),
            'B_path': str(self.zstacks_B[stack_idx_B]['path'])
        }
    
    def _load_prestacked_file(self, file_path):
        """
        Load a pre-stacked Z-context TIFF file with shape (N, H, W).
        
        Args:
            file_path: Path to TIFF file with shape (zstack_context, H, W)
        
        Returns:
            torch.Tensor: [zstack_context, H, W] transformed and normalized tensor
        """
        # Load the pre-stacked file
        if HAS_TIFFFILE:
            img_array = tifffile.imread(file_path)
        else:
            # Fallback to PIL (may not work for 3D arrays)
            raise ImportError("tifffile is required for Z-stack loading. Please install: pip install tifffile")
        
        # Verify shape
        if len(img_array.shape) != 3 or img_array.shape[0] != self.zstack_context:
            raise ValueError(f"Expected shape ({self.zstack_context}, H, W), got {img_array.shape} for {file_path}")
        
        # Process each plane individually and stack
        planes = []
        for z in range(self.zstack_context):
            plane = img_array[z]  # [H, W]
            
            # Convert to float for normalization
            if plane.dtype == np.uint16:
                plane = plane.astype(np.float32) / 65535.0
            elif plane.dtype == np.uint8:
                plane = plane.astype(np.float32) / 255.0
            elif plane.dtype not in [np.float32, np.float64]:
                plane = plane.astype(np.float32)
                if plane.max() > 1.0 or plane.min() < 0.0:
                    plane = (plane - plane.min()) / (plane.max() - plane.min() + 1e-8)
            else:
                plane = plane.astype(np.float32)
            
            # Apply percentile-based or min-max normalization
            if self.use_percentile_norm:
                # Use percentile normalization for robust handling of outliers
                p_min = np.percentile(plane, self.percentile_low)
                p_max = np.percentile(plane, self.percentile_high)
                if p_max - p_min > 1e-8:
                    plane = (plane - p_min) / (p_max - p_min)
                    plane = np.clip(plane, 0.0, 1.0)
            else:
                # Traditional min-max normalization
                if plane.max() - plane.min() > 1e-8:
                    plane = (plane - plane.min()) / (plane.max() - plane.min())
            
            # Apply clipping if specified
            if self.clip_min is not None or self.clip_max is not None:
                if self.clip_min is not None:
                    plane = np.maximum(plane, self.clip_min)
                if self.clip_max is not None:
                    plane = np.minimum(plane, self.clip_max)
            
            # Convert to uint8 for PIL transforms
            if plane.dtype in [np.float32, np.float64]:
                plane = (plane * 255).astype(np.uint8)
            
            # Convert to PIL
            if len(plane.shape) == 2:
                img = Image.fromarray(plane, mode='L')
            else:
                img = Image.fromarray(plane)
            
            # Apply transforms (returns [1, H, W])
            plane_tensor = self.transform(img)
            planes.append(plane_tensor)
        
        # Stack along channel dimension: [zstack_context, H, W]
        stacked = torch.cat(planes, dim=0)
        
        return stacked


class InferenceMicroscopyDataset(Dataset):
    """
    Dataset for inference on microscopy images.
    
    Loads images from a single directory for restoration.
    
    Args:
        directory (str or Path): Directory containing images to restore
        img_size (int): Size to resize images to
        normalize_mean (float): Mean for normalization
        normalize_std (float): Standard deviation for normalization
        extensions (list): List of valid image extensions
    """
    def __init__(self, directory, img_size=256, normalize_mean=0.5, normalize_std=0.5,
                 extensions=['.png', '.jpg', '.jpeg', '.tif', '.tiff']):
        super(InferenceMicroscopyDataset, self).__init__()
        
        self.directory = Path(directory)
        self.img_size = img_size
        
        # Get list of image files
        self.files = self._get_image_files(self.directory, extensions)
        
        if len(self.files) == 0:
            raise ValueError(f"No images found in {directory}")
        
        print(f"Found {len(self.files)} images for inference")
        
        # Define transforms (no augmentation for inference)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[normalize_mean], std=[normalize_std])
        ])
    
    def _get_image_files(self, directory, extensions):
        """Get all image files in a directory with valid extensions."""
        image_files = []
        for ext in extensions:
            image_files.extend(sorted(directory.glob(f'*{ext}')))
            image_files.extend(sorted(directory.glob(f'*{ext.upper()}')))
        return image_files
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        """
        Get an image for inference.
        
        Args:
            index (int): Index
        
        Returns:
            dict: Dictionary containing image and metadata
        """
        path = self.files[index]
        img = self._load_image(path)
        
        return {
            'image': img,
            'path': str(path),
            'filename': path.name
        }
    
    def _load_image(self, path):
        """
        Load an image and apply transformations.
        Handles float32, uint16 TIFF, and uint8 images.
        """
        img = Image.open(path)
        img_array = np.array(img)
        
        # Handle different image types
        if img_array.dtype in [np.float32, np.float64]:
            # Float image - pre-normalized to [0, 1]
            if img_array.max() > 1.0 or img_array.min() < 0.0:
                print(f"Warning: Float image {path.name} not in [0,1] range, normalizing...")
                img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
            img_array = (img_array * 255).astype(np.uint8)
        
        elif img_array.dtype == np.uint16:
            # 16-bit TIFF - normalize to [0, 1] then to uint8
            img_array = (img_array.astype(np.float32) / 65535.0 * 255).astype(np.uint8)
        
        # Convert to grayscale PIL Image
        if len(img_array.shape) == 3:
            # If multi-dimensional, take first channel or mean
            img_array = img_array[..., 0] if img_array.shape[2] == 1 else np.mean(img_array, axis=2).astype(np.uint8)
        
        img = Image.fromarray(img_array, mode='L')
        
        img = self.transform(img)
        return img


def split_dataset_indices(dataset, val_split=0.2, random_seed=42):
    """
    Split dataset indices into train and validation sets.
    
    Args:
        dataset: Dataset object
        val_split (float): Fraction of data to use for validation (0.0 to 1.0)
        random_seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_indices, val_indices)
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    if val_split > 0:
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_split,
            random_state=random_seed,
            shuffle=True
        )
    else:
        train_indices = indices
        val_indices = []
    
    return train_indices, val_indices


def get_dataloaders(config, auto_split=False, val_split=0.2):
    """
    Create training and validation dataloaders.
    
    Supports three modes:
    1. Pre-split: Separate directories for train and validation (default)
    2. Auto-split: Single directory that gets split automatically
    3. Z-stack mode: Multi-plane context for denoising (if config.ZSTACK_MODE=True)
    
    Args:
        config: Configuration object
        auto_split (bool): If True, automatically split data from single directories
        val_split (float): Fraction of data for validation when auto_split=True
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    
    # Check if Z-stack mode is enabled
    use_zstack = getattr(config, 'ZSTACK_MODE', False) and getattr(config, 'ZSTACK_CONTEXT', 1) > 1
    
    if use_zstack:
        print(f"\n{'='*60}")
        print(f"Z-STACK MODE: Using {config.ZSTACK_CONTEXT}-plane context")
        print(f"Input channels: {config.IMG_CHANNELS}")
        print(f"{'='*60}\n")
    
    if auto_split:
        # ===== AUTO-SPLIT MODE =====
        # Load all data from a single directory and split automatically
        print(f"\n{'='*60}")
        print("AUTO-SPLIT MODE: Splitting data automatically")
        print(f"Validation split: {val_split*100:.1f}%")
        print(f"{'='*60}\n")
        
        # Create full dataset with augmentation
        if use_zstack:
            full_dataset_A = ZStackUnpairedDataset(
                dir_A=config.TRAIN_A_DIR,
                dir_B=config.TRAIN_B_DIR,
                zstack_context=config.ZSTACK_CONTEXT,
                img_size=config.IMG_SIZE,
                normalize_mean=config.NORMALIZE_MEAN,
                normalize_std=config.NORMALIZE_STD,
                use_random_flip=config.USE_RANDOM_FLIP,
                use_random_rotation=config.USE_RANDOM_ROTATION,
                clip_min=config.CLIP_MIN,
                clip_max=config.CLIP_MAX,
                use_percentile_norm=getattr(config, 'USE_PERCENTILE_NORM', True),
                percentile_low=getattr(config, 'PERCENTILE_LOW', 0.0),
                percentile_high=getattr(config, 'PERCENTILE_HIGH', 99.0)
            )
        else:
            full_dataset_A = UnpairedMicroscopyDataset(
                dir_A=config.TRAIN_A_DIR,
                dir_B=config.TRAIN_B_DIR,
                img_size=config.IMG_SIZE,
                normalize_mean=config.NORMALIZE_MEAN,
                normalize_std=config.NORMALIZE_STD,
                use_random_flip=config.USE_RANDOM_FLIP,
                use_random_rotation=config.USE_RANDOM_ROTATION,
                clip_min=config.CLIP_MIN,
                clip_max=config.CLIP_MAX,
                use_percentile_norm=getattr(config, 'USE_PERCENTILE_NORM', True),
                percentile_low=getattr(config, 'PERCENTILE_LOW', 0.0),
                percentile_high=getattr(config, 'PERCENTILE_HIGH', 99.0)
            )
        
        # Split indices
        train_indices, val_indices = split_dataset_indices(
            full_dataset_A,
            val_split=val_split,
            random_seed=config.RANDOM_SEED
        )
        
        print(f"Total images: {len(full_dataset_A)}")
        print(f"Training images: {len(train_indices)}")
        print(f"Validation images: {len(val_indices)}\n")
        
        # Create training subset
        train_dataset = Subset(full_dataset_A, train_indices)
        
        # Create validation dataset (no augmentation)
        if use_zstack:
            val_dataset_full = ZStackUnpairedDataset(
                dir_A=config.TRAIN_A_DIR,  # Same directory
                dir_B=config.TRAIN_B_DIR,  # Same directory
                zstack_context=config.ZSTACK_CONTEXT,
                img_size=config.IMG_SIZE,
                normalize_mean=config.NORMALIZE_MEAN,
                normalize_std=config.NORMALIZE_STD,
                use_random_flip=False,  # No augmentation for validation
                use_random_rotation=False,
                clip_min=config.CLIP_MIN,
                clip_max=config.CLIP_MAX,
                use_percentile_norm=getattr(config, 'USE_PERCENTILE_NORM', True),
                percentile_low=getattr(config, 'PERCENTILE_LOW', 0.0),
                percentile_high=getattr(config, 'PERCENTILE_HIGH', 99.0)
            )
        else:
            val_dataset_full = UnpairedMicroscopyDataset(
                dir_A=config.TRAIN_A_DIR,  # Same directory
                dir_B=config.TRAIN_B_DIR,  # Same directory
                img_size=config.IMG_SIZE,
                normalize_mean=config.NORMALIZE_MEAN,
                normalize_std=config.NORMALIZE_STD,
                use_random_flip=False,  # No augmentation for validation
                use_random_rotation=False,
                clip_min=config.CLIP_MIN,
                clip_max=config.CLIP_MAX,
                use_percentile_norm=getattr(config, 'USE_PERCENTILE_NORM', True),
                percentile_low=getattr(config, 'PERCENTILE_LOW', 0.0),
                percentile_high=getattr(config, 'PERCENTILE_HIGH', 99.0)
            )
        val_dataset = Subset(val_dataset_full, val_indices)
        
    else:
        # ===== PRE-SPLIT MODE =====
        # Use separate directories for train and validation
        print(f"\n{'='*60}")
        print("PRE-SPLIT MODE: Using separate train/val directories")
        print(f"{'='*60}\n")
        
        # Training dataset
        if use_zstack:
            train_dataset = ZStackUnpairedDataset(
                dir_A=config.TRAIN_A_DIR,
                dir_B=config.TRAIN_B_DIR,
                zstack_context=config.ZSTACK_CONTEXT,
                img_size=config.IMG_SIZE,
                normalize_mean=config.NORMALIZE_MEAN,
                normalize_std=config.NORMALIZE_STD,
                use_random_flip=config.USE_RANDOM_FLIP,
                use_random_rotation=config.USE_RANDOM_ROTATION,
                clip_min=config.CLIP_MIN,
                clip_max=config.CLIP_MAX,
                use_percentile_norm=getattr(config, 'USE_PERCENTILE_NORM', True),
                percentile_low=getattr(config, 'PERCENTILE_LOW', 0.0),
                percentile_high=getattr(config, 'PERCENTILE_HIGH', 99.0)
            )
        else:
            train_dataset = UnpairedMicroscopyDataset(
                dir_A=config.TRAIN_A_DIR,
                dir_B=config.TRAIN_B_DIR,
                img_size=config.IMG_SIZE,
                normalize_mean=config.NORMALIZE_MEAN,
                normalize_std=config.NORMALIZE_STD,
                use_random_flip=config.USE_RANDOM_FLIP,
                use_random_rotation=config.USE_RANDOM_ROTATION,
                clip_min=config.CLIP_MIN,
                clip_max=config.CLIP_MAX,
                use_percentile_norm=getattr(config, 'USE_PERCENTILE_NORM', True),
                percentile_low=getattr(config, 'PERCENTILE_LOW', 0.0),
                percentile_high=getattr(config, 'PERCENTILE_HIGH', 99.0)
            )
        
        # Validation dataset
        if use_zstack:
            val_dataset = ZStackUnpairedDataset(
                dir_A=config.VAL_A_DIR,
                dir_B=config.VAL_B_DIR,
                zstack_context=config.ZSTACK_CONTEXT,
                img_size=config.IMG_SIZE,
                normalize_mean=config.NORMALIZE_MEAN,
                normalize_std=config.NORMALIZE_STD,
                use_random_flip=False,  # No augmentation for validation
                use_random_rotation=False,
                clip_min=config.CLIP_MIN,
                clip_max=config.CLIP_MAX,
                use_percentile_norm=getattr(config, 'USE_PERCENTILE_NORM', True),
                percentile_low=getattr(config, 'PERCENTILE_LOW', 0.0),
                percentile_high=getattr(config, 'PERCENTILE_HIGH', 99.0)
            )
        else:
            val_dataset = UnpairedMicroscopyDataset(
                dir_A=config.VAL_A_DIR,
                dir_B=config.VAL_B_DIR,
                img_size=config.IMG_SIZE,
                normalize_mean=config.NORMALIZE_MEAN,
                normalize_std=config.NORMALIZE_STD,
                use_random_flip=False,  # No augmentation for validation
                use_random_rotation=False,
                clip_min=config.CLIP_MIN,
                clip_max=config.CLIP_MAX,
                use_percentile_norm=getattr(config, 'USE_PERCENTILE_NORM', True),
                percentile_low=getattr(config, 'PERCENTILE_LOW', 0.0),
                percentile_high=getattr(config, 'PERCENTILE_HIGH', 99.0)
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


if __name__ == "__main__":
    # Test the dataset
    print("Testing UnpairedMicroscopyDataset...")
    
    # This is just a test - you need to have actual data directories
    print("\nNote: This test requires actual image directories to run.")
    print("Create data/trainA, data/trainB, data/valA, data/valB with images.")
