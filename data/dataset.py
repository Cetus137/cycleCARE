"""
Dataset loader for unpaired microscopy images.
Supports both pre-split and automatic train/val splitting.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


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
        extensions (list): List of valid image extensions
    """
    def __init__(self, dir_A, dir_B, img_size=256, normalize_mean=0.5, normalize_std=0.5,
                 use_random_flip=True, use_random_rotation=False,
                 clip_min=None, clip_max=None,
                 extensions=['.png', '.jpg', '.jpeg', '.tif', '.tiff']):
        super(UnpairedMicroscopyDataset, self).__init__()
        
        self.dir_A = Path(dir_A)
        self.dir_B = Path(dir_B)
        self.img_size = img_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        
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
        
        # Determine image type and normalize appropriately
        if img_array.dtype in [np.float32, np.float64]:
            # Float image - already normalized to [0, 1] (common for TIF microscopy)
            if img_array.max() <= 1.0 and img_array.min() >= 0.0:
                # Already in [0, 1] range - perfect!
                pass
            else:
                # Unexpected range - normalize to [0, 1]
                print(f"Warning: Float image {path.name} not in [0,1] range: [{img_array.min():.3f}, {img_array.max():.3f}]")
                img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
        
        elif img_array.dtype == np.uint16:
            # 16-bit TIFF (common in microscopy) - normalize to [0, 1]
            img_array = img_array.astype(np.float32) / 65535.0
        
        elif img_array.dtype == np.uint8:
            # 8-bit image - will be normalized by ToTensor to [0, 1]
            pass
        
        else:
            # Unknown dtype - try to normalize
            print(f"Warning: Unknown image dtype {img_array.dtype} for {path.name}, normalizing...")
            img_array = img_array.astype(np.float32)
            img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
        
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
        
        # Ensure proper mode
        if len(img_array.shape) == 2:  # Grayscale
            img = Image.fromarray(img_array, mode='L')
        elif len(img_array.shape) == 3 and img_array.shape[2] == 3:  # RGB
            img = Image.fromarray(img_array, mode='RGB')
        else:
            img = Image.fromarray(img_array)
        
        # Apply transforms (ToTensor converts to [0, 1], then Normalize converts to [-1, 1])
        img = self.transform(img)
        
        return img


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
        
        # Convert to PIL
        if len(img_array.shape) == 2:
            img = Image.fromarray(img_array, mode='L')
        elif len(img_array.shape) == 3:
            img = Image.fromarray(img_array, mode='RGB')
        else:
            if img.mode != 'L' and img.mode != 'RGB':
                img = img.convert('L')
        
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
    
    Supports two modes:
    1. Pre-split: Separate directories for train and validation (default)
    2. Auto-split: Single directory that gets split automatically
    
    Args:
        config: Configuration object
        auto_split (bool): If True, automatically split data from single directories
        val_split (float): Fraction of data for validation when auto_split=True
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    
    if auto_split:
        # ===== AUTO-SPLIT MODE =====
        # Load all data from a single directory and split automatically
        print(f"\n{'='*60}")
        print("AUTO-SPLIT MODE: Splitting data automatically")
        print(f"Validation split: {val_split*100:.1f}%")
        print(f"{'='*60}\n")
        
        # Create full dataset with augmentation
        full_dataset_A = UnpairedMicroscopyDataset(
            dir_A=config.TRAIN_A_DIR,
            dir_B=config.TRAIN_B_DIR,
            img_size=config.IMG_SIZE,
            normalize_mean=config.NORMALIZE_MEAN,
            normalize_std=config.NORMALIZE_STD,
            use_random_flip=config.USE_RANDOM_FLIP,
            use_random_rotation=config.USE_RANDOM_ROTATION,
            clip_min=config.CLIP_MIN,
            clip_max=config.CLIP_MAX
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
        val_dataset_full = UnpairedMicroscopyDataset(
            dir_A=config.TRAIN_A_DIR,  # Same directory
            dir_B=config.TRAIN_B_DIR,  # Same directory
            img_size=config.IMG_SIZE,
            normalize_mean=config.NORMALIZE_MEAN,
            normalize_std=config.NORMALIZE_STD,
            use_random_flip=False,  # No augmentation for validation
            use_random_rotation=False,
            clip_min=config.CLIP_MIN,
            clip_max=config.CLIP_MAX
        )
        val_dataset = Subset(val_dataset_full, val_indices)
        
    else:
        # ===== PRE-SPLIT MODE =====
        # Use separate directories for train and validation
        print(f"\n{'='*60}")
        print("PRE-SPLIT MODE: Using separate train/val directories")
        print(f"{'='*60}\n")
        
        # Training dataset
        train_dataset = UnpairedMicroscopyDataset(
            dir_A=config.TRAIN_A_DIR,
            dir_B=config.TRAIN_B_DIR,
            img_size=config.IMG_SIZE,
            normalize_mean=config.NORMALIZE_MEAN,
            normalize_std=config.NORMALIZE_STD,
            use_random_flip=config.USE_RANDOM_FLIP,
            use_random_rotation=config.USE_RANDOM_ROTATION,
            clip_min=config.CLIP_MIN,
            clip_max=config.CLIP_MAX
        )
        
        # Validation dataset
        val_dataset = UnpairedMicroscopyDataset(
            dir_A=config.VAL_A_DIR,
            dir_B=config.VAL_B_DIR,
            img_size=config.IMG_SIZE,
            normalize_mean=config.NORMALIZE_MEAN,
            normalize_std=config.NORMALIZE_STD,
            use_random_flip=False,  # No augmentation for validation
            use_random_rotation=False,
            clip_min=config.CLIP_MIN,
            clip_max=config.CLIP_MAX
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
