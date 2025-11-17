"""
Configuration file for Cycle-CARE model - HPC Optimized.
Optimized for Linux HPC environments with GPU acceleration.
"""

import os
from pathlib import Path
import torch


class Config:
    """Configuration class for Cycle-CARE training and inference on HPC."""
    
    # ==================== Paths ====================
    # Data directories - use absolute paths on HPC
    DATA_ROOT = Path("/Users/ewheeler/cycleCARE/data")
    
    # Two modes supported:
    # 1. PRE-SPLIT: Use separate train/val directories (set AUTO_SPLIT_DATA=False)
    # 2. AUTO-SPLIT: Single directory, automatic split (set AUTO_SPLIT_DATA=True)
    
    TRAIN_A_DIR = DATA_ROOT / "z100_z105_tiles/T0_C0"   # Clean surface microscopy images
    TRAIN_B_DIR = DATA_ROOT / "z220_z225_tiles/"   # Noisy microscopy images to restore
    VAL_A_DIR = DATA_ROOT / "valA"      # Validation clean images (only for pre-split)
    VAL_B_DIR = DATA_ROOT / "valB"      # Validation noisy images (only for pre-split)
    
    # Output directories
    OUTPUT_ROOT = Path("./outputs")
    CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
    LOG_DIR = OUTPUT_ROOT / "logs"
    SAMPLE_DIR = OUTPUT_ROOT / "samples"
    INFERENCE_DIR = OUTPUT_ROOT / "inference"
    
    # Create directories if they don't exist
    for dir_path in [CHECKPOINT_DIR, LOG_DIR, SAMPLE_DIR, INFERENCE_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # ==================== Model Architecture ====================
    # Generator (CARE-style U-Net)
    
    # Z-stack context settings
    ZSTACK_CONTEXT = 5        # Number of adjacent Z-planes to use as input (1 = single plane, 3/5/7 = multi-plane)
    ZSTACK_MODE = True       # Enable Z-stack context mode (uses multi-plane input)
    
    # IMG_CHANNELS automatically set based on ZSTACK_CONTEXT if ZSTACK_MODE is True
    IMG_CHANNELS = ZSTACK_CONTEXT if ZSTACK_MODE else 1  # Input channels (1 for single plane, N for Z-context)
    
    UNET_DEPTH = 3            # Depth of U-Net (2 for 128x128: 128→64→32, then back up)
    UNET_FILTERS = 64         # Number of filters in first layer (doubles with each layer)
    UNET_KERNEL_SIZE = 3      # Kernel size for convolutions
    USE_DROPOUT = True        # Use dropout in U-Net
    DROPOUT_RATE = 0.5        # Dropout rate
    USE_BATCH_NORM = True     # Use batch normalization
    
    # Discriminator (PatchGAN)
    DISC_FILTERS = 64         # Number of filters in first discriminator layer
    DISC_NUM_LAYERS = 3       # Number of layers in discriminator
    DISC_KERNEL_SIZE = 4      # Kernel size for discriminator
    
    # ==================== Training Parameters (HPC Optimized) ====================
    # Optimization - larger batch size for GPU
    BATCH_SIZE = 16           # Increase for HPC GPU (V100/A100 can handle more)
    NUM_EPOCHS = 200
    LEARNING_RATE = 2e-4
    BETA1 = 0.5               # Adam optimizer beta1
    BETA2 = 0.999             # Adam optimizer beta2
    
    # Loss weights
    LAMBDA_CYCLE = 10.0       # Weight for cycle-consistency loss
    LAMBDA_IDENTITY = 5.0     # Weight for identity loss (helps preserve color)
    LAMBDA_ADV = 1.0          # Weight for adversarial loss

    CYCLE_LOSS_TYPE = 'combined'     # Recommended: 'combined' or 'ssim' for denoising
    IDENTITY_LOSS_TYPE = 'l1'        # Usually L1 is fine for identity
    
    # Weights for combined loss (only used if CYCLE_LOSS_TYPE='combined')
    SSIM_WEIGHT = 0.84        # Weight for SSIM component (typically 0.84)
    L1_WEIGHT = 0.16      
    
    # Learning rate scheduling
    LR_DECAY_START_EPOCH = 100  # Epoch to start decaying learning rate
    LR_DECAY_END_EPOCH = 200    # Epoch to finish decaying to zero
    
    # ==================== Data Processing ====================
    IMG_SIZE = 128            # Input image size (128x128 TIF images)
    
    # Percentile-based normalization for fluorescence microscopy
    # Robust to outliers like hot pixels, saturated spots, and variable background
    USE_PERCENTILE_NORM = True  # Use percentile-based normalization instead of min-max
    PERCENTILE_LOW = 0.0        # Lower percentile for normalization (removes dark background)
    PERCENTILE_HIGH = 99.0      # Upper percentile for normalization (removes hot pixels)
    
    # Images are pre-normalized to [0, 1] range
    # To convert to [-1, 1] for model: x_normalized = (x - 0.5) / 0.5
    NORMALIZE_MEAN = 0.5      # Mean for normalization (converts [0,1] to [-1,1])
    NORMALIZE_STD = 0.5       # Std for normalization (converts [0,1] to [-1,1])
    
    # Clipping (set to None for pre-normalized images in [0, 1])
    CLIP_MIN = None           # Min value for clipping (None for no clipping)
    CLIP_MAX = None           # Max value for clipping (None for no clipping)
    
    # Data splitting
    AUTO_SPLIT_DATA = True    # If True, automatically split single directory into train/val
    VAL_SPLIT_RATIO = 0.2     # Validation split ratio when AUTO_SPLIT_DATA=True (0.0-1.0)
    
    # Data augmentation
    USE_RANDOM_FLIP = False   # Random horizontal flip
    USE_RANDOM_ROTATION = False  # Random rotation (can cause edge artifacts)
    
    # ==================== HPC Training Settings ====================
    NUM_WORKERS = 8           # Increase workers for HPC (more CPU cores available)
    PIN_MEMORY = True         # Pin memory for faster GPU transfer on Linux
    PREFETCH_FACTOR = 2       # Prefetch batches for faster loading
    PERSISTENT_WORKERS = True # Keep workers alive between epochs
    
    SAVE_CHECKPOINT_FREQ = 10 # Save checkpoint every N epochs
    SAVE_SAMPLE_FREQ = 1      # Save sample images every N epochs (1 = every epoch)
    LOG_FREQ = 50             # Log training stats every N iterations
    
    # Resume training
    RESUME_TRAINING = False
    RESUME_CHECKPOINT = None  # Path to checkpoint to resume from
    
    # ==================== Inference Settings ====================
    INFERENCE_INPUT_DIR = Path("./data/test")
    INFERENCE_OUTPUT_DIR = INFERENCE_DIR
    INFERENCE_CHECKPOINT = CHECKPOINT_DIR / "best_model.pth"
    INFERENCE_BATCH_SIZE = 4  # Can increase on HPC GPU
    SAVE_NOISY_INPUT = True   # Also save the noisy input for comparison
    
    # ==================== Device Settings (HPC Optimized) ====================
    # Auto-detect device: CUDA for NVIDIA GPUs (standard on HPC), CPU as fallback
    if torch.cuda.is_available():
        DEVICE = "cuda"
        # Enable cuDNN autotuner for better performance
        torch.backends.cudnn.benchmark = True
        # Deterministic operations for reproducibility (slightly slower)
        # torch.backends.cudnn.deterministic = True
    else:
        DEVICE = "cpu"
    
    MIXED_PRECISION = True    # Use mixed precision (FP16) for faster training on modern GPUs
    GRADIENT_ACCUMULATION_STEPS = 1  # Accumulate gradients for larger effective batch size
    
    # Multi-GPU settings (if available)
    MULTI_GPU = torch.cuda.device_count() > 1
    GPU_IDS = list(range(torch.cuda.device_count()))  # Use all available GPUs
    
    # ==================== Reproducibility ====================
    RANDOM_SEED = 42
    
    # ==================== Logging ====================
    EXPERIMENT_NAME = "cycle_care_hpc"
    VERBOSE = True            # Print detailed logs
    USE_TENSORBOARD = True    # Use TensorBoard for logging
    
    @classmethod
    def print_config(cls):
        """Print all configuration parameters."""
        print("\n" + "="*60)
        print("Cycle-CARE Configuration (HPC Optimized)")
        print("="*60)
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print("="*60)
        for attr in dir(cls):
            if not attr.startswith("_") and attr.isupper():
                print(f"{attr:30s}: {getattr(cls, attr)}")
        print("="*60 + "\n")


if __name__ == "__main__":
    Config.print_config()
