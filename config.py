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
    DATA_ROOT = Path("/users/kir-fritzsche/aif490/devel/tissue_analysis/CARE/cycleCARE/data/")
    
    # Two modes supported:
    # 1. PRE-SPLIT: Use separate train/val directories (set AUTO_SPLIT_DATA=False)
    # 2. AUTO-SPLIT: Single directory, automatic split (set AUTO_SPLIT_DATA=True)
    
    TRAIN_A_DIR = DATA_ROOT /'node2_z85_z89_256'   # Clean surface microscopy images
    TRAIN_B_DIR = DATA_ROOT /'node2_z205_z209_256'   # Noisy microscopy images to restore
    VAL_A_DIR = DATA_ROOT / "valA"      # Validation clean images (only for pre-split)
    VAL_B_DIR = DATA_ROOT / "valB"      # Validation noisy images (only for pre-split)
    
    # Output directories
    OUTPUT_ROOT = Path("./outputs_node2_CARE_unet3_D2e5_256_")  # Change as needed
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
    ZSTACK_CONTEXT = 5           # Number of adjacent Z-planes to use as input (1 = single plane, 3/5/7 = multi-plane)
    ZSTACK_MODE    = True      # Enable Z-stack context mode (uses multi-plane input)
    
    # IMG_CHANNELS automatically set based on ZSTACK_CONTEXT if ZSTACK_MODE is True
    IMG_CHANNELS = ZSTACK_CONTEXT if ZSTACK_MODE else 1  # Input channels (1 for single plane, N for Z-context)
    
    UNET_DEPTH = 3            # Depth of U-Net (3 for 128x128: 128→64→32→16, fewer parameters, faster than depth=4)
    UNET_FILTERS = 64         # Number of filters in first layer (doubles with each layer)
    UNET_KERNEL_SIZE = 3      # Kernel size for convolutions
    USE_DROPOUT = False        # Use dropout in U-Net
    DROPOUT_RATE = 0.5        # Dropout rate
    USE_BATCH_NORM = True     # Use batch normalization
    
    # Discriminator (PatchGAN)
    DISC_FILTERS = 64         # Number of filters in first discriminator layer
    DISC_NUM_LAYERS = 3       # Number of layers in discriminator (4 to match generator depth)
    DISC_KERNEL_SIZE = 3      # Kernel size for discriminator
    
    # Discriminator input mode - focus on noise features
    # Provide list of representations to concatenate. Options: 'raw', 'highpass', 'fft'
    # Examples:
    #   ['raw'] - original images only
    #   ['highpass'] - high-pass filtered only (emphasizes spatial noise)
    #   ['fft'] - FFT magnitude spectrum only (emphasizes frequency characteristics)
    #   ['raw', 'highpass'] - both raw and high-pass (2x channels)
    #   ['highpass', 'fft'] - high-pass and FFT (2x channels)
    #   ['raw', 'highpass', 'fft'] - all three (3x channels)
    DISC_INPUT_MODE = ['raw']  # Current: high-pass filtering only
    DISC_HIGHPASS_SIGMA = 1.0      # Gaussian blur sigma for high-pass filter (1.0-2.0 recommended)
    
    # Discriminator input channels (automatically adjusted based on DISC_INPUT_MODE)
    DISC_CHANNELS = IMG_CHANNELS * len(DISC_INPUT_MODE)
    
    # ==================== Training Parameters (HPC Optimized) ====================
    # Optimization - larger batch size for GPU
    BATCH_SIZE = 16           # Optimized for 128x128 images with 5-plane Z-stack (HPC GPU can handle more)
    NUM_EPOCHS = 50
    LEARNING_RATE = 2e-4      # Generator learning rate
    LEARNING_RATE_D = 2e-5    # Discriminator learning rate (half of generator to slow down discriminator)
    BETA1 = 0.5               # Adam optimizer beta1
    BETA2 = 0.999             # Adam optimizer beta2
    
    # Loss weights
    LAMBDA_CYCLE = 15.0       # Weight for cycle-consistency loss (increased for stronger denoising)
    LAMBDA_IDENTITY = 0.5     # Weight for identity loss (reduced to focus more on cycle consistency)
    LAMBDA_ADV = 5.0          # Weight for adversarial loss (increased to prevent discriminator overpowering)

    CYCLE_LOSS_TYPE = 'combined'     # Recommended: 'combined' or 'ssim' for denoising
    IDENTITY_LOSS_TYPE = 'l1'        # Usually L1 is fine for identity
    
    # Weights for combined loss (only used if CYCLE_LOSS_TYPE='combined')
    # Balanced combination: SSIM for perceptual quality + L1 for pixel accuracy + Gradient for edges
    SSIM_WEIGHT = 0.7        # Weight for SSIM component (perceptual similarity, structure preservation)
    L1_WEIGHT = 0.2          # Weight for L1 component (pixel-level accuracy)
    GRAD_LOSS_WEIGHT = 0.1   # Weight for gradient loss (edge/morphology preservation)
                              # Total weights sum to 1.0 for balanced contribution
    
    # SSIM window sizes (should be smaller than your cell size)
    # For ~30 pixel cells: use window_size=7 (covers ~23% of cell)
    # Must be odd numbers for SSIM
    SSIM_WINDOW = 7           # 2D SSIM window size (optimized for 128×128 images)
    SSIM3D_WINDOW = 5         # 3D SSIM window size (default: 5 for shallow 5-plane stacks)      
    
    # Learning rate scheduling
    LR_DECAY_START_EPOCH = 25  # Epoch to start decaying learning rate
    LR_DECAY_END_EPOCH = 50    # Epoch to finish decaying to zero
    
    # ==================== Data Processing ====================
    IMG_SIZE = 256              # Input image size (128x128 TIF images)
    
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
    NUM_WORKERS = 16           # Increase workers for HPC (more CPU cores available)
    PIN_MEMORY = True         # Pin memory for faster GPU transfer on Linux
    PREFETCH_FACTOR = 2       # Prefetch batches for faster loading
    PERSISTENT_WORKERS = True # Keep workers alive between epochs
    
    SAVE_CHECKPOINT_FREQ = 5 # Save checkpoint every N epochs
    SAVE_SAMPLE_FREQ = 1      # Save sample images every N epochs (1 = every epoch)
    LOG_FREQ = 50             # Log training stats every N iterations
    
    # Resume training
    RESUME_TRAINING = False
    RESUME_CHECKPOINT = None #r'/users/kir-fritzsche/aif490/devel/tissue_analysis/CARE/cycleCARE/SLURM/outputs_node1_CARE_C10_I1_A10_raw_highpass_long_multiz/checkpoints/checkpoint_epoch_0050.pth' #'/users/kir-fritzsche/aif490/devel/tissue_analysis/CARE/cycleCARE/SLURM/outputs_node1_CARE_plane_C10_I1_A5/checkpoints/checkpoint_epoch_0030.pth'  # Path to checkpoint to resume from
    
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
