"""
Configuration file for 3D Cycle-CARE model - HPC Optimized.
Extends the base config for full 3D volumetric processing.
"""

import os
from pathlib import Path
import torch


class Config3D:
    """Configuration class for 3D Cycle-CARE training and inference on HPC."""
    
    # ==================== Model Mode ====================
    USE_3D_MODEL = True   # Enable full 3D volumetric processing
    
    # ==================== Paths ====================
    DATA_ROOT = Path("/Users/ewheeler/cycleCARE/data/")
    
    # 3D volume directories
    TRAIN_A_DIR = DATA_ROOT / "z100_z105_tiles"   # Clean volumes
    TRAIN_B_DIR = DATA_ROOT / "z220_z225_tiles"   # Noisy volumes
    VAL_A_DIR = DATA_ROOT / "valA"       # Validation clean volumes
    VAL_B_DIR = DATA_ROOT / "valB"       # Validation noisy volumes
    
    # Output directories
    OUTPUT_ROOT = Path("./outputs_3d")
    CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
    LOG_DIR = OUTPUT_ROOT / "logs"
    SAMPLE_DIR = OUTPUT_ROOT / "samples"
    INFERENCE_DIR = OUTPUT_ROOT / "inference"
    
    # Create directories if they don't exist
    for dir_path in [CHECKPOINT_DIR, LOG_DIR, SAMPLE_DIR, INFERENCE_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # ==================== 3D Model Architecture ====================
    # Generator (CARE-style 3D U-Net)
    IMG_CHANNELS = 1          # Input channels (1 for grayscale microscopy)
    VOLUME_DEPTH = 32         # Number of Z-planes in training volumes
    IMG_HEIGHT = 128          # Height of training volumes
    IMG_WIDTH = 128           # Width of training volumes
    
    UNET_DEPTH = 3            # Depth of 3D U-Net (3 for 32×128×128)
    UNET_FILTERS = 32         # Base filters (reduced from 64 due to memory)
    UNET_KERNEL_SIZE = 3      # Kernel size for 3D convolutions
    USE_DROPOUT = True        # Use dropout in 3D U-Net
    DROPOUT_RATE = 0.5        # Dropout rate
    USE_BATCH_NORM = True     # Use batch normalization
    
    # Discriminator (3D PatchGAN)
    DISC_FILTERS = 32         # Base filters (reduced from 64 due to memory)
    DISC_NUM_LAYERS = 3       # Number of layers in 3D discriminator
    DISC_KERNEL_SIZE = 4      # Kernel size for 3D discriminator
    
    # ==================== Training Parameters (3D Optimized) ====================
    # CRITICAL: 3D models require MUCH smaller batch sizes
    BATCH_SIZE = 1            # Likely max 1-2 for 3D on most GPUs
    NUM_EPOCHS = 200
    LEARNING_RATE = 2e-4
    BETA1 = 0.5
    BETA2 = 0.999
    
    # Loss weights
    LAMBDA_CYCLE = 10.0       # Weight for cycle-consistency loss
    LAMBDA_IDENTITY = 5.0     # Weight for identity loss
    LAMBDA_ADV = 1.0          # Weight for adversarial loss
    
    # Learning rate scheduling
    LR_DECAY_START_EPOCH = 100
    LR_DECAY_END_EPOCH = 200
    
    # ==================== 3D Data Processing ====================
    USE_RANDOM_CROP = False    # Use random 3D crops (more efficient than resizing)
    
    # Percentile-based normalization for fluorescence microscopy
    USE_PERCENTILE_NORM = True
    PERCENTILE_LOW = 0.0
    PERCENTILE_HIGH = 99.0
    
    # Normalization to [-1, 1]
    NORMALIZE_MEAN = 0.5
    NORMALIZE_STD = 0.5
    
    # Clipping
    CLIP_MIN = None
    CLIP_MAX = None
    
    # Data splitting
    AUTO_SPLIT_DATA = True   # Use pre-split directories for 3D
    VAL_SPLIT_RATIO = 0.2
    
    # 3D augmentation
    USE_RANDOM_FLIP = False    # Random flips in all 3 axes
    USE_RANDOM_ROTATION = False  # Random 90° rotations (expensive in 3D)
    
    # ==================== HPC Training Settings (3D Optimized) ====================
    # Reduced workers due to memory constraints
    NUM_WORKERS = 4           # Fewer workers for 3D (memory intensive)
    PIN_MEMORY = True
    PREFETCH_FACTOR = 1       # Reduced prefetch for 3D
    PERSISTENT_WORKERS = False  # May need to disable for memory
    
    # CRITICAL: Gradient accumulation to simulate larger batch sizes
    GRADIENT_ACCUMULATION_STEPS = 16  # Effective batch size = 16
    
    # Checkpointing
    SAVE_CHECKPOINT_FREQ = 10
    SAVE_SAMPLE_FREQ = 5      # Less frequent due to slower training
    LOG_FREQ = 10             # More frequent logging
    
    # Resume training
    RESUME_TRAINING = False
    RESUME_CHECKPOINT = None
    
    # ==================== 3D Inference Settings ====================
    INFERENCE_INPUT_DIR = Path("./data/3d/test")
    INFERENCE_OUTPUT_DIR = INFERENCE_DIR
    INFERENCE_CHECKPOINT = CHECKPOINT_DIR / "best_model.pth"
    INFERENCE_BATCH_SIZE = 1  # Always 1 for 3D inference
    SAVE_NOISY_INPUT = True
    
    # 3D tiling for large volumes
    USE_TILING = True         # Process large volumes in overlapping tiles
    TILE_DEPTH = 32
    TILE_HEIGHT = 128
    TILE_WIDTH = 128
    TILE_OVERLAP = 16         # Overlap in each dimension
    
    # ==================== Device Settings (3D Optimized) ====================
    if torch.cuda.is_available():
        DEVICE = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = "cpu"
    
    # ESSENTIAL for 3D training
    MIXED_PRECISION = True    # FP16 is critical for 3D memory usage
    GRADIENT_CHECKPOINTING = True  # Trade compute for memory
    
    # Multi-GPU (if available) - use DataParallel for 3D
    MULTI_GPU = torch.cuda.device_count() > 1
    GPU_IDS = list(range(torch.cuda.device_count()))
    
    # ==================== Memory Optimization ====================
    # Additional settings for 3D memory management
    EMPTY_CACHE_FREQ = 10     # Empty CUDA cache every N iterations
    
    # ==================== Reproducibility ====================
    RANDOM_SEED = 42
    
    # ==================== Logging ====================
    EXPERIMENT_NAME = "cycle_care_3d_hpc"
    VERBOSE = True
    USE_TENSORBOARD = True
    
    @classmethod
    def print_config(cls):
        """Print all configuration parameters."""
        print("\n" + "="*60)
        print("Cycle-CARE 3D Configuration (HPC Optimized)")
        print("="*60)
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                # Get GPU memory
                total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"    Memory: {total_mem:.1f} GB")
        print("="*60)
        print("3D Volume Settings:")
        print(f"  Volume size: {cls.VOLUME_DEPTH}×{cls.IMG_HEIGHT}×{cls.IMG_WIDTH}")
        print(f"  Batch size: {cls.BATCH_SIZE}")
        print(f"  Effective batch (with accumulation): {cls.BATCH_SIZE * cls.GRADIENT_ACCUMULATION_STEPS}")
        print(f"  Random crop: {cls.USE_RANDOM_CROP}")
        print(f"  Mixed precision: {cls.MIXED_PRECISION}")
        print(f"  Gradient checkpointing: {cls.GRADIENT_CHECKPOINTING}")
        print("="*60)
        for attr in dir(cls):
            if not attr.startswith("_") and attr.isupper():
                value = getattr(cls, attr)
                if not callable(value):
                    print(f"{attr:30s}: {value}")
        print("="*60 + "\n")
    
    @classmethod
    def estimate_memory(cls):
        """Estimate GPU memory requirements."""
        print("\n" + "="*60)
        print("Estimated GPU Memory Requirements (3D)")
        print("="*60)
        
        # Rough estimates based on typical usage
        # Generator parameters
        gen_params = cls.UNET_FILTERS * (2 ** cls.UNET_DEPTH) * cls.UNET_KERNEL_SIZE ** 3
        gen_params_mb = gen_params * 4 / 1024**2  # 4 bytes per float32
        
        # Activation memory (main memory hog in 3D)
        activation_size = (cls.BATCH_SIZE * cls.UNET_FILTERS * 4 * 
                          cls.VOLUME_DEPTH * cls.IMG_HEIGHT * cls.IMG_WIDTH)
        activation_mb = activation_size * 4 / 1024**2
        
        # Total rough estimate
        total_mb = gen_params_mb * 4 + activation_mb * 2  # 4 models (2 gen + 2 disc), 2x for gradients
        
        print(f"Generator parameters: ~{gen_params_mb:.0f} MB")
        print(f"Activation memory: ~{activation_mb:.0f} MB per forward pass")
        print(f"Total estimated: ~{total_mb:.0f} MB ({total_mb/1024:.1f} GB)")
        print(f"\nRecommended GPU: {total_mb/1024:.0f}+ GB VRAM")
        if total_mb > 8000:
            print("WARNING: May require 16GB+ GPU (A100, V100, RTX 3090+)")
        print("="*60 + "\n")


if __name__ == "__main__":
    Config3D.print_config()
    Config3D.estimate_memory()
