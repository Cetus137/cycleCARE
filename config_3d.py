"""
Configuration file for 3D Cycle-CARE model - HPC Optimized.;
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
    DATA_ROOT = Path("/users/kir-fritzsche/aif490/devel/tissue_analysis/CARE/cycleCARE/data_3D_node2")
    
    # 3D volume directories
    TRAIN_A_DIR = DATA_ROOT / "surface_subset"     # Clean surface volumes (Z=0-64)
    TRAIN_B_DIR = DATA_ROOT / "deep_subset"        # Degraded deep volumes (Z=64-128)
    VAL_A_DIR = DATA_ROOT / "valA"                        # Validation clean surface volumes
    VAL_B_DIR = DATA_ROOT / "valB"                        # Validation degraded deep volumes
    
    # Output directories
    OUTPUT_ROOT = Path("./outputs_3d_subset_long_A100_node2_Identity_subset")
    CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
    LOG_DIR = OUTPUT_ROOT / "logs"
    SAMPLE_DIR = OUTPUT_ROOT / "samples"
    INFERENCE_DIR = OUTPUT_ROOT / "inference"
    
    # Create directories if they don't exist
    for dir_path in [CHECKPOINT_DIR, LOG_DIR, SAMPLE_DIR, INFERENCE_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # ==================== 3D Model Architecture (64×64×64 Volumes) ====================
    # Generator (CARE-style 3D U-Net)
    IMG_CHANNELS = 1          # Input channels (1 for grayscale microscopy)
    VOLUME_DEPTH = 64         # Number of Z-planes (MUST match your data!)
    IMG_HEIGHT = 128          # Height of volumes (64×128×128 tiles)
    IMG_WIDTH = 128           # Width of volumes (64×128×128 tiles)
    
    # Network architecture - deeper network enabled by larger dimensions
    UNET_DEPTH = 3            # Depth of 3D U-Net (64×128×128 → 32×64×64 → 16×32×32 → 8×16×16)
    UNET_FILTERS = 24         # Base filters (reduced to 24 for memory efficiency with 64×128×128 volumes)
    UNET_KERNEL_SIZE = 3      # Kernel size for 3D convolutions
    USE_DROPOUT = True        # Use dropout in 3D U-Net (important for depth generalization)
    DROPOUT_RATE = 0.5        # Dropout rate
    USE_BATCH_NORM = True     # Use batch normalization
    
    # Discriminator (3D PatchGAN)
    DISC_FILTERS = 24         # Base filters (matched to generator)
    DISC_NUM_LAYERS = 3       # Number of layers in 3D discriminator (increased for 64³)
    DISC_KERNEL_SIZE = 4      # Kernel size for 3D discriminator
    
    # ==================== Training Parameters (64³ Optimized) ====================
    # CRITICAL: 64³ volumes are ~3× larger than 5×128×128, requires batch_size=1
    BATCH_SIZE = 1            # Maximum for 64³ volumes on most GPUs
    NUM_EPOCHS = 50          # More epochs for depth degradation learning
    LEARNING_RATE = 2e-4      # Generator learning rate
    LEARNING_RATE_D = 5e-5    # Discriminator LR — lower than G to prevent D from winning too fast
    BETA1 = 0.5
    BETA2 = 0.999
    
    # Generator / Discriminator update ratio
    # 1 = standard 1:1, 2 = two G updates per D update (helps when D dominates)
    GENERATOR_UPDATES_PER_ITER = 1

    # Loss weights - optimized for depth-dependent restoration
    LAMBDA_CYCLE = 20.0       # High cycle consistency to preserve depth coherence
    LAMBDA_IDENTITY = 1.0     # 0 = skip identity forward passes entirely (saves 2/6 gen passes)
    LAMBDA_ADV = 5.0          # Balanced adversarial weight
    
    # Loss types - optimized for depth restoration
    # - 'l1': Mean Absolute Error (pixel-wise, sharp but can be blocky)
    # - 'l2': Mean Squared Error (pixel-wise, sensitive to outliers, can blur)
    # - 'ssim': Structural Similarity (perceptual, preserves texture/structure)
    # - 'combined': 3D SSIM + L1 + Gradient (best for depth-dependent features)
    CYCLE_LOSS_TYPE = 'l1'     # 'l1' is much faster than 'combined' (no 3D SSIM/gradient cost)
    IDENTITY_LOSS_TYPE = 'l1'        # L1 for identity
    
    # Weights for combined loss - balanced for depth coherence
    SSIM_WEIGHT = 0.2        # High SSIM to preserve 3D structural similarity across depth
    L1_WEIGHT = 0.7          # Moderate L1 for pixel accuracy
    GRAD_LOSS_WEIGHT = 0.1   # Gradient loss to preserve edges/features across depth
    
    # SSIM window sizes for 3D
    SSIM_WINDOW = 7          # 2D SSIM window (if used)
    SSIM3D_WINDOW = 7        # 3D SSIM window size (smaller than default 11 for 64³)
    
    # Learning rate scheduling
    LR_DECAY_START_EPOCH = 25
    LR_DECAY_END_EPOCH = 50
    
    # ==================== 3D Data Processing (64×128×128 Volumes) ====================
    USE_RANDOM_CROP = False    # Volumes are already preprocessed to 64×128×128, no cropping needed
    
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
    AUTO_SPLIT_DATA = True   # Automatically split training data for validation
    VAL_SPLIT_RATIO = 0.2
    
    # 3D augmentation - important for generalization across depth patterns
    USE_RANDOM_FLIP = True     # Random flips in all 3 axes (helps learn depth invariance)
    USE_RANDOM_ROTATION = False  # Random 90° rotations (expensive, disable for speed)
    
    # ==================== HPC Training Settings (64×128×128 Optimized) ====================
    # Reduced workers due to memory constraints of 64×128×128 volumes
    NUM_WORKERS = 2           # Further reduced for large 3D volumes (very memory intensive)
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2       # Minimal prefetch for large 3D volumes
    PERSISTENT_WORKERS = False  # Disabled for memory with large 3D volumes
    
    # CRITICAL: Gradient accumulation to simulate larger batch sizes
    GRADIENT_ACCUMULATION_STEPS = 32  # Effective batch size = 32 (important for stable depth learning)
    
    # Checkpointing
    SAVE_CHECKPOINT_FREQ = 10
    SAVE_SAMPLE_FREQ = 1      # Less frequent due to slower training
    LOG_FREQ = 1           # More frequent logging
    
    # Resume training
    RESUME_TRAINING = False
    RESUME_CHECKPOINT = None
    
    # ==================== 3D Inference Settings ====================
    INFERENCE_INPUT_DIR = Path("./data/3d/deep_tissue_test")
    INFERENCE_OUTPUT_DIR = INFERENCE_DIR
    INFERENCE_CHECKPOINT = CHECKPOINT_DIR / "best_model.pth"
    INFERENCE_BATCH_SIZE = 1  # Always 1 for 3D inference
    SAVE_NOISY_INPUT = True
    
    # 3D tiling for inference on large volumes (process in 64×128×128 tiles)
    USE_TILING = True         # Process large volumes in overlapping tiles during inference only
    TILE_DEPTH = 64           # Match training volume depth
    TILE_HEIGHT = 128         # Match training dimensions
    TILE_WIDTH = 128          # 64×128×128 tiles
    TILE_OVERLAP = 16         # Overlap to avoid edge artifacts
    
    # ==================== Device Settings (3D Optimized) ====================
    if torch.cuda.is_available():
        DEVICE = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = "cpu"
    
    # ESSENTIAL for 3D training
    MIXED_PRECISION = True    # FP16 is critical for 3D memory usage
    GRADIENT_CHECKPOINTING = False  # NOTE: not yet implemented in model/train — has no effect currently

    # torch.compile (PyTorch >= 2.0): fuses 3D conv ops for 10-30% throughput gain.
    # First epoch will be slow (compilation). Disable if using gradient checkpointing or multi-GPU.
    USE_TORCH_COMPILE = False
    
    # Multi-GPU (if available) - use DataParallel for 3D
    MULTI_GPU = torch.cuda.device_count() > 1
    GPU_IDS = list(range(torch.cuda.device_count()))
    
    # ==================== Memory Optimization ====================
    # Additional settings for 3D memory management
    EMPTY_CACHE_FREQ = 10     # Empty CUDA cache every N iterations
    
    # ==================== Reproducibility ====================
    RANDOM_SEED = 42
    
    # ==================== Logging ====================
    EXPERIMENT_NAME = "cycle_care_3d_64cubes_depth_restoration"
    VERBOSE = True
    USE_TENSORBOARD = True
    
    @classmethod
    def print_config(cls):
        """Print all configuration parameters."""
        print("\n" + "="*60)
        print("Cycle-CARE 3D Configuration (64³ Depth Restoration)")
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
