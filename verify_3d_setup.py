#!/usr/bin/env python3
"""
Verify 3D CycleCARE setup for 64³ depth restoration.

Checks:
- Configuration parameters
- GPU availability and memory
- Data directory structure
- Sample data loading
- Model architecture initialization

Run this before starting training to catch issues early!

Usage:
    python verify_3d_setup.py
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config_3d import Config3D
from models.cycle_care_3d import CycleCARE3D
from data.dataset_3d import Volume3DUnpairedDataset


def check_gpu():
    """Check GPU availability and properties."""
    print("\n" + "="*60)
    print("GPU CHECK")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        print("   3D training requires GPU")
        return False
    
    print(f"✓ CUDA available")
    print(f"✓ GPU count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\n  GPU {i}: {props.name}")
        print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"    Compute capability: {props.major}.{props.minor}")
        
        # Check if enough memory for 64³
        if props.total_memory / 1024**3 < 12:
            print(f"    ⚠️  Warning: May need 16+ GB for 64³ volumes")
    
    return True


def check_config():
    """Check configuration parameters."""
    print("\n" + "="*60)
    print("CONFIGURATION CHECK")
    print("="*60)
    
    config = Config3D
    
    # Volume dimensions
    print(f"\nVolume dimensions: {config.VOLUME_DEPTH}×{config.IMG_HEIGHT}×{config.IMG_WIDTH}")
    if config.VOLUME_DEPTH != 64 or config.IMG_HEIGHT != 64 or config.IMG_WIDTH != 64:
        print(f"  ⚠️  Expected 64³, got {config.VOLUME_DEPTH}×{config.IMG_HEIGHT}×{config.IMG_WIDTH}")
    else:
        print("  ✓ Correct 64³ dimensions")
    
    # Network architecture
    print(f"\nNetwork architecture:")
    print(f"  U-Net depth: {config.UNET_DEPTH}")
    print(f"  U-Net filters: {config.UNET_FILTERS}")
    print(f"  Discriminator layers: {config.DISC_NUM_LAYERS}")
    
    if config.UNET_DEPTH == 3:
        print("  ✓ Good depth for 64³ (enables 64→32→16→8)")
    else:
        print(f"  ⚠️  Recommended depth=3 for 64³, got {config.UNET_DEPTH}")
    
    # Training parameters
    print(f"\nTraining parameters:")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Gradient accumulation: {config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective batch size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Mixed precision: {config.MIXED_PRECISION}")
    
    if config.MIXED_PRECISION:
        print("  ✓ Mixed precision enabled (essential for 64³)")
    else:
        print("  ⚠️  Mixed precision disabled - may run out of memory!")
    
    # Loss configuration
    print(f"\nLoss configuration:")
    print(f"  Cycle loss type: {config.CYCLE_LOSS_TYPE}")
    print(f"  Lambda cycle: {config.LAMBDA_CYCLE}")
    print(f"  Lambda identity: {config.LAMBDA_IDENTITY}")
    print(f"  Lambda adversarial: {config.LAMBDA_ADV}")
    
    if config.CYCLE_LOSS_TYPE == 'combined':
        print(f"  SSIM weight: {config.SSIM_WEIGHT}")
        print(f"  L1 weight: {config.L1_WEIGHT}")
        print(f"  Gradient weight: {config.GRAD_LOSS_WEIGHT}")
        print("  ✓ Combined loss good for depth restoration")
    
    return True


def check_data_dirs():
    """Check data directory structure."""
    print("\n" + "="*60)
    print("DATA DIRECTORY CHECK")
    print("="*60)
    
    config = Config3D
    
    checks = {
        "Domain A (clean)": config.TRAIN_A_DIR,
        "Domain B (degraded)": config.TRAIN_B_DIR,
    }
    
    all_good = True
    
    for name, path in checks.items():
        print(f"\n{name}: {path}")
        
        if not path.exists():
            print(f"  ❌ Directory does not exist!")
            print(f"     Create it with: mkdir -p {path}")
            all_good = False
            continue
        
        # Count TIFF files
        tif_files = list(path.glob("*.tif")) + list(path.glob("*.tiff"))
        print(f"  ✓ Directory exists")
        print(f"  TIFF files found: {len(tif_files)}")
        
        if len(tif_files) == 0:
            print(f"  ⚠️  No TIFF files found!")
            print(f"     Use extract_64cubes.py to create training data")
            all_good = False
        elif len(tif_files) < 10:
            print(f"  ⚠️  Only {len(tif_files)} files - recommend 20+ for good training")
        else:
            print(f"  ✓ Good number of training volumes")
        
        # Try loading one file
        if len(tif_files) > 0:
            try:
                import tifffile
                sample = tifffile.imread(tif_files[0])
                print(f"  Sample shape: {sample.shape}")
                
                if sample.shape != (64, 64, 64):
                    print(f"  ⚠️  Expected (64,64,64), got {sample.shape}")
                    print(f"      Will be cropped/resized during training")
                else:
                    print(f"  ✓ Correct cube dimensions")
                    
            except Exception as e:
                print(f"  ⚠️  Could not load sample: {e}")
    
    return all_good


def check_model_init():
    """Check model initialization."""
    print("\n" + "="*60)
    print("MODEL INITIALIZATION CHECK")
    print("="*60)
    
    config = Config3D
    
    try:
        print("\nCreating model...")
        model = CycleCARE3D(
            img_channels=config.IMG_CHANNELS,
            unet_depth=config.UNET_DEPTH,
            unet_filters=config.UNET_FILTERS,
            unet_kernel_size=config.UNET_KERNEL_SIZE,
            disc_filters=config.DISC_FILTERS,
            disc_num_layers=config.DISC_NUM_LAYERS,
            disc_kernel_size=config.DISC_KERNEL_SIZE,
            use_batch_norm=config.USE_BATCH_NORM,
            use_dropout=config.USE_DROPOUT,
            dropout_rate=config.DROPOUT_RATE
        )
        
        print("✓ Model created successfully")
        
        # Count parameters
        g_ab_params = sum(p.numel() for p in model.G_AB.parameters())
        g_ba_params = sum(p.numel() for p in model.G_BA.parameters())
        d_a_params = sum(p.numel() for p in model.D_A.parameters())
        d_b_params = sum(p.numel() for p in model.D_B.parameters())
        total_params = g_ab_params + g_ba_params + d_a_params + d_b_params
        
        print(f"\nParameter counts:")
        print(f"  G_AB: {g_ab_params:,}")
        print(f"  G_BA: {g_ba_params:,}")
        print(f"  D_A:  {d_a_params:,}")
        print(f"  D_B:  {d_b_params:,}")
        print(f"  Total: {total_params:,}")
        print(f"  Memory: ~{total_params * 4 / 1024**2:.1f} MB (params only)")
        
        # Test forward pass on CPU
        print("\nTesting forward pass (CPU)...")
        dummy_A = torch.randn(1, 1, 64, 64, 64)
        dummy_B = torch.randn(1, 1, 64, 64, 64)
        
        with torch.no_grad():
            outputs = model(dummy_A, dummy_B, mode='full')
        
        print("✓ Forward pass successful")
        print(f"  Output keys: {list(outputs.keys())}")
        print(f"  fake_A shape: {outputs['fake_A'].shape}")
        
        # Test GPU if available
        if torch.cuda.is_available():
            print("\nTesting GPU transfer...")
            model_gpu = model.cuda()
            dummy_A_gpu = dummy_A.cuda()
            dummy_B_gpu = dummy_B.cuda()
            
            with torch.no_grad():
                outputs_gpu = model_gpu(dummy_A_gpu, dummy_B_gpu, mode='full')
            
            print("✓ GPU forward pass successful")
            
            # Clean up
            del model_gpu, dummy_A_gpu, dummy_B_gpu, outputs_gpu
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_data_loading():
    """Check data loading."""
    print("\n" + "="*60)
    print("DATA LOADING CHECK")
    print("="*60)
    
    config = Config3D
    
    # Check if data directories exist and have files
    if not config.TRAIN_A_DIR.exists() or not config.TRAIN_B_DIR.exists():
        print("❌ Data directories not set up - skipping data loading test")
        return False
    
    tif_a = list(config.TRAIN_A_DIR.glob("*.tif")) + list(config.TRAIN_A_DIR.glob("*.tiff"))
    tif_b = list(config.TRAIN_B_DIR.glob("*.tif")) + list(config.TRAIN_B_DIR.glob("*.tiff"))
    
    if len(tif_a) == 0 or len(tif_b) == 0:
        print("❌ No TIFF files found - skipping data loading test")
        return False
    
    try:
        print("\nCreating dataset...")
        dataset = Volume3DUnpairedDataset(
            dir_A=config.TRAIN_A_DIR,
            dir_B=config.TRAIN_B_DIR,
            volume_depth=config.VOLUME_DEPTH,
            img_height=config.IMG_HEIGHT,
            img_width=config.IMG_WIDTH,
            use_random_crop=config.USE_RANDOM_CROP,
            use_random_flip=config.USE_RANDOM_FLIP,
            use_percentile_norm=config.USE_PERCENTILE_NORM,
            percentile_low=config.PERCENTILE_LOW,
            percentile_high=config.PERCENTILE_HIGH
        )
        
        print(f"✓ Dataset created")
        print(f"  Length: {len(dataset)} volume pairs")
        
        # Load one sample
        print("\nLoading sample...")
        sample = dataset[0]
        
        print(f"✓ Sample loaded")
        print(f"  Domain A shape: {sample['A'].shape}")
        print(f"  Domain B shape: {sample['B'].shape}")
        print(f"  Domain A range: [{sample['A'].min():.3f}, {sample['A'].max():.3f}]")
        print(f"  Domain B range: [{sample['B'].min():.3f}, {sample['B'].max():.3f}]")
        
        expected_shape = (1, config.VOLUME_DEPTH, config.IMG_HEIGHT, config.IMG_WIDTH)
        if sample['A'].shape == expected_shape:
            print(f"  ✓ Correct shape")
        else:
            print(f"  ⚠️  Expected {expected_shape}, got {sample['A'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks."""
    print("\n" + "="*60)
    print("3D CYCLECARE SETUP VERIFICATION (64³ DEPTH RESTORATION)")
    print("="*60)
    
    checks = [
        ("GPU", check_gpu),
        ("Configuration", check_config),
        ("Data Directories", check_data_dirs),
        ("Model Initialization", check_model_init),
        ("Data Loading", check_data_loading),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ {name} check failed with exception: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name in results:
        status = "✓ PASS" if results[name] else "❌ FAIL"
        print(f"{name:25s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Ready to train!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Review configuration in config_3d.py")
        print("  2. Start training: python train_3d.py")
        print("  3. Monitor with: tensorboard --logdir outputs_3d_64cubes_depth_restoration/logs")
    else:
        print("❌ SOME CHECKS FAILED - Fix issues before training")
        print("="*60)
        print("\nRecommended actions:")
        if not results.get("GPU"):
            print("  - Ensure CUDA is installed and GPU is available")
        if not results.get("Data Directories"):
            print("  - Create data directories and add training volumes")
            print("  - Use extract_64cubes.py to prepare data")
        if not results.get("Model Initialization"):
            print("  - Check that all dependencies are installed")
            print("  - Verify PyTorch version supports your GPU")
    
    print("\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
