"""
Training script for 3D Cycle-CARE model - HPC Optimized.
Implements full 3D volumetric training with memory optimizations.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

# Import configurations and models
from config_3d import Config3D as Config
from models.cycle_care_3d import CycleCARE3D, create_discriminators_3d
from data.dataset_3d import Volume3DUnpairedDataset
from utils.losses import CycleCarelosses
from utils.helpers import save_checkpoint, load_checkpoint

# Try to import tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: TensorBoard not available")


def get_dataloaders_3d(config):
    """
    Create 3D training and validation dataloaders.
    
    Args:
        config: Configuration object
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    print(f"\n{'='*60}")
    print("Creating 3D Volume Dataloaders")
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
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None,
        persistent_workers=config.PERSISTENT_WORKERS and config.NUM_WORKERS > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None,
        persistent_workers=config.PERSISTENT_WORKERS and config.NUM_WORKERS > 0
    )
    
    print(f"Training volumes: {len(train_dataset)}")
    print(f"Validation volumes: {len(val_dataset)}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Effective batch size (with gradient accumulation): {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
    
    return train_loader, val_loader


def train_3d(config):
    """
    Main 3D training loop.
    
    Args:
        config: Configuration object
    """
    # Set random seed
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # Print configuration
    config.print_config()
    config.estimate_memory()
    
    # Create dataloaders
    train_loader, val_loader = get_dataloaders_3d(config)
    
    # Create model
    print("\nCreating 3D Cycle-CARE model...")
    model = CycleCARE3D(config).to(config.DEVICE)
    D_A, D_B = create_discriminators_3d(config)
    D_A = D_A.to(config.DEVICE)
    D_B = D_B.to(config.DEVICE)
    
    # Multi-GPU support
    if config.MULTI_GPU and config.DEVICE == 'cuda':
        print(f"\nUsing {len(config.GPU_IDS)} GPUs")
        model = nn.DataParallel(model, device_ids=config.GPU_IDS)
        D_A = nn.DataParallel(D_A, device_ids=config.GPU_IDS)
        D_B = nn.DataParallel(D_B, device_ids=config.GPU_IDS)
    
    # Create optimizers
    optimizer_G = torch.optim.Adam(
        list(model.parameters()),
        lr=config.LEARNING_RATE,
        betas=(config.BETA1, config.BETA2)
    )
    
    optimizer_D = torch.optim.Adam(
        list(D_A.parameters()) + list(D_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(config.BETA1, config.BETA2)
    )
    
    # Learning rate schedulers
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - config.LR_DECAY_START_EPOCH) / float(config.LR_DECAY_END_EPOCH - config.LR_DECAY_START_EPOCH + 1)
        return lr_l
    
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)
    
    # Loss manager
    loss_manager = CycleCarelosses(config, config.DEVICE)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config.MIXED_PRECISION and config.DEVICE == 'cuda' else None
    
    # TensorBoard writer
    writer = None
    if config.USE_TENSORBOARD and HAS_TENSORBOARD:
        writer = SummaryWriter(config.LOG_DIR / config.EXPERIMENT_NAME)
    
    # Training loop
    print(f"\n{'='*60}")
    print("Starting 3D Training")
    print(f"{'='*60}\n")
    
    start_epoch = 0
    global_step = 0
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        model.train()
        D_A.train()
        D_B.train()
        
        epoch_start = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        for i, batch in enumerate(pbar):
            real_A = batch['A'].to(config.DEVICE)
            real_B = batch['B'].to(config.DEVICE)
            
            # ==================== Train Generators ====================
            # Forward pass with mixed precision
            if scaler:
                with torch.cuda.amp.autocast():
                    loss_G, loss_dict_G = loss_manager.compute_generator_loss(
                        model, real_A, real_B, D_A, D_B
                    )
                    loss_G = loss_G / config.GRADIENT_ACCUMULATION_STEPS
                
                scaler.scale(loss_G).backward()
            else:
                loss_G, loss_dict_G = loss_manager.compute_generator_loss(
                    model, real_A, real_B, D_A, D_B
                )
                loss_G = loss_G / config.GRADIENT_ACCUMULATION_STEPS
                loss_G.backward()
            
            # Update generators (with gradient accumulation)
            if (i + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                if scaler:
                    scaler.step(optimizer_G)
                    scaler.update()
                else:
                    optimizer_G.step()
                optimizer_G.zero_grad()
            
            # ==================== Train Discriminators ====================
            if scaler:
                with torch.cuda.amp.autocast():
                    loss_D_A, loss_D_B, loss_dict_D = loss_manager.compute_discriminator_losses(
                        model, real_A, real_B, D_A, D_B
                    )
                    loss_D = (loss_D_A + loss_D_B) / config.GRADIENT_ACCUMULATION_STEPS
                
                scaler.scale(loss_D).backward()
            else:
                loss_D_A, loss_D_B, loss_dict_D = loss_manager.compute_discriminator_losses(
                    model, real_A, real_B, D_A, D_B
                )
                loss_D = (loss_D_A + loss_D_B) / config.GRADIENT_ACCUMULATION_STEPS
                loss_D.backward()
            
            # Update discriminators (with gradient accumulation)
            if (i + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                if scaler:
                    scaler.step(optimizer_D)
                    scaler.update()
                else:
                    optimizer_D.step()
                optimizer_D.zero_grad()
            
            # Update progress bar
            pbar.set_postfix({
                'G': f"{loss_dict_G['G_total']:.3f}",
                'D': f"{loss_dict_D['D_A_total'] + loss_dict_D['D_B_total']:.3f}"
            })
            
            # Log to tensorboard
            if writer and (i + 1) % config.LOG_FREQ == 0:
                for key, value in loss_dict_G.items():
                    writer.add_scalar(f'Generator/{key}', value, global_step)
                for key, value in loss_dict_D.items():
                    writer.add_scalar(f'Discriminator/{key}', value, global_step)
            
            global_step += 1
            
            # Empty cache periodically
            if config.DEVICE == 'cuda' and (i + 1) % config.EMPTY_CACHE_FREQ == 0:
                torch.cuda.empty_cache()
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")
        
        # Save checkpoint
        if (epoch + 1) % config.SAVE_CHECKPOINT_FREQ == 0:
            checkpoint_path = config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(
                checkpoint_path,
                epoch,
                model,
                optimizer_G,
                D_A,
                D_B,
                optimizer_D
            )
            print(f"Saved checkpoint: {checkpoint_path}")
    
    print("\n✓ Training completed!")
    
    if writer:
        writer.close()


if __name__ == "__main__":
    # Run training
    train_3d(Config)
