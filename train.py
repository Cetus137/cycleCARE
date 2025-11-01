"""
Training script for Cycle-CARE model - HPC Optimized.
Includes mixed precision training, gradient accumulation, and multi-GPU support.
"""

import os
import time
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from config import Config
from models import CycleCARE
from data import get_dataloaders
from utils.losses import CycleCarelosses
from utils.helpers import (
    save_checkpoint, load_checkpoint, save_comparison_grid,
    get_learning_rate, update_learning_rate, set_requires_grad,
    ImagePool, AverageMeter, print_training_info
)


def setup_training(config):
    """
    Setup model, optimizers, and training utilities for HPC.
    
    Args:
        config: Configuration object
    
    Returns:
        tuple: (model, optimizers, image_pools, loss_manager, device, scaler)
    """
    # Set device
    device = torch.device(config.DEVICE)
    print(f"\nUsing device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
    
    # Create model
    print("\nCreating Cycle-CARE model...")
    model = CycleCARE(
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
    
    # Multi-GPU support
    if config.MULTI_GPU and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model, device_ids=config.GPU_IDS)
    
    model = model.to(device)
    
    # Print model summary (handle DataParallel wrapper)
    if isinstance(model, nn.DataParallel):
        model.module.print_model_summary()
    else:
        model.print_model_summary()
    
    # Create optimizers
    print("Creating optimizers...")
    # Handle DataParallel wrapper for accessing model components
    model_ref = model.module if isinstance(model, nn.DataParallel) else model
    
    optimizer_G = torch.optim.Adam(
        list(model_ref.G_AB.parameters()) + list(model_ref.G_BA.parameters()),
        lr=config.LEARNING_RATE,
        betas=(config.BETA1, config.BETA2)
    )
    
    optimizer_D_A = torch.optim.Adam(
        model_ref.D_A.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.BETA1, config.BETA2)
    )
    
    optimizer_D_B = torch.optim.Adam(
        model_ref.D_B.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.BETA1, config.BETA2)
    )
    
    optimizers = {
        'G': optimizer_G,
        'D_A': optimizer_D_A,
        'D_B': optimizer_D_B
    }
    
    # Create GradScaler for mixed precision training
    scaler = GradScaler() if config.MIXED_PRECISION else None
    if config.MIXED_PRECISION:
        print("Mixed precision training enabled (FP16)")
    
    # Create image pools to store generated images
    fake_A_pool = ImagePool(pool_size=50)
    fake_B_pool = ImagePool(pool_size=50)
    image_pools = {'A': fake_A_pool, 'B': fake_B_pool}
    
    # Create loss manager
    loss_manager = CycleCarelosses(config, device)
    
    return model, optimizers, image_pools, loss_manager, device, scaler


def train_epoch(model, train_loader, optimizers, image_pools, loss_manager, 
                epoch, config, scaler=None, writer=None):
    """
    Train for one epoch with HPC optimizations.
    
    Args:
        model: Cycle-CARE model
        train_loader: Training data loader
        optimizers: Dictionary of optimizers
        image_pools: Dictionary of image pools
        loss_manager: Loss manager
        epoch (int): Current epoch
        config: Configuration object
        scaler: GradScaler for mixed precision (optional)
        writer: TensorBoard writer (optional)
    
    Returns:
        dict: Average losses for the epoch
    """
    model.train()
    
    # Handle DataParallel wrapper
    model_ref = model.module if isinstance(model, nn.DataParallel) else model
    
    # Average meters for tracking losses
    meters = {
        'G_total': AverageMeter(),
        'G_AB': AverageMeter(),
        'G_BA': AverageMeter(),
        'cycle_A': AverageMeter(),
        'cycle_B': AverageMeter(),
        'identity_A': AverageMeter(),
        'identity_B': AverageMeter(),
        'D_A': AverageMeter(),
        'D_B': AverageMeter()
    }
    
    num_iters = len(train_loader)
    start_time = time.time()
    
    for i, batch in enumerate(train_loader):
        iter_start_time = time.time()
        
        # Get data
        real_A = batch['A'].to(config.DEVICE, non_blocking=True)
        real_B = batch['B'].to(config.DEVICE, non_blocking=True)
        
        # ===================== Train Generators =====================
        set_requires_grad([model_ref.D_A, model_ref.D_B], False)  # Freeze discriminators
        optimizers['G'].zero_grad()
        
        # Mixed precision forward pass
        if config.MIXED_PRECISION:
            with autocast():
                loss_G, loss_dict_G = loss_manager.compute_generator_loss(
                    model, real_A, real_B, model_ref.D_A, model_ref.D_B
                )
            scaler.scale(loss_G).backward()
            scaler.step(optimizers['G'])
            scaler.update()
        else:
            loss_G, loss_dict_G = loss_manager.compute_generator_loss(
                model, real_A, real_B, model_ref.D_A, model_ref.D_B
            )
            loss_G.backward()
            optimizers['G'].step()
        
        # ===================== Train Discriminator A =====================
        set_requires_grad(model_ref.D_A, True)  # Unfreeze discriminator A
        optimizers['D_A'].zero_grad()
        
        # Generate fake A and get from pool
        with torch.no_grad():
            if config.MIXED_PRECISION:
                with autocast():
                    fake_A = model_ref.G_BA(real_B)
            else:
                fake_A = model_ref.G_BA(real_B)
        fake_A_pooled = image_pools['A'].query(fake_A)
        
        # Compute discriminator A loss
        if config.MIXED_PRECISION:
            with autocast():
                loss_D_A, _, loss_dict_D = loss_manager.compute_discriminator_losses(
                    model, real_A, real_B, model_ref.D_A, model_ref.D_B
                )
            scaler.scale(loss_D_A).backward()
            scaler.step(optimizers['D_A'])
            scaler.update()
        else:
            loss_D_A, _, loss_dict_D = loss_manager.compute_discriminator_losses(
                model, real_A, real_B, model_ref.D_A, model_ref.D_B
            )
            loss_D_A.backward()
            optimizers['D_A'].step()
        
        # ===================== Train Discriminator B =====================
        set_requires_grad(model_ref.D_B, True)  # Unfreeze discriminator B
        optimizers['D_B'].zero_grad()
        
        # Generate fake B and get from pool
        with torch.no_grad():
            if config.MIXED_PRECISION:
                with autocast():
                    fake_B = model_ref.G_AB(real_A)
            else:
                fake_B = model_ref.G_AB(real_A)
        fake_B_pooled = image_pools['B'].query(fake_B)
        
        # Compute discriminator B loss
        if config.MIXED_PRECISION:
            with autocast():
                _, loss_D_B, _ = loss_manager.compute_discriminator_losses(
                    model, real_A, real_B, model_ref.D_A, model_ref.D_B
                )
            scaler.scale(loss_D_B).backward()
            scaler.step(optimizers['D_B'])
            scaler.update()
        else:
            _, loss_D_B, _ = loss_manager.compute_discriminator_losses(
                model, real_A, real_B, model_ref.D_A, model_ref.D_B
            )
            loss_D_B.backward()
            optimizers['D_B'].step()
        
        # ===================== Update Meters =====================
        meters['G_total'].update(loss_dict_G['G_total'])
        meters['G_AB'].update(loss_dict_G['G_AB'])
        meters['G_BA'].update(loss_dict_G['G_BA'])
        meters['cycle_A'].update(loss_dict_G['cycle_A'])
        meters['cycle_B'].update(loss_dict_G['cycle_B'])
        meters['identity_A'].update(loss_dict_G['identity_A'])
        meters['identity_B'].update(loss_dict_G['identity_B'])
        meters['D_A'].update(loss_dict_D['D_A_total'])
        meters['D_B'].update(loss_dict_D['D_B_total'])
        
        # ===================== Logging =====================
        if (i + 1) % config.LOG_FREQ == 0 or (i + 1) == num_iters:
            iter_time = time.time() - iter_start_time
            losses = {key: meter.avg for key, meter in meters.items()}
            
            if config.VERBOSE:
                print_training_info(epoch, config.NUM_EPOCHS, i + 1, num_iters, losses, iter_time)
            
            # TensorBoard logging
            if writer is not None:
                global_step = (epoch - 1) * num_iters + i
                for key, value in losses.items():
                    writer.add_scalar(f'Train/{key}', value, global_step)
                writer.add_scalar('Train/learning_rate', get_learning_rate(optimizers['G']), global_step)
    
    epoch_time = time.time() - start_time
    print(f"\nEpoch {epoch} completed in {epoch_time:.2f}s ({epoch_time/60:.1f} min)")
    
    # Return average losses
    return {key: meter.avg for key, meter in meters.items()}


def validate(model, val_loader, loss_manager, epoch, config, writer=None):
    """
    Validate the model.
    
    Args:
        model: Cycle-CARE model
        val_loader: Validation data loader
        loss_manager: Loss manager
        epoch (int): Current epoch
        config: Configuration object
        writer: TensorBoard writer (optional)
    
    Returns:
        dict: Average validation losses
    """
    model.eval()
    
    meters = {
        'cycle_A': AverageMeter(),
        'cycle_B': AverageMeter()
    }
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            real_A = batch['A'].to(config.DEVICE, non_blocking=True)
            real_B = batch['B'].to(config.DEVICE, non_blocking=True)
            
            # Forward pass (with mixed precision if enabled)
            if config.MIXED_PRECISION:
                with autocast():
                    outputs = model(real_A=real_A, real_B=real_B, mode='full')
                    cycle_loss_A = loss_manager.cycle_loss(outputs['reconstructed_A'], real_A)
                    cycle_loss_B = loss_manager.cycle_loss(outputs['reconstructed_B'], real_B)
            else:
                outputs = model(real_A=real_A, real_B=real_B, mode='full')
                cycle_loss_A = loss_manager.cycle_loss(outputs['reconstructed_A'], real_A)
                cycle_loss_B = loss_manager.cycle_loss(outputs['reconstructed_B'], real_B)
            
            meters['cycle_A'].update(cycle_loss_A.item())
            meters['cycle_B'].update(cycle_loss_B.item())
            
            # Save sample images
            if i == 0 and epoch % config.SAVE_SAMPLE_FREQ == 0:
                save_path = config.SAMPLE_DIR / f'epoch_{epoch:04d}.tif'
                save_comparison_grid(
                    real_A, outputs['fake_B'], outputs['reconstructed_A'],
                    real_B, outputs['fake_A'], outputs['reconstructed_B'],
                    save_path,
                    mean=config.NORMALIZE_MEAN,
                    std=config.NORMALIZE_STD
                )
                print(f"Saved sample images to {save_path}")
    
    val_losses = {key: meter.avg for key, meter in meters.items()}
    print(f"Validation - Cycle A: {val_losses['cycle_A']:.4f}, Cycle B: {val_losses['cycle_B']:.4f}")
    
    # TensorBoard logging
    if writer is not None:
        for key, value in val_losses.items():
            writer.add_scalar(f'Val/{key}', value, epoch)
    
    return val_losses


def train(config):
    """
    Main training function for HPC.
    
    Args:
        config: Configuration object
    """
    print("\n" + "="*60)
    print("Starting Cycle-CARE Training (HPC Optimized)")
    print("="*60)
    
    config.print_config()
    
    # Setup
    model, optimizers, image_pools, loss_manager, device, scaler = setup_training(config)
    
    # Get data loaders with HPC-optimized settings
    print("\nLoading datasets...")
    train_loader, val_loader = get_dataloaders(
        config,
        auto_split=config.AUTO_SPLIT_DATA,
        val_split=config.VAL_SPLIT_RATIO
    )
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Effective batch size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
    
    # TensorBoard writer
    writer = None
    if config.USE_TENSORBOARD:
        log_dir = config.LOG_DIR / config.EXPERIMENT_NAME
        writer = SummaryWriter(log_dir)
        print(f"\nTensorBoard logs will be saved to {log_dir}")
    
    # Resume training if specified
    start_epoch = 1
    best_val_loss = float('inf')
    
    if config.RESUME_TRAINING and config.RESUME_CHECKPOINT is not None:
        checkpoint = load_checkpoint(
            config.RESUME_CHECKPOINT,
            model,
            optimizers['G'],
            optimizers['D_A'],
            optimizers['D_B'],
            device
        )
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch-1}")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training loop...")
    print("="*60 + "\n")
    
    total_start_time = time.time()
    
    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config.NUM_EPOCHS}")
        print(f"{'='*60}")
        
        # Update learning rate (linear decay after certain epoch)
        if epoch > config.LR_DECAY_START_EPOCH:
            decay_epochs = config.LR_DECAY_END_EPOCH - config.LR_DECAY_START_EPOCH
            decay_progress = (epoch - config.LR_DECAY_START_EPOCH) / decay_epochs
            new_lr = config.LEARNING_RATE * (1.0 - decay_progress)
            for optimizer in optimizers.values():
                update_learning_rate(optimizer, new_lr)
            print(f"Learning rate: {new_lr:.6f}")
        
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizers, image_pools, loss_manager,
            epoch, config, scaler, writer
        )
        
        # Validate
        val_losses = validate(model, val_loader, loss_manager, epoch, config, writer)
        
        # Save checkpoint
        if epoch % config.SAVE_CHECKPOINT_FREQ == 0:
            # Handle DataParallel for state dict
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            
            checkpoint_path = config.CHECKPOINT_DIR / f'checkpoint_epoch_{epoch:04d}.pth'
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_G_state_dict': optimizers['G'].state_dict(),
                'optimizer_D_A_state_dict': optimizers['D_A'].state_dict(),
                'optimizer_D_B_state_dict': optimizers['D_B'].state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'config': config
            }, checkpoint_path)
        
        # Save best model
        current_val_loss = val_losses['cycle_B']  # Use restoration cycle loss as metric
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            
            # Handle DataParallel for state dict
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            
            best_model_path = config.CHECKPOINT_DIR / 'best_model.pth'
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_G_state_dict': optimizers['G'].state_dict(),
                'optimizer_D_A_state_dict': optimizers['D_A'].state_dict(),
                'optimizer_D_B_state_dict': optimizers['D_B'].state_dict(),
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'config': config
            }, best_model_path)
            print(f"✓ New best model saved! (val cycle_B loss: {best_val_loss:.4f})")
        
        # Estimate remaining time
        elapsed_time = time.time() - total_start_time
        avg_epoch_time = elapsed_time / (epoch - start_epoch + 1)
        remaining_epochs = config.NUM_EPOCHS - epoch
        estimated_remaining = avg_epoch_time * remaining_epochs
        print(f"Estimated remaining time: {estimated_remaining/3600:.1f} hours")
    
    total_time = time.time() - total_start_time
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Total training time: {total_time/3600:.2f} hours")
    print("="*60)
    
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    # Load configuration
    config = Config()
    
    # Start training
    train(config)
