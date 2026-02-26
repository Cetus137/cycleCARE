"""
Training script for 3D Cycle-CARE model - HPC Optimized.
Implements full 3D volumetric training with memory optimizations.
Based on train.py structure for consistency.
"""

import os
import time
import torch
import torch.nn as nn
import random
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import numpy as np

from config_3d import Config3D as Config
from models.cycle_care_3d import CycleCARE3D
from data.dataset_3d import get_dataloaders_3d
from utils.losses import CycleCarelosses
from utils.helpers import (
    save_checkpoint, load_checkpoint, save_comparison_grid,
    get_learning_rate, update_learning_rate, set_requires_grad,
    ImagePool, AverageMeter, print_training_info
)


def config_to_dict(config):
    """
    Convert Config object to dictionary for safe checkpoint saving.
    
    Args:
        config: Configuration object with class-level attributes
    
    Returns:
        Dictionary containing all configuration values
    """
    config_dict = {}
    
    # Get all public attributes from the config class
    for attr in dir(config):
        if not attr.startswith('_') and not callable(getattr(config, attr)):
            value = getattr(config, attr)
            # Convert Path objects to strings for serialization
            if isinstance(value, Path):
                value = str(value)
            config_dict[attr] = value
    
    return config_dict


def plot_losses(loss_history, epoch, config):
    """
    Plot loss curves and save to disk (replaces previous plot).
    
    Args:
        loss_history: Dictionary with keys 'train' and 'val', each containing
                     dictionaries of loss lists (e.g., {'G_total': [...], ...})
        epoch (int): Current epoch number
        config: Configuration object
    """
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f'3D Training Progress - Epoch {epoch}/{config.NUM_EPOCHS}', fontsize=16, fontweight='bold')
    
    epochs = list(range(1, epoch + 1))
    
    # Plot 1: Total Generator Loss
    ax = axes[0, 0]
    if 'G_total' in loss_history['train']:
        ax.plot(epochs, loss_history['train']['G_total'], 'b-', linewidth=2, label='Train')
    ax.set_title('Total Generator Loss', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Generator Adversarial Losses
    ax = axes[0, 1]
    if 'G_AB' in loss_history['train']:
        ax.plot(epochs, loss_history['train']['G_AB'], 'g-', linewidth=2, label='G_AB (Surface→Deep)')
    if 'G_BA' in loss_history['train']:
        ax.plot(epochs, loss_history['train']['G_BA'], 'm-', linewidth=2, label='G_BA (Deep→Surface)')
    ax.set_title('Generator Adversarial Losses', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Cycle Consistency Losses
    ax = axes[0, 2]
    if 'cycle_A' in loss_history['train']:
        ax.plot(epochs, loss_history['train']['cycle_A'], 'c-', linewidth=2, label='Cycle A (Surface→Deep→Surface)')
    if 'cycle_B' in loss_history['train']:
        ax.plot(epochs, loss_history['train']['cycle_B'], 'orange', linewidth=2, label='Cycle B (Deep→Surface→Deep)')
    if 'cycle_A' in loss_history['val']:
        ax.plot(epochs, loss_history['val']['cycle_A'], 'c--', linewidth=2, alpha=0.7, label='Val Cycle A')
    if 'cycle_B' in loss_history['val']:
        ax.plot(epochs, loss_history['val']['cycle_B'], 'orange', linewidth=2, linestyle='--', alpha=0.7, label='Val Cycle B')
    ax.set_title('Cycle Consistency Losses', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Identity Losses
    ax = axes[1, 0]
    if 'identity_A' in loss_history['train']:
        ax.plot(epochs, loss_history['train']['identity_A'], 'r-', linewidth=2, label='Identity A (Surface)')
    if 'identity_B' in loss_history['train']:
        ax.plot(epochs, loss_history['train']['identity_B'], 'purple', linewidth=2, label='Identity B (Deep)')
    ax.set_title('Identity Preservation Losses', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 5: Discriminator Losses
    ax = axes[1, 1]
    if 'D_A' in loss_history['train']:
        ax.plot(epochs, loss_history['train']['D_A'], 'brown', linewidth=2, label='D_A (Surface domain)')
    if 'D_B' in loss_history['train']:
        ax.plot(epochs, loss_history['train']['D_B'], 'olive', linewidth=2, label='D_B (Deep domain)')
    ax.set_title('Discriminator Losses', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 6: Loss Ratios (for balance analysis)
    ax = axes[1, 2]
    if all(k in loss_history['train'] for k in ['cycle_B', 'G_BA', 'D_B']):
        cycle_ratio = [c / (g + 1e-8) for c, g in zip(loss_history['train']['cycle_B'], loss_history['train']['G_BA'])]
        disc_ratio = [d / (g + 1e-8) for d, g in zip(loss_history['train']['D_B'], loss_history['train']['G_BA'])]
        ax.plot(epochs, cycle_ratio, 'b-', linewidth=2, label='Cycle/Adv Ratio')
        ax.plot(epochs, disc_ratio, 'r-', linewidth=2, label='Disc/Gen Ratio')
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Balanced (1.0)')
    ax.set_title('Loss Component Ratios', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Ratio')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 7: Recent trends (last 20% of epochs or last 20 epochs, whichever is smaller)
    ax = axes[2, 0]
    window = min(20, max(1, epoch // 5))
    recent_epochs = epochs[-window:]
    if 'G_total' in loss_history['train'] and len(recent_epochs) > 1:
        ax.plot(recent_epochs, loss_history['train']['G_total'][-window:], 'b-', linewidth=2, marker='o', label='G_total')
    if 'cycle_B' in loss_history['train'] and len(recent_epochs) > 1:
        ax.plot(recent_epochs, loss_history['train']['cycle_B'][-window:], 'orange', linewidth=2, marker='s', label='Cycle B')
    ax.set_title(f'Recent Trends (Last {window} Epochs)', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 8: Validation Cycle Losses (zoomed)
    ax = axes[2, 1]
    if 'cycle_B' in loss_history['val']:
        ax.plot(epochs, loss_history['val']['cycle_B'], 'orange', linewidth=2, marker='o', markersize=3, label='Val Cycle B (Restoration)')
    if 'cycle_A' in loss_history['val']:
        ax.plot(epochs, loss_history['val']['cycle_A'], 'c', linewidth=2, marker='s', markersize=3, label='Val Cycle A')
    ax.set_title('Validation Performance', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 9: Statistics text
    ax = axes[2, 2]
    ax.axis('off')
    stats_text = f"""EPOCH {epoch} STATISTICS\n\n"""
    if epoch > 0:
        stats_text += f"Generator Total: {loss_history['train']['G_total'][-1]:.4f}\n"
        stats_text += f"Cycle B (Restoration): {loss_history['train']['cycle_B'][-1]:.4f}\n"
        stats_text += f"Val Cycle B: {loss_history['val']['cycle_B'][-1]:.4f}\n\n"
        
        if epoch > 1:
            prev_val = loss_history['val']['cycle_B'][-2]
            curr_val = loss_history['val']['cycle_B'][-1]
            change = ((curr_val - prev_val) / prev_val) * 100
            stats_text += f"Val change: {change:+.2f}%\n"
    
    ax.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = config.OUTPUT_ROOT / 'loss_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to {plot_path}")


def plot_iter_losses(iter_history, epoch, config):
    """
    Plot per-iteration loss curves (finer resolution than epoch averages).
    
    Args:
        iter_history: List of dicts, one entry per LOG_FREQ iterations across all epochs
        epoch (int): Current epoch number
        config: Configuration object
    """
    if len(iter_history) < 2:
        return
    
    iters = list(range(1, len(iter_history) + 1))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Per-Iteration Losses - Epoch {epoch}/{config.NUM_EPOCHS}', fontsize=14, fontweight='bold')
    
    # Plot 1: Total generator loss per iteration
    ax = axes[0, 0]
    ax.plot(iters, [d['G_total'] for d in iter_history], 'b-', linewidth=0.8, alpha=0.7)
    ax.set_title('Total Generator Loss (per iter)', fontweight='bold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Generator adversarial losses
    ax = axes[0, 1]
    ax.plot(iters, [d['G_AB'] for d in iter_history], 'g-', linewidth=0.8, alpha=0.7, label='G_AB (Surface→Deep)')
    ax.plot(iters, [d['G_BA'] for d in iter_history], 'm-', linewidth=0.8, alpha=0.7, label='G_BA (Deep→Surface)')
    ax.set_title('Generator Adversarial Losses (per iter)', fontweight='bold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Plot 3: Cycle consistency losses
    ax = axes[1, 0]
    ax.plot(iters, [d['cycle_A'] for d in iter_history], 'c-', linewidth=0.8, alpha=0.7, label='Cycle A')
    ax.plot(iters, [d['cycle_B'] for d in iter_history], 'orange', linewidth=0.8, alpha=0.7, label='Cycle B (Restoration)')
    ax.set_title('Cycle Consistency Losses (per iter)', fontweight='bold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    # Plot 4: Discriminator losses — most informative for collapse detection
    ax = axes[1, 1]
    ax.plot(iters, [d['D_A'] for d in iter_history], 'brown', linewidth=0.8, alpha=0.7, label='D_A (Surface)')
    ax.plot(iters, [d['D_B'] for d in iter_history], 'olive', linewidth=0.8, alpha=0.7, label='D_B (Deep)')
    ax.axhline(y=0.25, color='k', linestyle='--', alpha=0.4, linewidth=1, label='Random baseline (0.25)')
    ax.axhline(y=0.05, color='r', linestyle='--', alpha=0.4, linewidth=1, label='Collapse threshold (0.05)')
    ax.set_title('Discriminator Losses (per iter)', fontweight='bold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plot_path = config.OUTPUT_ROOT / 'loss_curves_iter.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Per-iter loss curves saved to {plot_path}")


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
    print("\nCreating 3D Cycle-CARE model...")
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
        lr=config.LEARNING_RATE_D,
        betas=(config.BETA1, config.BETA2)
    )
    
    optimizer_D_B = torch.optim.Adam(
        model_ref.D_B.parameters(),
        lr=config.LEARNING_RATE_D,
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
    
    # Create image pools to store generated images (200 keeps broader history, reduces mode cycling)
    fake_A_pool = ImagePool(pool_size=200)
    fake_B_pool = ImagePool(pool_size=200)
    image_pools = {'A': fake_A_pool, 'B': fake_B_pool}
    
    # Create loss manager
    loss_manager = CycleCarelosses(config, device)
    
    return model, optimizers, image_pools, loss_manager, device, scaler


def train_epoch(model, train_loader, optimizers, image_pools, loss_manager, 
                epoch, config, scaler=None, writer=None):
    """
    Train for one epoch with HPC optimizations.
    
    Args:
        model: 3D Cycle-CARE model
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
    iter_losses = []  # per-iteration losses for fine-grained plotting
    
    for i, batch in enumerate(train_loader):
        iter_start_time = time.time()
        
        # Get data
        real_A = batch['A'].to(config.DEVICE, non_blocking=True)
        real_B = batch['B'].to(config.DEVICE, non_blocking=True)
        
        # ===================== Train Generators =====================
        set_requires_grad([model_ref.D_A, model_ref.D_B], False)  # Freeze discriminators
        optimizers['G'].zero_grad()
        
        # Mixed precision forward pass - also caches fake_A, fake_B for discriminator reuse
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
        
        # Generate fakes ONCE for both discriminators (reused below via image pools)
        with torch.no_grad():
            if config.MIXED_PRECISION:
                with autocast():
                    fake_A = model_ref.G_BA(real_B)
                    fake_B = model_ref.G_AB(real_A)
            else:
                fake_A = model_ref.G_BA(real_B)
                fake_B = model_ref.G_AB(real_A)
        
        # Query image pools (for training stability via historical fakes)
        fake_A_pooled = image_pools['A'].query(fake_A)
        fake_B_pooled = image_pools['B'].query(fake_B)
        
        # ===================== Train Discriminator A =====================
        set_requires_grad(model_ref.D_A, True)
        set_requires_grad(model_ref.D_B, False)
        optimizers['D_A'].zero_grad()
        
        if config.MIXED_PRECISION:
            with autocast():
                loss_D_A, _, loss_dict_D = loss_manager.compute_discriminator_losses_from_fakes(
                    real_A, real_B, fake_A_pooled, fake_B_pooled, model_ref.D_A, model_ref.D_B
                )
            scaler.scale(loss_D_A).backward()
            scaler.step(optimizers['D_A'])
            scaler.update()
        else:
            loss_D_A, _, loss_dict_D = loss_manager.compute_discriminator_losses_from_fakes(
                real_A, real_B, fake_A_pooled, fake_B_pooled, model_ref.D_A, model_ref.D_B
            )
            loss_D_A.backward()
            optimizers['D_A'].step()
        
        # ===================== Train Discriminator B =====================
        set_requires_grad(model_ref.D_A, False)
        set_requires_grad(model_ref.D_B, True)
        optimizers['D_B'].zero_grad()
        
        if config.MIXED_PRECISION:
            with autocast():
                _, loss_D_B, _ = loss_manager.compute_discriminator_losses_from_fakes(
                    real_A, real_B, fake_A_pooled, fake_B_pooled, model_ref.D_A, model_ref.D_B
                )
            scaler.scale(loss_D_B).backward()
            scaler.step(optimizers['D_B'])
            scaler.update()
        else:
            _, loss_D_B, _ = loss_manager.compute_discriminator_losses_from_fakes(
                real_A, real_B, fake_A_pooled, fake_B_pooled, model_ref.D_A, model_ref.D_B
            )
            loss_D_B.backward()
            optimizers['D_B'].step()
        
        # ===================== Second Generator Update (optional, config.GENERATOR_UPDATES_PER_ITER >= 2) =====================
        if config.GENERATOR_UPDATES_PER_ITER >= 2:
            set_requires_grad([model_ref.D_A, model_ref.D_B], False)
            optimizers['G'].zero_grad()

            if config.MIXED_PRECISION:
                with autocast():
                    loss_G2, loss_dict_G2 = loss_manager.compute_generator_loss(
                        model, real_A, real_B, model_ref.D_A, model_ref.D_B
                    )
                scaler.scale(loss_G2).backward()
                scaler.step(optimizers['G'])
                scaler.update()
            else:
                loss_G2, loss_dict_G2 = loss_manager.compute_generator_loss(
                    model, real_A, real_B, model_ref.D_A, model_ref.D_B
                )
                loss_G2.backward()
                optimizers['G'].step()
        else:
            loss_G2, loss_dict_G2 = loss_dict_G, loss_dict_G  # reuse first update for meters

        # ===================== Update Meters =====================
        # Average both G update steps in the meters (same values when GENERATOR_UPDATES_PER_ITER=1)
        for key in ['G_total', 'G_AB', 'G_BA', 'cycle_A', 'cycle_B', 'identity_A', 'identity_B']:
            meters[key].update((loss_dict_G[key] + loss_dict_G2[key]) / 2)
        meters['D_A'].update(loss_dict_D['D_A_total'])
        meters['D_B'].update(loss_dict_D['D_B_total'])
        
        # Collect per-iteration losses (every LOG_FREQ iters to limit memory)
        if (i + 1) % config.LOG_FREQ == 0 or (i + 1) == num_iters:
            iter_losses.append({
                'G_total': (loss_dict_G['G_total'] + loss_dict_G2['G_total']) / 2,
                'G_AB':    (loss_dict_G['G_AB']    + loss_dict_G2['G_AB'])    / 2,
                'G_BA':    (loss_dict_G['G_BA']    + loss_dict_G2['G_BA'])    / 2,
                'cycle_A': (loss_dict_G['cycle_A'] + loss_dict_G2['cycle_A']) / 2,
                'cycle_B': (loss_dict_G['cycle_B'] + loss_dict_G2['cycle_B']) / 2,
                'D_A': loss_dict_D['D_A_total'],
                'D_B': loss_dict_D['D_B_total'],
            })
        
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
    
    # Return average losses AND per-iteration losses
    return {key: meter.avg for key, meter in meters.items()}, iter_losses


def save_3d_comparison_grid(real_A, fake_B, reconstructed_A, real_B, fake_A, reconstructed_B,
                            save_path, mean=0.5, std=0.5):
    """
    Save comparison grid for 3D volumes by extracting central Z-plane.
    
    Args:
        real_A: Real volumes from domain A (B, 1, D, H, W)
        fake_B: Generated volumes in domain B (B, 1, D, H, W)
        reconstructed_A: Reconstructed volumes in domain A (B, 1, D, H, W)
        real_B: Real volumes from domain B (B, 1, D, H, W)
        fake_A: Generated volumes in domain A (B, 1, D, H, W)
        reconstructed_B: Reconstructed volumes in domain B (B, 1, D, H, W)
        save_path: Path to save the comparison grid
        mean: Normalization mean used during preprocessing
        std: Normalization std used during preprocessing
    """
    # Extract central Z-plane from each volume (shape: B, 1, D, H, W -> B, 1, H, W)
    D = real_A.shape[2]
    central_z = D // 2
    
    real_A_2d = real_A[:, :, central_z, :, :]
    fake_B_2d = fake_B[:, :, central_z, :, :]
    reconstructed_A_2d = reconstructed_A[:, :, central_z, :, :]
    real_B_2d = real_B[:, :, central_z, :, :]
    fake_A_2d = fake_A[:, :, central_z, :, :]
    reconstructed_B_2d = reconstructed_B[:, :, central_z, :, :]
    
    # Use the existing 2D save function
    save_comparison_grid(
        real_A_2d, fake_B_2d, reconstructed_A_2d,
        real_B_2d, fake_A_2d, reconstructed_B_2d,
        save_path,
        mean=mean,
        std=std
    )


def validate(model, val_loader, loss_manager, epoch, config, writer=None):
    """
    Validate the 3D model.
    
    Args:
        model: 3D Cycle-CARE model
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
    
    # Select a random batch index for saving sample images
    save_sample_idx = random.randint(0, len(val_loader) - 1) if epoch % config.SAVE_SAMPLE_FREQ == 0 else -1
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            real_A = batch['A'].to(config.DEVICE, non_blocking=True)
            real_B = batch['B'].to(config.DEVICE, non_blocking=True)
            
            # Forward pass (with mixed precision if enabled)
            if config.MIXED_PRECISION:
                with autocast():
                    outputs = model(real_A=real_A, real_B=real_B, mode='full')
                    cycle_loss_A = loss_manager.cycle_loss(outputs['reconstructed_A'], real_A) * loss_manager.lambda_cycle
                    cycle_loss_B = loss_manager.cycle_loss(outputs['reconstructed_B'], real_B) * loss_manager.lambda_cycle
            else:
                outputs = model(real_A=real_A, real_B=real_B, mode='full')
                cycle_loss_A = loss_manager.cycle_loss(outputs['reconstructed_A'], real_A) * loss_manager.lambda_cycle
                cycle_loss_B = loss_manager.cycle_loss(outputs['reconstructed_B'], real_B) * loss_manager.lambda_cycle
            
            meters['cycle_A'].update(cycle_loss_A.item())
            meters['cycle_B'].update(cycle_loss_B.item())
            
            # Save sample images from a random batch each epoch
            if i == save_sample_idx:
                save_path = config.SAMPLE_DIR / f'epoch_{epoch:04d}.tif'
                
                save_3d_comparison_grid(
                    real_A, outputs['fake_B'], outputs['reconstructed_A'],
                    real_B, outputs['fake_A'], outputs['reconstructed_B'],
                    save_path,
                    mean=config.NORMALIZE_MEAN,
                    std=config.NORMALIZE_STD
                )
                print(f"Saved sample images (central Z-plane) to {save_path} (batch {i+1}/{len(val_loader)})")
    
    val_losses = {key: meter.avg for key, meter in meters.items()}
    print(f"Validation - Cycle A: {val_losses['cycle_A']:.4f}, Cycle B: {val_losses['cycle_B']:.4f}")
    
    # TensorBoard logging
    if writer is not None:
        for key, value in val_losses.items():
            writer.add_scalar(f'Val/{key}', value, epoch)
    
    return val_losses


def train(config):
    """
    Main training function for 3D HPC.
    
    Args:
        config: Configuration object
    """
    print("\n" + "="*60)
    print("Starting 3D Cycle-CARE Training (HPC Optimized)")
    print("="*60)
    
    config.print_config()
    config.estimate_memory()
    
    # Setup
    model, optimizers, image_pools, loss_manager, device, scaler = setup_training(config)
    
    # Get data loaders with HPC-optimized settings
    print("\nLoading datasets...")
    train_loader, val_loader = get_dataloaders_3d(
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
    
    # Initialize loss history tracking
    loss_history = {
        'train': {
            'G_total': [], 'G_AB': [], 'G_BA': [],
            'cycle_A': [], 'cycle_B': [],
            'identity_A': [], 'identity_B': [],
            'D_A': [], 'D_B': []
        },
        'val': {
            'cycle_A': [], 'cycle_B': []
        }
    }
    iter_history = []  # per-iteration losses across all epochs
    
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
        
        # Restore loss history if available
        if 'loss_history' in checkpoint:
            loss_history = checkpoint['loss_history']
            print(f"Restored loss history from checkpoint")
        if 'iter_history' in checkpoint:
            iter_history = checkpoint['iter_history']
            print(f"Restored iter history ({len(iter_history)} entries) from checkpoint")
        
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
        # G and D decay separately to preserve the G:D LR ratio
        if epoch > config.LR_DECAY_START_EPOCH:
            decay_epochs = config.LR_DECAY_END_EPOCH - config.LR_DECAY_START_EPOCH
            decay_progress = (epoch - config.LR_DECAY_START_EPOCH) / decay_epochs
            scale = 1.0 - decay_progress
            new_lr_G = config.LEARNING_RATE * scale
            new_lr_D = config.LEARNING_RATE_D * scale
            update_learning_rate(optimizers['G'], new_lr_G)
            update_learning_rate(optimizers['D_A'], new_lr_D)
            update_learning_rate(optimizers['D_B'], new_lr_D)
            print(f"Learning rate — G: {new_lr_G:.2e}, D: {new_lr_D:.2e}")
        
        # Train
        train_losses, new_iter_losses = train_epoch(
            model, train_loader, optimizers, image_pools, loss_manager,
            epoch, config, scaler, writer
        )
        iter_history.extend(new_iter_losses)
        
        # Validate
        val_losses = validate(model, val_loader, loss_manager, epoch, config, writer)
        
        # Update loss history
        for key in loss_history['train'].keys():
            if key in train_losses:
                loss_history['train'][key].append(train_losses[key])
        for key in loss_history['val'].keys():
            if key in val_losses:
                loss_history['val'][key].append(val_losses[key])
        
        # Plot loss curves (replaces previous plot)
        plot_losses(loss_history, epoch, config)
        plot_iter_losses(iter_history, epoch, config)
        
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
                'loss_history': loss_history,
                'iter_history': iter_history[-5000:],  # keep last 5000 entries (~few epochs)
                'config': config_to_dict(config)  # Save as dict for proper serialization
            }, checkpoint_path)
        
        # Save best model
        current_val_loss = val_losses['cycle_B']  # Use restoration cycle loss as metric
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            
            best_model_path = config.CHECKPOINT_DIR / 'best_model.pth'
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_G_state_dict': optimizers['G'].state_dict(),
                'optimizer_D_A_state_dict': optimizers['D_A'].state_dict(),
                'optimizer_D_B_state_dict': optimizers['D_B'].state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'loss_history': loss_history,
                'iter_history': iter_history[-5000:],
                'config': config_to_dict(config)
            }, best_model_path)
            print(f"Saved best model (val_loss: {best_val_loss:.4f})")
    
    # Training complete
    total_time = time.time() - total_start_time
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)
    
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    # Load configuration
    config = Config()
    
    # Run training
    train(config)
