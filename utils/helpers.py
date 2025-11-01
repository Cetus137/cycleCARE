"""
Helper utilities for Cycle-CARE.
Includes checkpoint management, image processing, and visualization.
"""

import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def save_checkpoint(state, filename):
    """
    Save a training checkpoint.
    
    Args:
        state (dict): State dictionary containing model, optimizer, and training info
        filename (str or Path): Path to save checkpoint
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(filename, model, optimizer_G=None, optimizer_D_A=None, optimizer_D_B=None, device='cuda'):
    """
    Load a training checkpoint.
    
    Args:
        filename (str or Path): Path to checkpoint
        model: Cycle-CARE model
        optimizer_G: Generator optimizer (optional)
        optimizer_D_A: Discriminator A optimizer (optional)
        optimizer_D_B: Discriminator B optimizer (optional)
        device (str): Device to load checkpoint to
    
    Returns:
        dict: Checkpoint state
    """
    checkpoint = torch.load(filename, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer_G is not None and 'optimizer_G_state_dict' in checkpoint:
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    
    if optimizer_D_A is not None and 'optimizer_D_A_state_dict' in checkpoint:
        optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A_state_dict'])
    
    if optimizer_D_B is not None and 'optimizer_D_B_state_dict' in checkpoint:
        optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B_state_dict'])
    
    print(f"Checkpoint loaded from {filename}")
    print(f"Resuming from epoch {checkpoint.get('epoch', 0)}")
    
    return checkpoint


def denormalize(tensor, mean=0.5, std=0.5):
    """
    Denormalize a tensor from [-1, 1] to [0, 1] range.
    
    Args:
        tensor (torch.Tensor): Normalized tensor
        mean (float): Mean used for normalization
        std (float): Std used for normalization
    
    Returns:
        torch.Tensor: Denormalized tensor
    """
    # Reverse normalization: x = x * std + mean
    denorm = tensor * std + mean
    return torch.clamp(denorm, 0, 1)


def tensor_to_image(tensor, mean=0.5, std=0.5):
    """
    Convert a tensor to a PIL Image.
    
    Args:
        tensor (torch.Tensor): Image tensor of shape (C, H, W) or (B, C, H, W)
        mean (float): Mean used for normalization
        std (float): Std used for normalization
    
    Returns:
        PIL.Image: Image
    """
    # Handle batch dimension
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Denormalize
    tensor = denormalize(tensor, mean, std)
    
    # Convert to numpy
    array = tensor.cpu().numpy()
    
    # Transpose to (H, W, C) if needed
    if array.shape[0] in [1, 3]:
        array = np.transpose(array, (1, 2, 0))
    
    # Squeeze if grayscale
    if array.shape[2] == 1:
        array = array.squeeze(2)
    
    # Convert to uint8
    array = (array * 255).astype(np.uint8)
    
    # Convert to PIL Image
    if len(array.shape) == 2:  # Grayscale
        image = Image.fromarray(array, mode='L')
    else:  # RGB
        image = Image.fromarray(array, mode='RGB')
    
    return image


def save_image(tensor, path, mean=0.5, std=0.5):
    """
    Save a tensor as an image file.
    
    Args:
        tensor (torch.Tensor): Image tensor
        path (str or Path): Path to save image
        mean (float): Mean used for normalization
        std (float): Std used for normalization
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    image = tensor_to_image(tensor, mean, std)
    image.save(path)


def save_comparison_grid(real_A, fake_B, reconstructed_A, real_B, fake_A, reconstructed_B,
                         save_path, mean=0.5, std=0.5, max_images=4):
    """
    Save a grid comparing real, fake, and reconstructed images.
    
    Args:
        real_A: Real images from domain A
        fake_B: Fake images for domain B (from real_A)
        reconstructed_A: Reconstructed A (from fake_B)
        real_B: Real images from domain B
        fake_A: Fake images for domain A (from real_B)
        reconstructed_B: Reconstructed B (from fake_A)
        save_path (str or Path): Path to save grid
        mean (float): Mean used for normalization
        std (float): Std used for normalization
        max_images (int): Maximum number of images to show
    """
    # Limit number of images
    batch_size = min(real_A.size(0), max_images)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Convert tensors to images
    def to_img(tensor, idx):
        return tensor_to_image(tensor[idx].unsqueeze(0), mean, std)
    
    # Plot images for first sample
    idx = 0
    
    # Top row: A -> B -> A cycle
    axes[0, 0].imshow(to_img(real_A, idx), cmap='gray' if real_A.size(1) == 1 else None)
    axes[0, 0].set_title('Real A (Clean)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(to_img(fake_B, idx), cmap='gray' if fake_B.size(1) == 1 else None)
    axes[0, 1].set_title('Fake B (A→B)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(to_img(reconstructed_A, idx), cmap='gray' if reconstructed_A.size(1) == 1 else None)
    axes[0, 2].set_title('Reconstructed A')
    axes[0, 2].axis('off')
    
    # Bottom row: B -> A -> B cycle
    axes[1, 0].imshow(to_img(real_B, idx), cmap='gray' if real_B.size(1) == 1 else None)
    axes[1, 0].set_title('Real B (Noisy)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(to_img(fake_A, idx), cmap='gray' if fake_A.size(1) == 1 else None)
    axes[1, 1].set_title('Fake A (B→A, Restored)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(to_img(reconstructed_B, idx), cmap='gray' if reconstructed_B.size(1) == 1 else None)
    axes[1, 2].set_title('Reconstructed B')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def get_learning_rate(optimizer):
    """
    Get the current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
    
    Returns:
        float: Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def update_learning_rate(optimizer, new_lr):
    """
    Update the learning rate of an optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        new_lr (float): New learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly after a warmup period.
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps (int): Number of warmup steps
        num_training_steps (int): Total number of training steps
        last_epoch (int): The index of the last epoch
    
    Returns:
        torch.optim.lr_scheduler: Learning rate scheduler
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def set_requires_grad(nets, requires_grad=False):
    """
    Set requires_grad for all parameters in a list of networks.
    
    Useful for freezing networks during certain phases of training.
    
    Args:
        nets (list or nn.Module): List of networks or single network
        requires_grad (bool): Whether parameters require gradients
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class ImagePool:
    """
    Image pool for storing previously generated images.
    
    This helps reduce model oscillation by showing the discriminator
    a history of generated images rather than only the latest ones.
    
    Args:
        pool_size (int): Maximum number of images to store (0 to disable)
    """
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []
    
    def query(self, images):
        """
        Return images from the pool.
        
        With probability 0.5, return input images.
        With probability 0.5, return an image from the pool and insert the input image.
        
        Args:
            images (torch.Tensor): Latest generated images
        
        Returns:
            torch.Tensor: Images from pool or input images
        """
        if self.pool_size == 0:
            return images
        
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = np.random.uniform(0, 1)
                if p > 0.5:
                    random_id = np.random.randint(0, self.pool_size)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        
        return_images = torch.cat(return_images, 0)
        return return_images


class AverageMeter:
    """Compute and store the average and current value."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_training_info(epoch, num_epochs, iter_num, num_iters, losses, time_elapsed=None):
    """
    Print training information.
    
    Args:
        epoch (int): Current epoch
        num_epochs (int): Total number of epochs
        iter_num (int): Current iteration
        num_iters (int): Total iterations per epoch
        losses (dict): Dictionary of loss values
        time_elapsed (float): Time elapsed (optional)
    """
    info = f"Epoch [{epoch}/{num_epochs}] Iter [{iter_num}/{num_iters}]"
    
    for key, value in losses.items():
        info += f" {key}: {value:.4f}"
    
    if time_elapsed is not None:
        info += f" Time: {time_elapsed:.2f}s"
    
    print(info)


if __name__ == "__main__":
    print("Testing helper utilities...")
    
    # Test denormalization
    print("\n1. Testing denormalization...")
    tensor = torch.randn(1, 1, 256, 256)
    denorm = denormalize(tensor)
    print(f"   Original range: [{tensor.min():.2f}, {tensor.max():.2f}]")
    print(f"   Denormalized range: [{denorm.min():.2f}, {denorm.max():.2f}]")
    print("   ✓ Test passed!")
    
    # Test image pool
    print("\n2. Testing image pool...")
    pool = ImagePool(pool_size=5)
    images = torch.randn(2, 1, 256, 256)
    returned = pool.query(images)
    print(f"   Input shape: {images.shape}")
    print(f"   Returned shape: {returned.shape}")
    print("   ✓ Test passed!")
    
    print("\n✓ All helper tests passed!")
