"""
Cycle-CARE Model: Combines two generators and two discriminators for unpaired
image-to-image translation with cycle-consistency.
"""

import torch
import torch.nn as nn
from .generator import CAREUNet
from .discriminator import PatchGANDiscriminator


class CycleCARE(nn.Module):
    """
    Cycle-CARE: Unpaired image restoration using CycleGAN framework with CARE-style generators.
    
    The model consists of:
    - G_AB: Generator that translates from domain A (clean) to domain B (noisy)
    - G_BA: Generator that translates from domain B (noisy) to domain A (clean) - RESTORATION
    - D_A: Discriminator for domain A (distinguishes real clean from fake clean)
    - D_B: Discriminator for domain B (distinguishes real noisy from fake noisy)
    
    The model is trained with:
    - Adversarial loss: Makes generated images indistinguishable from real ones
    - Cycle-consistency loss: G_BA(G_AB(A)) ≈ A and G_AB(G_BA(B)) ≈ B
    - Identity loss: G_BA(A) ≈ A and G_AB(B) ≈ B (preserves image content)
    
    Args:
        img_channels (int): Number of image channels
        unet_depth (int): Depth of U-Net generators
        unet_filters (int): Base number of filters in generators
        unet_kernel_size (int): Kernel size for generator convolutions
        disc_filters (int): Base number of filters in discriminators
        disc_num_layers (int): Number of layers in discriminators
        disc_kernel_size (int): Kernel size for discriminator convolutions
        disc_channels (int): Number of input channels for discriminators (default: same as img_channels)
        use_batch_norm (bool): Whether to use batch normalization
        use_dropout (bool): Whether to use dropout in generators
        dropout_rate (float): Dropout probability
    """
    def __init__(self, img_channels=1, unet_depth=3, unet_filters=64, unet_kernel_size=3,
                 disc_filters=64, disc_num_layers=3, disc_kernel_size=4, disc_channels=None,
                 use_batch_norm=True, use_dropout=True, dropout_rate=0.5):
        super(CycleCARE, self).__init__()
        
        # Default disc_channels to img_channels if not specified
        if disc_channels is None:
            disc_channels = img_channels
        
        # Generators
        self.G_AB = CAREUNet(
            in_channels=img_channels,
            out_channels=img_channels,
            depth=unet_depth,
            base_filters=unet_filters,
            kernel_size=unet_kernel_size,
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate
        )
        
        self.G_BA = CAREUNet(
            in_channels=img_channels,
            out_channels=img_channels,
            depth=unet_depth,
            base_filters=unet_filters,
            kernel_size=unet_kernel_size,
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate
        )
        
        # Discriminators
        self.D_A = PatchGANDiscriminator(
            in_channels=disc_channels,
            base_filters=disc_filters,
            num_layers=disc_num_layers,
            kernel_size=disc_kernel_size,
            use_batch_norm=use_batch_norm
        )
        
        self.D_B = PatchGANDiscriminator(
            in_channels=disc_channels,
            base_filters=disc_filters,
            num_layers=disc_num_layers,
            kernel_size=disc_kernel_size,
            use_batch_norm=use_batch_norm
        )
    
    def forward(self, real_A=None, real_B=None, mode='full'):
        """
        Forward pass through the model.
        
        Args:
            real_A (torch.Tensor): Real images from domain A (clean)
            real_B (torch.Tensor): Real images from domain B (noisy)
            mode (str): Forward mode
                - 'full': Full cycle (requires both real_A and real_B)
                - 'restore': Only B->A restoration (requires real_B)
                - 'degrade': Only A->B degradation (requires real_A)
        
        Returns:
            dict: Dictionary containing generated images and reconstructions
        """
        outputs = {}
        
        if mode == 'full' or mode == 'degrade':
            if real_A is not None:
                # Forward cycle: A -> B -> A
                fake_B = self.G_AB(real_A)
                reconstructed_A = self.G_BA(fake_B)
                
                # Identity mapping: B -> B (generator B should not change B images)
                identity_B = self.G_AB(real_B) if real_B is not None else None
                
                outputs.update({
                    'fake_B': fake_B,
                    'reconstructed_A': reconstructed_A,
                    'identity_B': identity_B
                })
        
        if mode == 'full' or mode == 'restore':
            if real_B is not None:
                # Backward cycle: B -> A -> B (RESTORATION)
                fake_A = self.G_BA(real_B)
                reconstructed_B = self.G_AB(fake_A)
                
                # Identity mapping: A -> A (generator A should not change A images)
                identity_A = self.G_BA(real_A) if real_A is not None else None
                
                outputs.update({
                    'fake_A': fake_A,
                    'reconstructed_B': reconstructed_B,
                    'identity_A': identity_A
                })
        
        return outputs
    
    def restore(self, noisy_image):
        """
        Restore a noisy image (convenience method for inference).
        
        Args:
            noisy_image (torch.Tensor): Noisy image from domain B
        
        Returns:
            torch.Tensor: Restored image (domain A)
        """
        with torch.no_grad():
            restored = self.G_BA(noisy_image)
        return restored
    
    def degrade(self, clean_image):
        """
        Degrade a clean image (convenience method for inference).
        
        Args:
            clean_image (torch.Tensor): Clean image from domain A
        
        Returns:
            torch.Tensor: Degraded image (domain B)
        """
        with torch.no_grad():
            degraded = self.G_AB(clean_image)
        return degraded
    
    def count_parameters(self):
        """Count the number of trainable parameters for each component."""
        return {
            'G_AB': self.G_AB.count_parameters(),
            'G_BA': self.G_BA.count_parameters(),
            'D_A': self.D_A.count_parameters(),
            'D_B': self.D_B.count_parameters(),
            'Total': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def print_model_summary(self):
        """Print a summary of the model architecture."""
        params = self.count_parameters()
        print("\n" + "="*60)
        print("Cycle-CARE Model Summary")
        print("="*60)
        print(f"Generator A->B parameters: {params['G_AB']:,}")
        print(f"Generator B->A parameters: {params['G_BA']:,}")
        print(f"Discriminator A parameters: {params['D_A']:,}")
        print(f"Discriminator B parameters: {params['D_B']:,}")
        print("-"*60)
        print(f"Total parameters: {params['Total']:,}")
        print("="*60 + "\n")


def test_cycle_care():
    """Test the Cycle-CARE model."""
    print("Testing Cycle-CARE Model...")
    
    # Create model
    model = CycleCARE(
        img_channels=1,
        unet_depth=3,
        unet_filters=64,
        disc_filters=64,
        disc_num_layers=3
    )
    
    model.print_model_summary()
    
    # Test forward pass
    batch_size = 2
    img_size = 256
    real_A = torch.randn(batch_size, 1, img_size, img_size)
    real_B = torch.randn(batch_size, 1, img_size, img_size)
    
    print("Testing full forward pass...")
    outputs = model(real_A=real_A, real_B=real_B, mode='full')
    
    print(f"\nInput shapes:")
    print(f"  Real A (clean): {real_A.shape}")
    print(f"  Real B (noisy): {real_B.shape}")
    
    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        if value is not None:
            print(f"  {key}: {value.shape}")
    
    # Test restoration mode
    print("\nTesting restoration mode...")
    restored = model.restore(real_B)
    print(f"  Noisy input: {real_B.shape}")
    print(f"  Restored output: {restored.shape}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_cycle_care()
