"""
3D Cycle-CARE Model for volumetric image restoration.
Combines two 3D generators (G_AB, G_BA) for bidirectional volumetric translation.
"""

import torch
import torch.nn as nn
from models.generator_3d import CAREUNet3D
from models.discriminator_3d import PatchGANDiscriminator3D


class CycleCARE3D(nn.Module):
    """
    3D Cycle-CARE: Volumetric CycleGAN for microscopy restoration.
    
    Uses two 3D U-Net generators and two 3D PatchGAN discriminators
    for unpaired volumetric image translation.
    
    Architecture:
    - G_AB: Clean volumes (A) → Noisy volumes (B)
    - G_BA: Noisy volumes (B) → Clean volumes (A) [DENOISER]
    - D_A: Discriminates clean volumes
    - D_B: Discriminates noisy volumes
    
    The cycle-consistency ensures:
    - A → G_AB → B' → G_BA → A' ≈ A
    - B → G_BA → A' → G_AB → B' ≈ B
    
    Args:
        img_channels (int): Number of image channels (default: 1)
        unet_depth (int): Depth of U-Net generators (default: 3)
        unet_filters (int): Base number of filters in generators (default: 32)
        unet_kernel_size (int): Kernel size for generator convolutions (default: 3)
        disc_filters (int): Base number of filters in discriminators (default: 32)
        disc_num_layers (int): Number of layers in discriminators (default: 3)
        disc_kernel_size (int): Kernel size for discriminator convolutions (default: 4)
        use_batch_norm (bool): Whether to use batch normalization (default: True)
        use_dropout (bool): Whether to use dropout in generators (default: True)
        dropout_rate (float): Dropout probability (default: 0.5)
    """
    def __init__(self, img_channels=1, unet_depth=3, unet_filters=32, unet_kernel_size=3,
                 disc_filters=32, disc_num_layers=3, disc_kernel_size=4,
                 use_batch_norm=True, use_dropout=True, dropout_rate=0.5):
        super(CycleCARE3D, self).__init__()
        
        # Generators (3D U-Nets)
        self.G_AB = CAREUNet3D(
            in_channels=img_channels,
            out_channels=img_channels,
            depth=unet_depth,
            base_filters=unet_filters,
            kernel_size=unet_kernel_size,
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate
        )
        
        self.G_BA = CAREUNet3D(
            in_channels=img_channels,
            out_channels=img_channels,
            depth=unet_depth,
            base_filters=unet_filters,
            kernel_size=unet_kernel_size,
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate
        )
        
        # Discriminators (3D PatchGAN)
        self.D_A = PatchGANDiscriminator3D(
            in_channels=img_channels,
            base_filters=disc_filters,
            num_layers=disc_num_layers,
            kernel_size=disc_kernel_size,
            use_batch_norm=use_batch_norm
        )
        
        self.D_B = PatchGANDiscriminator3D(
            in_channels=img_channels,
            base_filters=disc_filters,
            num_layers=disc_num_layers,
            kernel_size=disc_kernel_size,
            use_batch_norm=use_batch_norm
        )
    
    def forward(self, real_A=None, real_B=None, mode='full', **kwargs):
        """
        Forward pass through the Cycle-CARE 3D model.
        
        Args:
            real_A (torch.Tensor): Real volumes from domain A (clean) [B, 1, D, H, W]
            real_B (torch.Tensor): Real volumes from domain B (noisy) [B, 1, D, H, W]
            mode (str): Forward mode
                - 'full': Complete cycle (for training)
                - 'denoise': Only G_BA (B→A denoising)
                - 'add_noise': Only G_AB (A→B)
        
        Returns:
            dict: Dictionary containing generated volumes
        """
        outputs = {}
        
        if mode == 'full':
            # Full cycle for training
            if real_A is None or real_B is None:
                raise ValueError("Both real_A and real_B required for full mode")
            
            # Generate fake volumes
            fake_B = self.G_AB(real_A)  # Clean → Noisy
            fake_A = self.G_BA(real_B)  # Noisy → Clean (DENOISING)
            
            # Reconstruct (cycle)
            reconstructed_A = self.G_BA(fake_B)  # A → B → A
            reconstructed_B = self.G_AB(fake_A)  # B → A → B
            
            # Identity mapping (only if requested - skipping saves 2 forward passes)
            compute_identity = kwargs.get('compute_identity', True)
            if compute_identity:
                identity_A = self.G_BA(real_A)  # Should preserve clean
                identity_B = self.G_AB(real_B)  # Should preserve noisy
            else:
                identity_A = None
                identity_B = None
            
            outputs = {
                'fake_A': fake_A,
                'fake_B': fake_B,
                'reconstructed_A': reconstructed_A,
                'reconstructed_B': reconstructed_B,
                'identity_A': identity_A,
                'identity_B': identity_B
            }
        
        elif mode == 'denoise':
            # Only denoising (B → A)
            if real_B is None:
                raise ValueError("real_B required for denoise mode")
            
            fake_A = self.G_BA(real_B)
            outputs = {'denoised': fake_A}
        
        elif mode == 'add_noise':
            # Only add noise (A → B)
            if real_A is None:
                raise ValueError("real_A required for add_noise mode")
            
            fake_B = self.G_AB(real_A)
            outputs = {'noisy': fake_B}
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return outputs
    
    def denoise(self, noisy_volume):
        """
        Denoise a volume (convenience method for inference).
        
        Args:
            noisy_volume (torch.Tensor): Noisy volume [B, 1, D, H, W]
        
        Returns:
            torch.Tensor: Denoised volume [B, 1, D, H, W]
        """
        with torch.no_grad():
            return self.G_BA(noisy_volume)
    
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
        print("3D Cycle-CARE Model Summary")
        print("="*60)
        print(f"Generator A->B parameters: {params['G_AB']:,}")
        print(f"Generator B->A parameters: {params['G_BA']:,}")
        print(f"Discriminator A parameters: {params['D_A']:,}")
        print(f"Discriminator B parameters: {params['D_B']:,}")
        print("-"*60)
        print(f"Total parameters: {params['Total']:,}")
        print("="*60 + "\n")


def test_model_3d():
    """Test the 3D Cycle-CARE model."""
    print("Testing 3D Cycle-CARE Model...")
    
    # Create model
    print("\n1. Creating 3D Cycle-CARE model...")
    model = CycleCARE3D(
        img_channels=1,
        unet_depth=2,
        unet_filters=16,
        unet_kernel_size=3,
        disc_filters=16,
        disc_num_layers=2,
        disc_kernel_size=4,
        use_batch_norm=True,
        use_dropout=True,
        dropout_rate=0.5
    )
    model.print_model_summary()
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    batch_size = 1
    depth = 16
    img_size = 32
    
    real_A = torch.randn(batch_size, 1, depth, img_size, img_size)
    real_B = torch.randn(batch_size, 1, depth, img_size, img_size)
    
    print(f"   Input A shape: {real_A.shape}")
    print(f"   Input B shape: {real_B.shape}")
    
    # Full cycle
    outputs = model(real_A=real_A, real_B=real_B, mode='full')
    
    print(f"   fake_A shape: {outputs['fake_A'].shape}")
    print(f"   fake_B shape: {outputs['fake_B'].shape}")
    print(f"   reconstructed_A shape: {outputs['reconstructed_A'].shape}")
    print(f"   reconstructed_B shape: {outputs['reconstructed_B'].shape}")
    
    # Test discriminators
    print("\n4. Testing discriminators...")
    pred_A = model.D_A(outputs['fake_A'])
    pred_B = model.D_B(outputs['fake_B'])
    
    print(f"   D_A prediction shape: {pred_A.shape}")
    print(f"   D_B prediction shape: {pred_B.shape}")
    
    # Test denoising mode
    print("\n5. Testing denoise mode...")
    denoised = model.denoise(real_B)
    print(f"   Denoised shape: {denoised.shape}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_model_3d()
