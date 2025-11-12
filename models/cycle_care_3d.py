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
        config: Configuration object with model parameters
    """
    def __init__(self, config):
        super(CycleCARE3D, self).__init__()
        
        self.config = config
        
        # Generators (3D U-Nets)
        self.G_AB = CAREUNet3D(
            in_channels=config.IMG_CHANNELS,
            out_channels=config.IMG_CHANNELS,
            depth=config.UNET_DEPTH,
            base_filters=config.UNET_FILTERS,
            kernel_size=config.UNET_KERNEL_SIZE,
            use_batch_norm=config.USE_BATCH_NORM,
            use_dropout=config.USE_DROPOUT,
            dropout_rate=config.DROPOUT_RATE
        )
        
        self.G_BA = CAREUNet3D(
            in_channels=config.IMG_CHANNELS,
            out_channels=config.IMG_CHANNELS,
            depth=config.UNET_DEPTH,
            base_filters=config.UNET_FILTERS,
            kernel_size=config.UNET_KERNEL_SIZE,
            use_batch_norm=config.USE_BATCH_NORM,
            use_dropout=config.USE_DROPOUT,
            dropout_rate=config.DROPOUT_RATE
        )
        
        print(f"Generator G_AB parameters: {self.G_AB.count_parameters():,}")
        print(f"Generator G_BA parameters: {self.G_BA.count_parameters():,}")
    
    def forward(self, real_A=None, real_B=None, mode='full'):
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
            
            # Identity mapping
            identity_A = self.G_BA(real_A)  # Should preserve clean
            identity_B = self.G_AB(real_B)  # Should preserve noisy
            
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
        """Count total trainable parameters in both generators."""
        return self.G_AB.count_parameters() + self.G_BA.count_parameters()


def create_discriminators_3d(config):
    """
    Create discriminators for 3D Cycle-CARE.
    
    Args:
        config: Configuration object
    
    Returns:
        tuple: (D_A, D_B) discriminators
    """
    D_A = PatchGANDiscriminator3D(
        in_channels=config.IMG_CHANNELS,
        base_filters=config.DISC_FILTERS,
        num_layers=config.DISC_NUM_LAYERS,
        kernel_size=config.DISC_KERNEL_SIZE,
        use_batch_norm=config.USE_BATCH_NORM
    )
    
    D_B = PatchGANDiscriminator3D(
        in_channels=config.IMG_CHANNELS,
        base_filters=config.DISC_FILTERS,
        num_layers=config.DISC_NUM_LAYERS,
        kernel_size=config.DISC_KERNEL_SIZE,
        use_batch_norm=config.USE_BATCH_NORM
    )
    
    print(f"Discriminator D_A parameters: {D_A.count_parameters():,}")
    print(f"Discriminator D_B parameters: {D_B.count_parameters():,}")
    
    return D_A, D_B


def test_model_3d():
    """Test the 3D Cycle-CARE model."""
    print("Testing 3D Cycle-CARE Model...")
    
    # Create a mock config
    class MockConfig:
        IMG_CHANNELS = 1
        UNET_DEPTH = 2
        UNET_FILTERS = 16
        UNET_KERNEL_SIZE = 3
        USE_BATCH_NORM = True
        USE_DROPOUT = True
        DROPOUT_RATE = 0.5
        DISC_FILTERS = 16
        DISC_NUM_LAYERS = 2
        DISC_KERNEL_SIZE = 4
    
    config = MockConfig()
    
    # Create model
    print("\n1. Creating 3D Cycle-CARE model...")
    model = CycleCARE3D(config)
    print(f"   Total generator parameters: {model.count_parameters():,}")
    
    # Create discriminators
    print("\n2. Creating 3D discriminators...")
    D_A, D_B = create_discriminators_3d(config)
    
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
    pred_A = D_A(outputs['fake_A'])
    pred_B = D_B(outputs['fake_B'])
    
    print(f"   D_A prediction shape: {pred_A.shape}")
    print(f"   D_B prediction shape: {pred_B.shape}")
    
    # Test denoising mode
    print("\n4. Testing denoise mode...")
    denoised = model.denoise(real_B)
    print(f"   Denoised shape: {denoised.shape}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_model_3d()
