"""
3D PatchGAN Discriminator for adversarial training on volumetric data.
Outputs a 3D matrix of predictions rather than a single value.
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator3D(nn.Module):
    """
    3D PatchGAN Discriminator.
    
    Classifies whether overlapping volumetric patches are real or fake.
    The discriminator outputs a 3D matrix where each element corresponds to
    the classification of a receptive field patch in the input volume.
    
    This architecture extends the 2D PatchGAN to 3D for volumetric microscopy data,
    focusing on 3D texture and structure rather than overall volume characteristics.
    
    Args:
        in_channels (int): Number of input channels (typically 1 for grayscale)
        base_filters (int): Number of filters in the first layer
        num_layers (int): Number of discriminator layers (depth)
        kernel_size (int): Kernel size for convolutions
        use_batch_norm (bool): Whether to use batch normalization
    
    Input shape: (B, C, D, H, W) where:
        B = batch size
        C = channels (1 for grayscale)
        D = depth (number of Z-planes)
        H = height
        W = width
    
    Output shape: (B, 1, D', H', W') - 3D patch prediction map
    """
    def __init__(self, in_channels=1, base_filters=64, num_layers=3, 
                 kernel_size=4, use_batch_norm=True):
        super(PatchGANDiscriminator3D, self).__init__()
        
        self.num_layers = num_layers
        
        # Build discriminator layers
        layers = []
        
        # First layer (no batch norm)
        layers.append(nn.Conv3d(in_channels, base_filters, kernel_size, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Intermediate layers
        nf_mult = 1
        for n in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)  # Cap at 8x base filters
            
            layers.append(nn.Conv3d(base_filters * nf_mult_prev, 
                                   base_filters * nf_mult,
                                   kernel_size, stride=2, padding=1, 
                                   bias=not use_batch_norm))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm3d(base_filters * nf_mult))
            
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Penultimate layer (stride=1 to preserve spatial dimensions)
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** num_layers, 8)
        
        layers.append(nn.Conv3d(base_filters * nf_mult_prev,
                               base_filters * nf_mult,
                               kernel_size, stride=1, padding=1,
                               bias=not use_batch_norm))
        
        if use_batch_norm:
            layers.append(nn.BatchNorm3d(base_filters * nf_mult))
        
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Final layer (output single channel prediction map)
        layers.append(nn.Conv3d(base_filters * nf_mult, 1, kernel_size, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the 3D discriminator.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, 1, D', H', W') where D', H', W'
                         depend on the input size and number of layers.
                         Each element represents the classification for a 3D patch.
        """
        return self.model(x)
    
    def count_parameters(self):
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiScaleDiscriminator3D(nn.Module):
    """
    Multi-scale 3D discriminator that operates at multiple volume resolutions.
    
    Helps capture both fine and coarse 3D details by discriminating at different scales.
    
    Args:
        in_channels (int): Number of input channels
        base_filters (int): Number of filters in the first layer
        num_layers (int): Number of discriminator layers
        num_scales (int): Number of different scales
        kernel_size (int): Kernel size for convolutions
        use_batch_norm (bool): Whether to use batch normalization
    """
    def __init__(self, in_channels=1, base_filters=64, num_layers=3, 
                 num_scales=2, kernel_size=4, use_batch_norm=True):
        super(MultiScaleDiscriminator3D, self).__init__()
        
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()
        
        for _ in range(num_scales):
            self.discriminators.append(
                PatchGANDiscriminator3D(in_channels, base_filters, num_layers, 
                                       kernel_size, use_batch_norm)
            )
        
        # Downsampling for multi-scale
        self.downsample = nn.AvgPool3d(3, stride=2, padding=1)
    
    def forward(self, x):
        """
        Forward pass through all discriminators at different scales.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W)
        
        Returns:
            list: List of output tensors, one for each scale
        """
        outputs = []
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            outputs.append(disc(x))
        return outputs
    
    def count_parameters(self):
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_discriminator_3d():
    """Test the 3D discriminator architectures."""
    print("Testing 3D PatchGAN Discriminator...")
    
    # Test single-scale discriminator
    print("\n1. Single-scale 3D PatchGAN:")
    disc = PatchGANDiscriminator3D(in_channels=1, base_filters=32, num_layers=3)
    print(f"   Parameters: {disc.count_parameters():,}")
    
    # Test forward pass
    batch_size = 1
    depth = 16
    img_size = 64
    x = torch.randn(batch_size, 1, depth, img_size, img_size)
    y = disc(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Output is a {y.shape[2]}x{y.shape[3]}x{y.shape[4]} 3D patch prediction map")
    print("   ✓ Test passed!")
    
    # Test multi-scale discriminator
    print("\n2. Multi-scale 3D PatchGAN:")
    multi_disc = MultiScaleDiscriminator3D(in_channels=1, base_filters=32, 
                                          num_layers=3, num_scales=2)
    print(f"   Parameters: {multi_disc.count_parameters():,}")
    
    outputs = multi_disc(x)
    print(f"   Input shape: {x.shape}")
    for i, out in enumerate(outputs):
        print(f"   Scale {i+1} output shape: {out.shape}")
    print("   ✓ Test passed!")


if __name__ == "__main__":
    test_discriminator_3d()
