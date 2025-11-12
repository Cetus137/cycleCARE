"""
CARE-style 3D U-Net Generator for volumetric image restoration.
Implements a symmetric encoder-decoder architecture with skip connections for 3D volumes.
"""

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """
    3D Convolutional block: Conv3D -> BatchNorm3D -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, use_batch_norm=True, use_dropout=False, dropout_rate=0.5):
        super(ConvBlock3D, self).__init__()
        
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_batch_norm)
        ]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm3d(out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        if use_dropout:
            layers.append(nn.Dropout3d(dropout_rate))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class DownsampleBlock3D(nn.Module):
    """
    3D Downsampling block: Double Conv (Conv-ReLU-Conv-ReLU) -> MaxPool3D
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, use_batch_norm=True, 
                 use_dropout=False, dropout_rate=0.5):
        super(DownsampleBlock3D, self).__init__()
        
        # Double convolution
        self.conv1 = ConvBlock3D(in_channels, out_channels, kernel_size, 
                                use_batch_norm=use_batch_norm, 
                                use_dropout=use_dropout, 
                                dropout_rate=dropout_rate)
        self.conv2 = ConvBlock3D(out_channels, out_channels, kernel_size, 
                                use_batch_norm=use_batch_norm, 
                                use_dropout=use_dropout, 
                                dropout_rate=dropout_rate)
        self.pool = nn.MaxPool3d(2, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        skip = self.conv2(x)
        down = self.pool(skip)
        return down, skip


class UpsampleBlock3D(nn.Module):
    """
    3D Upsampling block: ConvTranspose3D -> Concatenate skip -> Double Conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, use_batch_norm=True, 
                 use_dropout=False, dropout_rate=0.5):
        super(UpsampleBlock3D, self).__init__()
        
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concatenation, we have out_channels + in_channels (from skip connection)
        self.conv1 = ConvBlock3D(in_channels + out_channels, out_channels, kernel_size,
                                use_batch_norm=use_batch_norm, 
                                use_dropout=use_dropout, 
                                dropout_rate=dropout_rate)
        self.conv2 = ConvBlock3D(out_channels, out_channels, kernel_size,
                                use_batch_norm=use_batch_norm, 
                                use_dropout=use_dropout, 
                                dropout_rate=dropout_rate)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        
        # Handle size mismatch due to pooling/upsampling
        if x.size() != skip.size():
            diff_d = skip.size(2) - x.size(2)
            diff_h = skip.size(3) - x.size(3)
            diff_w = skip.size(4) - x.size(4)
            x = nn.functional.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                                     diff_h // 2, diff_h - diff_h // 2,
                                     diff_d // 2, diff_d - diff_d // 2])
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Double convolution
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CAREUNet3D(nn.Module):
    """
    CARE-style 3D U-Net Generator for volumetric image restoration.
    
    Extends the 2D CARE U-Net to 3D for true volumetric processing of microscopy Z-stacks.
    
    Architecture:
    - Initial: Double 3D convolution (Conv3D-ReLU-Conv3D-ReLU)
    - Encoder: Series of double 3D convolution blocks with MaxPooling3D
    - Bottleneck: Double 3D convolution at lowest resolution
    - Decoder: Series of upsampling + skip connection + double 3D convolution
    - Output: Final 3D convolution (1x1x1) with Tanh activation
    
    Key features:
    - True 3D convolutions learn volumetric features
    - Skip connections preserve spatial information across scales
    - Processes entire volumes maintaining 3D coherence
    - Compatible with CycleGAN framework (Tanh output)
    
    Args:
        in_channels (int): Number of input channels (typically 1 for grayscale microscopy)
        out_channels (int): Number of output channels (typically 1)
        depth (int): Depth of the U-Net (number of downsampling/upsampling layers)
        base_filters (int): Number of filters in the first layer (doubles with each layer)
        kernel_size (int): Kernel size for convolutions (default: 3)
        use_batch_norm (bool): Whether to use batch normalization
        use_dropout (bool): Whether to use dropout
        dropout_rate (float): Dropout probability
    
    Input shape: (B, C, D, H, W) where:
        B = batch size
        C = channels (1 for grayscale)
        D = depth (number of Z-planes)
        H = height
        W = width
    
    Output shape: Same as input shape
    """
    def __init__(self, in_channels=1, out_channels=1, depth=3, base_filters=32, 
                 kernel_size=3, use_batch_norm=True, use_dropout=True, dropout_rate=0.5):
        super(CAREUNet3D, self).__init__()
        
        self.depth = depth
        self.base_filters = base_filters
        
        # Initial double convolution
        self.initial_conv = nn.Sequential(
            ConvBlock3D(in_channels, base_filters, kernel_size, use_batch_norm=use_batch_norm),
            ConvBlock3D(base_filters, base_filters, kernel_size, use_batch_norm=use_batch_norm)
        )
        
        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        for i in range(depth):
            in_ch = base_filters * (2 ** i)
            out_ch = base_filters * (2 ** (i + 1))
            # Use dropout in the deeper layers
            use_drop = use_dropout and (i >= depth - 2)
            self.encoder_blocks.append(
                DownsampleBlock3D(in_ch, out_ch, kernel_size, use_batch_norm, use_drop, dropout_rate)
            )
        
        # Bottleneck
        bottleneck_channels = base_filters * (2 ** depth)
        self.bottleneck = nn.Sequential(
            ConvBlock3D(bottleneck_channels, bottleneck_channels, kernel_size, 
                       use_batch_norm=use_batch_norm, use_dropout=use_dropout, dropout_rate=dropout_rate),
            ConvBlock3D(bottleneck_channels, bottleneck_channels, kernel_size, 
                       use_batch_norm=use_batch_norm, use_dropout=use_dropout, dropout_rate=dropout_rate)
        )
        
        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        for i in range(depth):
            in_ch = base_filters * (2 ** (depth - i))
            out_ch = base_filters * (2 ** (depth - i - 1))
            # Use dropout in the deeper layers
            use_drop = use_dropout and (i < 2)
            self.decoder_blocks.append(
                UpsampleBlock3D(in_ch, out_ch, kernel_size, use_batch_norm, use_drop, dropout_rate)
            )
        
        # Final convolution (1x1x1 conv for output projection)
        self.final_conv = nn.Sequential(
            ConvBlock3D(base_filters, base_filters, kernel_size, use_batch_norm=False),
            nn.Conv3d(base_filters, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Tanh()  # Output in range [-1, 1] for CycleGAN training
        )
    
    def forward(self, x):
        """
        Forward pass through the 3D U-Net.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, C, D, H, W)
        """
        # Initial convolution
        x = self.initial_conv(x)
        
        # Encoder with skip connections
        skips = []
        for encoder in self.encoder_blocks:
            x, skip = encoder(x)
            skips.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        for i, decoder in enumerate(self.decoder_blocks):
            skip = skips[-(i + 1)]  # Get skip connection in reverse order
            x = decoder(x, skip)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x
    
    def count_parameters(self):
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_generator_3d():
    """Test the 3D generator architecture."""
    print("Testing CARE 3D U-Net Generator...")
    
    # Test with different configurations
    configs = [
        {"in_channels": 1, "out_channels": 1, "depth": 2, "base_filters": 16},
        {"in_channels": 1, "out_channels": 1, "depth": 3, "base_filters": 32},
    ]
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}:")
        print(f"  Input channels: {config['in_channels']}")
        print(f"  Output channels: {config['out_channels']}")
        print(f"  Depth: {config['depth']}")
        print(f"  Base filters: {config['base_filters']}")
        
        model = CAREUNet3D(**config)
        print(f"  Parameters: {model.count_parameters():,}")
        
        # Test forward pass
        batch_size = 1
        depth = 16
        img_size = 64
        x = torch.randn(batch_size, config['in_channels'], depth, img_size, img_size)
        y = model(x)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
        assert y.shape == x.shape, "Output shape should match input shape"
        print("  ✓ Test passed!")


if __name__ == "__main__":
    test_generator_3d()
