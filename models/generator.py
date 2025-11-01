"""
CARE-style U-Net Generator for image restoration.
Implements a symmetric encoder-decoder architecture with skip connections.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Convolutional block: Conv -> BatchNorm -> ReLU
    CARE uses ReLU activation (not LeakyReLU) and typically no batch norm.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, use_batch_norm=True, use_dropout=False, dropout_rate=0.5):
        super(ConvBlock, self).__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_batch_norm)
        ]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        # CARE uses ReLU, not LeakyReLU
        layers.append(nn.ReLU(inplace=True))
        
        if use_dropout:
            layers.append(nn.Dropout2d(dropout_rate))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class DownsampleBlock(nn.Module):
    """
    Downsampling block: Double Conv (Conv-ReLU-Conv-ReLU) -> MaxPool
    CARE uses double convolutions at each level for better feature extraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, use_batch_norm=True, 
                 use_dropout=False, dropout_rate=0.5):
        super(DownsampleBlock, self).__init__()
        
        # Double convolution as in CARE
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, 
                              use_batch_norm=use_batch_norm, 
                              use_dropout=use_dropout, 
                              dropout_rate=dropout_rate)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, 
                              use_batch_norm=use_batch_norm, 
                              use_dropout=use_dropout, 
                              dropout_rate=dropout_rate)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        skip = self.conv2(x)
        down = self.pool(skip)
        return down, skip


class UpsampleBlock(nn.Module):
    """
    Upsampling block: ConvTranspose -> Concatenate skip -> Double Conv (Conv-ReLU-Conv-ReLU)
    CARE uses double convolutions after concatenation for feature refinement.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, use_batch_norm=True, 
                 use_dropout=False, dropout_rate=0.5):
        super(UpsampleBlock, self).__init__()
        
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concatenation, we have out_channels + in_channels (from skip connection)
        # Skip connection has same channels as input to decoder block
        self.conv1 = ConvBlock(in_channels + out_channels, out_channels, kernel_size,
                              use_batch_norm=use_batch_norm, 
                              use_dropout=use_dropout, 
                              dropout_rate=dropout_rate)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size,
                              use_batch_norm=use_batch_norm, 
                              use_dropout=use_dropout, 
                              dropout_rate=dropout_rate)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        
        # Handle size mismatch due to pooling/upsampling
        if x.size() != skip.size():
            diff_h = skip.size(2) - x.size(2)
            diff_w = skip.size(3) - x.size(3)
            x = nn.functional.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                                     diff_h // 2, diff_h - diff_h // 2])
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Double convolution
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CAREUNet(nn.Module):
    """
    CARE-style U-Net Generator for image restoration.
    
    Based on "Content-aware image restoration: pushing the limits of fluorescence 
    microscopy" (Weigert et al., Nature Methods 2018).
    
    Architecture (following CARE design principles):
    - Initial: Double convolution (Conv-ReLU-Conv-ReLU)
    - Encoder: Series of double convolution blocks with MaxPooling
    - Bottleneck: Double convolution at lowest resolution
    - Decoder: Series of upsampling + skip connection + double convolution
    - Output: Final convolution (1x1) with Tanh activation for CycleGAN
    
    Key CARE features:
    - Double convolutions at each level for better feature extraction
    - ReLU activation (original CARE uses ReLU, not LeakyReLU)
    - Skip connections from encoder to decoder (U-Net structure)
    - Progressive channel doubling in encoder, halving in decoder
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        depth (int): Depth of the U-Net (number of downsampling/upsampling layers)
        base_filters (int): Number of filters in the first layer (doubles with each layer)
        kernel_size (int): Kernel size for convolutions (default: 3)
        use_batch_norm (bool): Whether to use batch normalization (not in original CARE)
        use_dropout (bool): Whether to use dropout
        dropout_rate (float): Dropout probability
    """
    def __init__(self, in_channels=1, out_channels=1, depth=3, base_filters=64, 
                 kernel_size=3, use_batch_norm=True, use_dropout=True, dropout_rate=0.5):
        super(CAREUNet, self).__init__()
        
        self.depth = depth
        self.base_filters = base_filters
        
        # Initial double convolution (CARE-style)
        self.initial_conv = nn.Sequential(
            ConvBlock(in_channels, base_filters, kernel_size, use_batch_norm=use_batch_norm),
            ConvBlock(base_filters, base_filters, kernel_size, use_batch_norm=use_batch_norm)
        )
        
        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        for i in range(depth):
            in_ch = base_filters * (2 ** i)
            out_ch = base_filters * (2 ** (i + 1))
            # Use dropout in the deeper layers
            use_drop = use_dropout and (i >= depth - 2)
            self.encoder_blocks.append(
                DownsampleBlock(in_ch, out_ch, kernel_size, use_batch_norm, use_drop, dropout_rate)
            )
        
        # Bottleneck
        bottleneck_channels = base_filters * (2 ** depth)
        self.bottleneck = nn.Sequential(
            ConvBlock(bottleneck_channels, bottleneck_channels, kernel_size, 
                     use_batch_norm=use_batch_norm, use_dropout=use_dropout, dropout_rate=dropout_rate),
            ConvBlock(bottleneck_channels, bottleneck_channels, kernel_size, 
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
                UpsampleBlock(in_ch, out_ch, kernel_size, use_batch_norm, use_drop, dropout_rate)
            )
        
        # Final convolution (CARE uses 1x1 conv for output projection)
        # Tanh is added for CycleGAN compatibility (output range [-1, 1])
        self.final_conv = nn.Sequential(
            ConvBlock(base_filters, base_filters, kernel_size, use_batch_norm=False),
            nn.Conv2d(base_filters, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Tanh()  # Output in range [-1, 1] for CycleGAN training
        )
    
    def forward(self, x):
        """
        Forward pass through the U-Net.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W)
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


def test_generator():
    """Test the generator architecture."""
    print("Testing CARE U-Net Generator...")
    
    # Test with different configurations
    configs = [
        {"in_channels": 1, "out_channels": 1, "depth": 3, "base_filters": 64},
        {"in_channels": 3, "out_channels": 3, "depth": 4, "base_filters": 32},
    ]
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}:")
        print(f"  Input channels: {config['in_channels']}")
        print(f"  Output channels: {config['out_channels']}")
        print(f"  Depth: {config['depth']}")
        print(f"  Base filters: {config['base_filters']}")
        
        model = CAREUNet(**config)
        print(f"  Parameters: {model.count_parameters():,}")
        
        # Test forward pass
        batch_size = 2
        img_size = 256
        x = torch.randn(batch_size, config['in_channels'], img_size, img_size)
        y = model(x)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
        assert y.shape == x.shape, "Output shape should match input shape"
        print("  ✓ Test passed!")


if __name__ == "__main__":
    test_generator()
