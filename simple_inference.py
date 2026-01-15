"""
Simple inference script for Cycle-CARE with tiling and Z-stack support.
Automatically tiles large images (>128x128) and processes Z-stacks plane-by-plane.
"""

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
import tifffile
from tqdm import tqdm


def load_model(checkpoint_path, device='cuda'):
    """
    Load the trained model from checkpoint by inferring architecture from weights.
    
    Returns:
        tuple: (model, zstack_context)
            - model: Loaded CycleCARE model
            - zstack_context: Number of Z-planes used (1 for single-plane, 3/5/7 for multi-plane)
    """
    from models import CycleCARE
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    print("Inferring model architecture from checkpoint weights...")
    
    # 1. Infer img_channels (Z-stack context) from first conv layer
    img_channels = 1
    for key in state_dict.keys():
        if 'G_AB.initial_conv.0.block.0.weight' in key:
            # Shape: [out_channels, in_channels, kernel_h, kernel_w]
            img_channels = state_dict[key].shape[1]
            break
    
    # 2. Infer unet_depth from encoder blocks
    max_encoder_idx = -1
    for key in state_dict.keys():
        if 'encoder_blocks.' in key:
            idx = int(key.split('encoder_blocks.')[1].split('.')[0])
            max_encoder_idx = max(max_encoder_idx, idx)
    unet_depth = max_encoder_idx + 1 if max_encoder_idx >= 0 else 2
    
    # 3. Infer unet_filters from bottleneck
    unet_filters = 64  # default
    for key in state_dict.keys():
        if 'bottleneck.0.block.0.weight' in key:
            bottleneck_channels = state_dict[key].shape[0]
            unet_filters = bottleneck_channels // (2 ** unet_depth)
            break
    
    zstack_context = img_channels
    
    print(f"  UNET_DEPTH: {unet_depth}")
    print(f"  UNET_FILTERS: {unet_filters}")
    print(f"  IMG_CHANNELS: {img_channels}")
    print(f"  ZSTACK_CONTEXT: {zstack_context}")
    
    # Build and load model
    model = CycleCARE(
        img_channels=img_channels,
        unet_depth=unet_depth,
        unet_filters=unet_filters,
        unet_kernel_size=3,
        disc_filters=64,
        disc_num_layers=3,
        disc_kernel_size=4,
        use_batch_norm=True,
        use_dropout=True,
        dropout_rate=0.5
    ).to(device)
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"✓ Model loaded from {checkpoint_path}")
    return model, zstack_context


def load_noise_model(checkpoint_path, device='cuda'):
    """
    Load the noise model (G_BA) from checkpoint by inferring architecture from weights.

    This mirrors load_model but looks for G_BA keys when inferring the input channels
    (useful if the two generators were trained with different input contexts).

    Returns:
        tuple: (model, zstack_context)
            - model: Loaded CycleCARE model (with weights loaded)
            - zstack_context: Number of Z-planes used (1 for single-plane, 3/5/7 for multi-plane)
    """
    from models import CycleCARE

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    print("Inferring noise-model (G_BA) architecture from checkpoint weights...")

    # 1. Infer img_channels (Z-stack context) from first conv layer of G_BA
    img_channels = 1
    for key in state_dict.keys():
        if 'G_BA.initial_conv.0.block.0.weight' in key:
            # Shape: [out_channels, in_channels, kernel_h, kernel_w]
            img_channels = state_dict[key].shape[1]
            break

    # 2. Infer unet_depth from encoder blocks (shared naming)
    max_encoder_idx = -1
    for key in state_dict.keys():
        if 'encoder_blocks.' in key:
            idx = int(key.split('encoder_blocks.')[1].split('.')[0])
            max_encoder_idx = max(max_encoder_idx, idx)
    unet_depth = max_encoder_idx + 1 if max_encoder_idx >= 0 else 2

    # 3. Infer unet_filters from bottleneck
    unet_filters = 64  # default
    for key in state_dict.keys():
        if 'bottleneck.0.block.0.weight' in key:
            bottleneck_channels = state_dict[key].shape[0]
            unet_filters = bottleneck_channels // (2 ** unet_depth)
            break

    zstack_context = img_channels

    print(f"  UNET_DEPTH: {unet_depth}")
    print(f"  UNET_FILTERS: {unet_filters}")
    print(f"  IMG_CHANNELS: {img_channels}")
    print(f"  ZSTACK_CONTEXT: {zstack_context}")

    # Build and load model
    model = CycleCARE(
        img_channels=img_channels,
        unet_depth=unet_depth,
        unet_filters=unet_filters,
        unet_kernel_size=3,
        disc_filters=64,
        disc_num_layers=3,
        disc_kernel_size=4,
        use_batch_norm=True,
        use_dropout=True,
        dropout_rate=0.5
    ).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"✓ Noise model (G_BA) loaded from {checkpoint_path}")
    return model, zstack_context


def preprocess_image(image_array, use_percentile_norm=True, percentile_low=0.0, percentile_high=99.0, 
                     norm_min=None, norm_max=None):
    """
    Convert image to tensor and normalize.
    Uses percentile-based normalization for fluorescence microscopy.
    
    Args:
        image_array: Input image as numpy array
        use_percentile_norm: If True, normalize to percentile instead of min/max (recommended for fluorescence)
        percentile_low: Lower percentile for normalization (default 0.0)
        percentile_high: Upper percentile for normalization (default 99.0)
        norm_min: Pre-computed minimum value for normalization (overrides percentile_low if provided)
        norm_max: Pre-computed maximum value for normalization (overrides percentile_high if provided)
    
    Returns:
        Normalized tensor in [-1, 1] range
    """
    # Convert to float32 first
    if image_array.dtype == np.uint16:
        # 16-bit TIFF - convert to float
        image_array = image_array.astype(np.float32)
    elif image_array.dtype == np.uint8:
        # 8-bit image - convert to float
        image_array = image_array.astype(np.float32)
    elif image_array.dtype not in [np.float32, np.float64]:
        # Unknown dtype - convert to float
        image_array = image_array.astype(np.float32)
    else:
        image_array = image_array.astype(np.float32)
    
    # Normalize to [0, 1] range using percentile-based normalization
    if norm_min is not None and norm_max is not None:
        # Use pre-computed normalization parameters (for global Z-stack normalization)
        p_min = norm_min
        p_max = norm_max
    elif use_percentile_norm:
        # Percentile-based normalization (robust to outliers in fluorescence)
        p_min = np.percentile(image_array, percentile_low)  # Lower percentile (default 1%)
        p_max = np.percentile(image_array, percentile_high)  # Upper percentile (default 99%)
    else:
        # Simple min-max normalization (legacy behavior)
        p_min = image_array.min()
        p_max = image_array.max()
    
    # Avoid division by zero
    if p_max - p_min < 1e-8:
        print(f"  Warning: Very low contrast image (min={p_min:.3f}, max={p_max:.3f})")
        image_array = np.zeros_like(image_array)
    else:
        # Normalize to [0, 1] based on min/max
        image_array = (image_array - p_min) / (p_max - p_min)
        # Clip to [0, 1] (values outside range get clipped)
        image_array = np.clip(image_array, 0.0, 1.0)
    
    # Convert to tensor
    tensor = torch.from_numpy(image_array).float()
    
    # Handle different input shapes
    if len(tensor.shape) == 2:
        # Grayscale (H, W) -> add channel dimension
        tensor = tensor.unsqueeze(0)  # -> (1, H, W)
    elif len(tensor.shape) == 3:
        # Multi-channel, already in (C, H, W) format - keep as is
        pass
    
    # Normalize to [-1, 1] using same parameters as training
    # Formula: (x - mean) / std where mean=0.5, std=0.5
    tensor = (tensor - 0.5) / 0.5
    
    return tensor


def compute_normalization_params(image_array, use_percentile_norm=True, percentile_low=0.0, percentile_high=99.0):
    """Compute and return normalization parameters for an image/plane.

    Returns a dict with keys: p_min, p_max, orig_dtype
    """
    orig_dtype = image_array.dtype
    arr = image_array.astype(np.float32)
    if use_percentile_norm:
        p_min = float(np.percentile(arr, percentile_low))
        p_max = float(np.percentile(arr, percentile_high))
    else:
        p_min = float(arr.min())
        p_max = float(arr.max())

    return {'p_min': p_min, 'p_max': p_max, 'orig_dtype': orig_dtype}


def normalize_to_model_tensor(image_array, params):
    """Normalize numpy image using params and return a torch tensor in [-1,1].

    Input image_array should be a 2D array (H,W).
    """
    p_min = params['p_min']
    p_max = params['p_max']
    if p_max - p_min < 1e-8:
        norm = np.zeros_like(image_array, dtype=np.float32)
    else:
        norm = (image_array.astype(np.float32) - p_min) / (p_max - p_min)
        norm = np.clip(norm, 0.0, 1.0)

    tensor = torch.from_numpy(norm).float()
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    # map to [-1, 1]
    tensor = (tensor - 0.5) / 0.5
    return tensor


def denormalize_from_model_array(arr_0_1, params):
    """Convert an array in [0,1] back to the original input value range.

    Returns a float32 numpy array in the original measurement units.
    """
    p_min = params['p_min']
    p_max = params['p_max']
    denorm = arr_0_1.astype(np.float32) * (p_max - p_min) + p_min
    return denorm


def postprocess_image(tensor):
    """Convert tensor back to numpy image as float32 in [0, 1] range."""
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor * 0.5) + 0.5
    tensor = torch.clamp(tensor, 0, 1)
    
    # If multi-channel output (shouldn't happen, but just in case), take first channel
    if tensor.dim() == 3 and tensor.size(0) > 1:
        tensor = tensor[0:1]
    
    # Convert to numpy as float32, ensure 2D output
    image_array = tensor.squeeze().cpu().numpy().astype(np.float32)
    
    # Ensure 2D output
    if image_array.ndim > 2:
        image_array = image_array[0] if image_array.shape[0] == 1 else image_array
    
    return image_array


def tile_image(image_array, tile_size=128, overlap=16):
    """
    Split image into overlapping tiles. Assumes channel-first for 3D arrays
    (C, H, W). For 2D inputs (H, W) a singleton channel is added.

    Tiles are returned in channel-first format: (C, tile_size, tile_size).

    Args:
        image_array: Input image (H, W) or (C, H, W)
        tile_size: Size of each tile (default 128)
        overlap: Overlap between tiles in pixels (default 16)

    Returns:
        tiles: List of tile arrays (each is (C, tile_size, tile_size) or (1, tile_size, tile_size))
        positions: List of (y, x, y_end, x_end) positions for each tile
        original_shape: Original spatial image shape (h, w)
    """
    # Normalize to channels-first layout
    if image_array.ndim == 2:
        # Grayscale (H, W)
        h, w = image_array.shape
        arr_cf = image_array[np.newaxis, :, :]
        c = 1
    elif image_array.ndim == 3:
        # Assume (C, H, W)
        c, h, w = image_array.shape
        arr_cf = image_array
    else:
        raise ValueError(f"Unsupported image_array ndim: {image_array.ndim}. Expected 2 or 3.")

    stride = tile_size - overlap
    tiles = []
    positions = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)

            # Extract tile in channels-first layout
            tile = arr_cf[:, y:y_end, x:x_end]

            # Pad if necessary to reach tile_size
            tile_c, tile_h, tile_w = tile.shape
            if tile_h < tile_size or tile_w < tile_size:
                padded = np.zeros((tile_c, tile_size, tile_size), dtype=image_array.dtype)
                padded[:, :tile_h, :tile_w] = tile
                tile = padded

            tiles.append(tile)
            positions.append((y, x, y_end, x_end))

    return tiles, positions, (h, w)


def stitch_tiles(tiles, positions, original_shape, overlap=16):
    """
    Stitch tiles back together with blending in overlap regions.
    
    Args:
        tiles: List of processed tile arrays
        positions: List of (y, x, y_end, x_end) positions
        original_shape: Original image shape (h, w)
        overlap: Overlap size used in tiling
    
    Returns:
        Stitched image array
    """
    h, w = original_shape
    output = np.zeros((h, w), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)
    
    for tile, (y, x, y_end, x_end) in zip(tiles, positions):
        # Calculate actual tile size (may be smaller at edges)
        tile_h = y_end - y
        tile_w = x_end - x
        
        # Extract valid portion of tile
        tile_crop = tile[:tile_h, :tile_w]
        
        # Create weight map (lower weight at edges for blending)
        tile_weights = np.ones((tile_h, tile_w), dtype=np.float32)
        
        # Apply soft edges in overlap regions
        if overlap > 0:
            # Top edge
            if y > 0:
                tile_weights[:overlap, :] *= np.linspace(0, 1, overlap)[:, np.newaxis]
            # Bottom edge
            if y_end < h:
                tile_weights[-overlap:, :] *= np.linspace(1, 0, overlap)[:, np.newaxis]
            # Left edge
            if x > 0:
                tile_weights[:, :overlap] *= np.linspace(0, 1, overlap)
            # Right edge
            if x_end < w:
                tile_weights[:, -overlap:] *= np.linspace(1, 0, overlap)
        
        # Add to output with weights
        output[y:y_end, x:x_end] += tile_crop * tile_weights
        weights[y:y_end, x:x_end] += tile_weights
    
    # Normalize by weights (avoid division by zero)
    weights = np.maximum(weights, 1e-8)
    output = output / weights
    
    return output.astype(np.float32)


def restore_image(model, image, device='cuda', tile_size=128, zstack_context=1, 
                  use_percentile_norm=True, percentile_low=1.0, percentile_high=99.0):
    """
    Restore a single image with automatic tiling for large images.
    
    Args:
        model: Trained CycleCARE model
        image_path: Path to input noisy image
        output_path: Path to save restored image (TIFF preserves float32, PNG/JPG converts to uint8)
        device: Device to use ('cuda' or 'cpu')
        tile_size: Tile size (default 128 to match training)
        zstack_context: Number of channels model expects (1 for single-plane, 5 for Z-stack)
        use_percentile_norm: Use percentile-based normalization (recommended for fluorescence)
        percentile_low: Lower percentile for normalization (default 1.0)
        percentile_high: Upper percentile for normalization (default 99.0)
    
    Returns:
        Restored image as float32 numpy array in [0, 1] range
    """
    

    # Handle different input shapes
    if len(image.shape) == 3:
        # Could be (H, W, C) or (C, H, W) - TIFF Z-stacks are usually (Z, H, W)
        if image.shape[0] == zstack_context:
            C, Y, X = image.shape
            print(f"  Image size: {X}×{Y}, {C} channels")
    
    # Check if tiling is needed
    if Y <= tile_size and X <= tile_size:
        print(f"  Using direct processing (image ≤ {tile_size}×{tile_size})")
        
        # Preprocess with percentile normalization
        tensor = preprocess_image(image, use_percentile_norm=use_percentile_norm, 
                                 percentile_low=percentile_low, percentile_high=percentile_high)
        
        # Handle Z-stack context: replicate single channel if needed
        if tensor.shape[0] == 1 and zstack_context > 1:
            # Single channel but model expects multi-channel - replicate
            tensor = tensor.repeat(zstack_context, 1, 1)
            print(f"  Replicated single channel to {zstack_context} channels for Z-stack model")
        elif tensor.shape[0] != zstack_context:
            print(f"  Warning: Tensor has {tensor.shape[0]} channels but model expects {zstack_context}")
        
        tensor = tensor.unsqueeze(0).to(device)  # Add batch dimension
        
        # Restore
        with torch.no_grad():
            restored_tensor = model.restore(tensor)
        
        # Postprocess
        restored = postprocess_image(restored_tensor[0])
        
    else:
        print(f"  Using tiled processing ({tile_size}×{tile_size} tiles with 16px overlap)")
        
        # Tile the image
        tiles, positions, original_shape = tile_image(image, tile_size=tile_size, overlap=16)
        print(f"  Created {len(tiles)} tiles")
        
        # Process each tile
        restored_tiles = []
        for i, tile in enumerate(tiles):
            # Preprocess with percentile normalization
            tensor = preprocess_image(tile, use_percentile_norm=use_percentile_norm, 
                                     percentile_low=percentile_low, percentile_high=percentile_high)
            tensor = tensor.unsqueeze(0).to(device)
            
            # Restore
            with torch.no_grad():
                restored_tensor = model.restore(tensor)
            
            # Postprocess
            restored_tile = postprocess_image(restored_tensor[0])
            restored_tiles.append(restored_tile)
            
            if (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/{len(tiles)} tiles")
        
        # Stitch tiles back together
        print("  Stitching tiles...")
        restored = stitch_tiles(restored_tiles, positions, original_shape, overlap=16)
    return restored  # Return float32 array

def degrade_image(model, image, device='cuda', tile_size=128, zstack_context=1,
                  use_percentile_norm=True, percentile_low=1.0, percentile_high=99.0,
                  normalization_type='global'):
    """
    Restore a single image with automatic tiling for large images.
    
    Args:
        model: Trained CycleCARE model
        image_path: Path to input noisy image
        output_path: Path to save restored image (TIFF preserves float32, PNG/JPG converts to uint8)
        device: Device to use ('cuda' or 'cpu')
        tile_size: Tile size (default 128 to match training)
        zstack_context: Number of channels model expects (1 for single-plane, 5 for Z-stack)
        use_percentile_norm: Use percentile-based normalization (recommended for fluorescence)
        percentile_low: Lower percentile for normalization (default 1.0)
        percentile_high: Upper percentile for normalization (default 99.0)
    
    Returns:
        Restored image as float32 numpy array in [0, 1] range
    """
    

    # Normalize input layout: assume channel-first for 3D (C, H, W)
    if image.ndim == 2:
        C = 1
        Y, X = image.shape
        image_cf = image[np.newaxis, :, :]
    elif image.ndim == 3:
        C, Y, X = image.shape
        image_cf = image
    else:
        raise ValueError(f"Unsupported image ndim: {image.ndim}")

    print(f"  Image size: {X}×{Y}, {C} channels")

    # Compute global normalization params if requested
    global_norm = None
    if normalization_type == 'global':
        # compute over entire image (all channels)
        params = compute_normalization_params(image_cf, use_percentile_norm=use_percentile_norm,
                                             percentile_low=percentile_low, percentile_high=percentile_high)
        global_p_min = params['p_min']
        global_p_max = params['p_max']
        global_norm = (global_p_min, global_p_max)
    
    # Check if tiling is needed
    if Y <= tile_size and X <= tile_size:
        print(f"  Using direct processing (image ≤ {tile_size}×{tile_size})")

        # Preprocess with percentile normalization (global or per-tile)
        if global_norm is not None:
            tensor = preprocess_image(image_cf, norm_min=global_p_min, norm_max=global_p_max)
        else:
            tensor = preprocess_image(image_cf, use_percentile_norm=use_percentile_norm,
                                     percentile_low=percentile_low, percentile_high=percentile_high)

        # Handle Z-stack context: replicate single channel if needed
        if tensor.shape[0] == 1 and zstack_context > 1:
            # Single channel but model expects multi-channel - replicate
            tensor = tensor.repeat(zstack_context, 1, 1)
            print(f"  Replicated single channel to {zstack_context} channels for Z-stack model")
        elif tensor.shape[0] != zstack_context:
            print(f"  Warning: Tensor has {tensor.shape[0]} channels but model expects {zstack_context}")

        tensor = tensor.unsqueeze(0).to(device)  # Add batch dimension

        # Degrade (model.forward)
        with torch.no_grad():
            restored_tensor = model.degrade(tensor)

        # Postprocess
        degraded = postprocess_image(restored_tensor[0])

    else:
        print(f"  Using tiled processing ({tile_size}×{tile_size} tiles with 16px overlap)")

        # Tile the image
        tiles, positions, original_shape = tile_image(image_cf, tile_size=tile_size, overlap=16)
        print(f"  Created {len(tiles)} tiles")

        # Process each tile
        degraded_tiles = []
        for i, tile in enumerate(tiles):
            if (i + 1) % 100 == 0:
                print(f"    Preparing tile {i+1}/{len(tiles)}")

            # Preprocess with percentile normalization (global or per-tile)
            if global_norm is not None:
                tensor = preprocess_image(tile, norm_min=global_p_min, norm_max=global_p_max)
            else:
                tensor = preprocess_image(tile, use_percentile_norm=use_percentile_norm,
                                         percentile_low=percentile_low, percentile_high=percentile_high)
            tensor = tensor.unsqueeze(0).to(device)

            # Degrade
            with torch.no_grad():
                restored_tensor = model.degrade(tensor)

            # Postprocess
            degraded_tile = postprocess_image(restored_tensor[0])
            degraded_tiles.append(degraded_tile)

            if (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/{len(tiles)} tiles")

        # Stitch tiles back together
        print("  Stitching tiles...")
        degraded = stitch_tiles(degraded_tiles, positions, original_shape, overlap=16)

    return degraded  # Return float32 array

def _denoise_with_tiling_and_context(stacked_input, model, device, tile_size, overlap=16):
    """
    Helper function to denoise a multi-channel input with tiling.
    
    Args:
        stacked_input: [N_channels, H, W] tensor (could be multi-plane context)
        model: CycleCARE model
        device: Device to use
        tile_size: Tile size for processing
        overlap: Overlap between tiles
    
    Returns:
        Restored image as float32 numpy array [H, W]
    """
    n_channels, h, w = stacked_input.shape
    
    # Create tile grid
    stride = tile_size - overlap
    tiles = []
    positions = []
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            
            # Extract tile from all channels
            tile = stacked_input[:, y:y_end, x:x_end]  # [N_channels, tile_h, tile_w]
            
            # Pad if necessary
            tile_h, tile_w = tile.shape[1], tile.shape[2]
            if tile_h < tile_size or tile_w < tile_size:
                padded = torch.zeros((n_channels, tile_size, tile_size), dtype=tile.dtype)
                padded[:, :tile_h, :tile_w] = tile
                tile = padded
            
            tiles.append(tile)
            positions.append((y, x, y_end, x_end))
    
    # Process tiles
    restored_tiles = []
    for tile in tiles:
        tile_batch = tile.unsqueeze(0).to(device)  # [1, N_channels, H, W]
        
        with torch.no_grad():
            # Check if model is 3D (has denoise method)
            if hasattr(model, 'denoise'):
                # 3D model: reshape [1, N, H, W] -> [1, 1, N, H, W] and call denoise
                input_3d = tile_batch.unsqueeze(1)  # [1, 1, N, H, W]
                output_3d = model.denoise(input_3d)  # [1, 1, D, H, W]
                # Extract center plane
                D = output_3d.size(2)
                center_idx = tile_batch.size(1) // 2 if tile_batch.size(1) <= D else D // 2
                restored_tensor = output_3d[:, :, center_idx, :, :].unsqueeze(1)  # [1, 1, 1, H, W] -> [1, 1, H, W]
                if restored_tensor.dim() == 5:
                    restored_tensor = restored_tensor.squeeze(2)  # [1, 1, H, W]
            else:
                # 2D model: use restore method
                restored_tensor = model.restore(tile_batch)
        
        restored_tile = postprocess_image(restored_tensor[0])
        restored_tiles.append(restored_tile)
    
    # Stitch tiles
    output = np.zeros((h, w), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)
    
    for tile, (y, x, y_end, x_end) in zip(restored_tiles, positions):
        tile_h = y_end - y
        tile_w = x_end - x
        
        # Ensure tile is 2D
        if tile.ndim > 2:
            tile = tile.squeeze()
        
        # Crop tile to actual size
        tile_crop = tile[:tile_h, :tile_w]
        
        # Create weight map with soft edges
        tile_weights = np.ones((tile_h, tile_w), dtype=np.float32)
        
        if overlap > 0:
            # Calculate actual overlap for edges
            overlap_top = min(overlap, tile_h) if y > 0 else 0
            overlap_bottom = min(overlap, tile_h) if y_end < h else 0
            overlap_left = min(overlap, tile_w) if x > 0 else 0
            overlap_right = min(overlap, tile_w) if x_end < w else 0
            
            if overlap_top > 0:
                tile_weights[:overlap_top, :] *= np.linspace(0, 1, overlap_top)[:, np.newaxis]
            if overlap_bottom > 0:
                tile_weights[-overlap_bottom:, :] *= np.linspace(1, 0, overlap_bottom)[:, np.newaxis]
            if overlap_left > 0:
                tile_weights[:, :overlap_left] *= np.linspace(0, 1, overlap_left)
            if overlap_right > 0:
                tile_weights[:, -overlap_right:] *= np.linspace(1, 0, overlap_right)
        
        output[y:y_end, x:x_end] += tile_crop * tile_weights
        weights[y:y_end, x:x_end] += tile_weights
    
    weights = np.maximum(weights, 1e-8)
    output = output / weights
    
    return output.astype(np.float32)


def restore_zstack(model, zstack, device='cuda', tile_size=128, zstack_context=1, 
                   percentile_low=1.0, percentile_high=99.0, normalization_type='global', 
                   sliding_window_size=20):
    """
    Restore a Z-stack using multi-plane context if model was trained with it.
    
    If zstack_context > 1, uses sliding window of N adjacent planes as input
    to denoise each plane, exploiting 3D spatial coherence.
    
    Args:
        model: Trained CycleCARE model
        zstack_path: Path to input Z-stack (TIFF file with multiple planes)
        output_path: Path to save restored Z-stack (saved as float32 TIFF if tifffile available)
        device: Device to use ('cuda' or 'cpu')
        tile_size: Tile size for tiling (default 128)
        zstack_context: Number of adjacent planes to use as input (1 = single plane, 5 = use 5-plane context)
        percentile_low: Lower percentile for normalization (default 1.0)
        percentile_high: Upper percentile for normalization (default 99.0)
        normalization_type: Type of normalization to use (default 'global')
            - 'global': Compute normalization values once across entire Z-stack (prevents stripe artifacts)
            - 'per_plane': Normalize each plane independently (may cause stripe artifacts)
            - 'sliding_window': Use sliding window of planes for local normalization
        sliding_window_size: Number of planes in sliding window for 'sliding_window' normalization (default 20)
    
    Returns:
        Restored Z-stack as float32 numpy array (Z, H, W) in [0, 1] range
    """
    print(f"\nProcessing Z-stack: with {zstack_context}-plane context")
    
    if zstack_context > 1:
        print(f"  Using {zstack_context}-plane context window")
    
    # Check if we have a Z-stack
    if len(zstack.shape) == 2:
        # Single plane
        print(f"  Single plane image: {zstack.shape[1]}×{zstack.shape[0]}")
        num_planes = 1
        zstack = zstack[np.newaxis, :, :]
    elif len(zstack.shape) == 3:
        # Z-stack
        num_planes, h, w = zstack.shape
        print(f"  Z-stack: {num_planes} planes, {w}×{h} each")
    else:
        raise ValueError(f"Unexpected array shape: {zstack.shape}")
    
    # Compute normalization values based on selected method
    if normalization_type == 'global':
        # Global normalization: compute once across entire Z-stack (prevents stripe artifacts)
        global_p_min = np.percentile(zstack, percentile_low)
        global_p_max = np.percentile(zstack, percentile_high)
        print(f"  Global normalization: p{percentile_low}={global_p_min:.3f}, p{percentile_high}={global_p_max:.3f}")
    elif normalization_type == 'per_plane':
        # Per-plane normalization: compute independently for each plane
        global_p_min = None
        global_p_max = None
        print(f"  Per-plane normalization: p{percentile_low}-p{percentile_high}")
    elif normalization_type == 'sliding_window':
        # Sliding window normalization: compute from local window of planes
        global_p_min = None
        global_p_max = None
        print(f"  Sliding window normalization: window size={sliding_window_size}, p{percentile_low}-p{percentile_high}")
    else:
        raise ValueError(f"Unknown normalization_type: {normalization_type}. Must be 'global', 'per_plane', or 'sliding_window'")

    
    # Process each plane
    restored_planes = []
    half_context = zstack_context // 2
    
    print(f"  Processing {num_planes} plane(s)...")
    for z in tqdm(range(num_planes), desc="Denoising planes"):
        
        # Compute sliding window normalization values if needed
        if normalization_type == 'sliding_window':
            # Define window bounds
            half_window = sliding_window_size // 2
            z_start = max(0, z - half_window)
            z_end = min(num_planes, z + half_window + 1)
            
            # Extract window and compute percentiles
            window_stack = zstack[z_start:z_end]
            window_p_min = np.percentile(window_stack, percentile_low)
            window_p_max = np.percentile(window_stack, percentile_high)
        else:
            window_p_min = global_p_min
            window_p_max = global_p_max
        
        if zstack_context > 1:
            # Load context planes (sliding window)
            context_planes = []
            for dz in range(-half_context, half_context + 1):
                z_actual = np.clip(z + dz, 0, num_planes - 1)  # Reflect at edges
                plane_ctx = zstack[z_actual]
                
                # Get dimensions and convert to grayscale if needed
                if len(plane_ctx.shape) == 3:
                    h, w, c = plane_ctx.shape
                    if c > 1:
                        plane_ctx = np.mean(plane_ctx, axis=2).astype(plane_ctx.dtype)
                else:
                    h, w = plane_ctx.shape
                
                context_planes.append(plane_ctx)
            
            # Stack context planes: [zstack_context, H, W]
            # Each plane preprocessed with appropriate normalization values
            context_tensors = []
            for plane_ctx in context_planes:
                tensor_ctx = preprocess_image(plane_ctx, norm_min=window_p_min, norm_max=window_p_max)  # Returns [1, H, W]
                context_tensors.append(tensor_ctx)
            
            # Stack to [zstack_context, H, W]
            stacked_input = torch.cat(context_tensors, dim=0)
            
        else:
            # Single plane mode (backward compatibility)
            plane = zstack[z]
            
            # Get dimensions
            if len(plane.shape) == 3:
                h, w, c = plane.shape
                if c > 1:
                    plane = np.mean(plane, axis=2).astype(plane.dtype)
            else:
                h, w = plane.shape
            
            stacked_input = preprocess_image(plane, norm_min=window_p_min, norm_max=window_p_max)  # [1, H, W]
        
        # Check if tiling is needed
        _, h, w = stacked_input.shape  # Get dimensions from tensor
        
        if h <= tile_size and w <= tile_size:
            # Direct processing
            input_batch = stacked_input.unsqueeze(0).to(device)  # Add batch dim: [1, N, H, W]
            
            with torch.no_grad():
                # Check if model is 3D (has denoise method)
                if hasattr(model, 'denoise'):
                    # 3D model: reshape [1, N, H, W] -> [1, 1, N, H, W] and call denoise
                    input_3d = input_batch.unsqueeze(1)  # [1, 1, N, H, W]
                    output_3d = model.denoise(input_3d)  # [1, 1, D, H, W]
                    # Extract center plane
                    D = output_3d.size(2)
                    center_idx = stacked_input.size(0) // 2 if stacked_input.size(0) <= D else D // 2
                    restored_tensor = output_3d[:, :, center_idx, :, :].unsqueeze(1)  # [1, 1, 1, H, W] -> [1, 1, H, W]
                    if restored_tensor.dim() == 5:
                        restored_tensor = restored_tensor.squeeze(2)  # [1, 1, H, W]
                else:
                    # 2D model: use restore method
                    restored_tensor = model.restore(input_batch)
            
            restored_plane = postprocess_image(restored_tensor[0])
        else:
            #ng for large image...')
            # Tiled processing with context
            # Need to tile all context planes together
            restored_plane = _denoise_with_tiling_and_context(
                stacked_input, model, device, tile_size, overlap=16
            )
        
        restored_planes.append(restored_plane)
    
    # Stack planes back together
    restored_zstack = np.array(restored_planes)
    return restored_zstack



def denoise_zstack(model, zstack, zstack_context=1, device='cuda', tile_size=128, 
                   percentile_low=0.0, percentile_high=99.0, normalization_type='global',
                   sliding_window_size=20):
    """
    Denoise a Z-stack (multi-plane TIFF) by processing each plane independently.
    
    Args:
        model: Loaded model object
        zstack: Input Z-stack as a numpy array (Z, H, W)
        output_path: Path to save restored Z-stack (saved as float32 TIFF)
        device: Device to use ('cuda' or 'cpu')
        tile_size: Tile size for large images (default 128)
        percentile_low: Lower percentile for normalization (default 1.0)
        percentile_high: Upper percentile for normalization (default 99.0)
        normalization_type: Type of normalization ('global', 'per_plane', or 'sliding_window', default 'global')
        sliding_window_size: Window size for sliding window normalization (default 20)
    
    Returns:
        Restored Z-stack as float32 numpy array (Z, H, W) in [0, 1] range
    
    Example:
        restored = denoise_zstack(
            checkpoint_path='outputs/checkpoints/best_model.pth',
            input_path='zstack_noisy.tif',
            output_path='zstack_restored.tif'
        )
    """
    device = device if device == 'cpu' or torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    

    restored = restore_zstack(model, zstack, device=device, tile_size=tile_size, 
                             zstack_context=zstack_context, percentile_low=percentile_low, 
                             percentile_high=percentile_high, normalization_type=normalization_type,
                             sliding_window_size=sliding_window_size)
    
    print("✓ Done!")
    return restored


def denoise_timelapse_stack(checkpoint_path, input_path, output_path, device='cuda', tile_size=128, three_views=False, t_list=None,
                           percentile_low=0.0, percentile_high=99.0, normalization_type='global',
                           sliding_window_size=20):
    
    """Denoise a single multi-dimensional time-lapse TIFF stack with Z-context.

    Expected input dimensions (TCZYX or TZYX):
      - 5D: (T, C, Z, Y, X) with C == 1 (singleton channel) -> processed as (T, Z, Y, X)
      - 4D: (T, Z, Y, X)

    For each timepoint, processes the Z-stack using the same logic as denoise_zstack.

    Args:
        checkpoint_path: Path to model checkpoint (.pth)
        input_path: Path to input multi-dim TIFF stack (TCZYX or TZYX)
        output_path: Path to output restored stack (same dimensionality)
        device: 'cuda' or 'cpu'
        tile_size: Tile size for large XY images
        t_list: List of specific timepoint indices to process (default None processes all timepoints)
        percentile_low: Lower percentile for normalization (default 1.0)
        percentile_high: Upper percentile for normalization (default 99.0)
        normalization_type: Type of normalization ('global', 'per_plane', or 'sliding_window', default 'global')
        sliding_window_size: Window size for sliding window normalization (default 20)

    Returns:
        Restored stack as float32 numpy array with same shape as input.
    """

    device = device if device == 'cpu' or torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model and obtain expected Z-context size
    model, zstack_context = load_model(checkpoint_path, device=device)
    print(f"Model Z-context: {zstack_context} channel(s)")

    output_path = Path(output_path)  # if not already a Path
    output_dir = output_path.parent if not output_path.is_dir() else output_path

    # Load the image stack using memory mapping for efficient lazy loading
    # Only loads metadata initially; timepoints loaded on-demand when accessed
    arr = tifffile.memmap(str(input_path), mode='r')
    orig_shape = arr.shape
    print(f"Loaded stack shape: {orig_shape} (memory-mapped)")
    # Note: float32 conversion happens per-timepoint below to avoid loading entire stack

    # Normalize dimensionality
    if arr.ndim == 5:  # (T,C,Z,Y,X)
        T, C, Z, Y, X = arr.shape
        if C != 1:
            raise ValueError(f"Multi-channel (C={C}) stacks not supported – expected singleton C=1")
        arr = arr.reshape(T, Z, Y, X)  # drop singleton C
    elif arr.ndim == 4:  # (T,Z,Y,X)
        T, Z, Y, X = arr.shape
    elif arr.ndim == 3:  # (Z,Y,X) single timepoint
        Z, Y, X = arr.shape
        T = 1
        arr = arr[np.newaxis, :, :, :]  # add time dim

    print(f"Detected {T} timepoints with Z-stack shape: ({Z}, {Y}, {X})")
    n_timepoints = T    


    # Determine which timepoints to process
    if t_list is None:
        # Process all timepoints
        timepoints_to_process = list(range(n_timepoints))
        print(f"Processing all {n_timepoints} timepoints")
    else:
        # Process specific timepoints from list
        timepoints_to_process = list(t_list)
        # Validate indices
        if any(t < 0 or t >= n_timepoints for t in timepoints_to_process):
            raise ValueError(f"t_list contains invalid timepoint indices. Valid range: 0-{n_timepoints-1}")
        print(f"Processing {len(timepoints_to_process)} specific timepoints: {timepoints_to_process}")

    # Process each timepoint
    print("Beginning timelapse denoising...")
    T_range = len(timepoints_to_process)

    if three_views == False:
        restored = np.zeros((T_range, Z, Y, X), dtype=np.float32)

        for idx, t in enumerate(timepoints_to_process):
            print(f"Processing timepoint {t} ({idx+1}/{len(timepoints_to_process)})")
            zstack = arr[t].astype(np.float32)  # (Z, Y, X) - load and convert this timepoint only
            restored_zstack  = denoise_zstack(model=model,
                                                zstack=zstack,
                                                device=device,
                                                tile_size=tile_size,
                                                zstack_context=zstack_context,
                                                percentile_low=percentile_low,
                                                percentile_high=percentile_high,
                                                normalization_type=normalization_type,
                                                sliding_window_size=sliding_window_size)
            restored[idx,...] = restored_zstack
            outfile = output_dir / f'restored_timepoint_{t:04d}.tif'
            tifffile.imwrite(outfile, restored[idx,...])
            print(f"✓ Saved restored timepoint {t} of shape {restored[idx,...].shape}")

    elif three_views == True:
        restored = np.zeros((3,T_range, Z, Y, X), dtype=np.float32)

        for idx, t in enumerate(timepoints_to_process):
            print(f"Processing timepoint {t} ({idx+1}/{len(timepoints_to_process)})")
            zstack = arr[t]  # (3 , Z, Y, X)

            # Create three views: xy , yz, xz
            zstack_xy = zstack  # original
            zstack_xz = np.transpose(zstack, (1,0,2))
            zstack_yz = np.transpose(zstack, (2,0,1))

            restored_xy = denoise_zstack(model=model,
                                                zstack=zstack_xy,
                                                device=device,
                                                tile_size=tile_size,
                                                zstack_context=zstack_context,
                                                percentile_low=percentile_low,
                                                percentile_high=percentile_high,
                                                normalization_type=normalization_type,
                                                sliding_window_size=sliding_window_size)
            restored_xz = denoise_zstack(model=model,
                                                zstack=zstack_xz,
                                                device=device,
                                                tile_size=tile_size,
                                                zstack_context=zstack_context,
                                                percentile_low=percentile_low,
                                                percentile_high=percentile_high,
                                                normalization_type=normalization_type,
                                                sliding_window_size=sliding_window_size)
            restored_yz = denoise_zstack(model=model,
                                                zstack=zstack_yz,
                                                device=device,
                                                tile_size=tile_size,
                                                zstack_context=zstack_context,
                                                percentile_low=percentile_low,
                                                percentile_high=percentile_high,
                                                normalization_type=normalization_type,
                                            sliding_window_size=sliding_window_size)
            restored[0,idx,...] = restored_xy
            restored[1,idx,...] = np.transpose(restored_xz, (1,0,2))
            restored[2,idx,...] = np.transpose(restored_yz, (1,2,0))
            
            outfile = output_dir / f'restored_timepoint_{t:04d}.tif'
            tifffile.imwrite(outfile, restored[:,idx,...])
            print(f"✓ Saved restored timepoint {t} of shape {restored[:,idx,...].shape} to {outfile}")
    
    if T_range >1:
        # Save full restored stack if multiple timepoints processed
        outfile = output_dir / f'restored_timelapse_stack.tif' 
        tifffile.imwrite(str(outfile), restored)
        print(f"✓ Saved restored timelapse stack to {outfile}")

    # Save output preserving shape
    #output_path = Path(output_path)
    #output_path.parent.mkdir(parents=True, exist_ok=True)
    #tifffile.imwrite(str(output_path), restored)


    return restored


def denoise_batch_zstacks(checkpoint_path, input_dir, output_dir, device='cuda', tile_size=128, 
                        percentile_low=1.0, percentile_high=99.0, normalization_type='global',
                        sliding_window_size=20, pattern='*.tif'):
    """
    Denoise all Z-stacks in a directory.
    
    Args:
        checkpoint_path: Path to model checkpoint
        input_dir: Directory containing input Z-stack files
        output_dir: Directory to save restored Z-stacks
        device: Device to use ('cuda' or 'cpu')
        tile_size: Tile size for large images (default 128)
        percentile_low: Lower percentile for normalization (default 1.0)
        percentile_high: Upper percentile for normalization (default 99.0)
        normalization_type: Type of normalization ('global', 'per_plane', or 'sliding_window', default 'global')
        sliding_window_size: Window size for sliding window normalization (default 20)
        pattern: File pattern to match (default '*.tif')
    
    Example:
        denoise_batch_zstacks(
            checkpoint_path='outputs/checkpoints/best_model.pth',
            input_dir='zstack_noisy/',
            output_dir='zstack_restored/',
            pattern='*.tif'
        )
    """
    device = device if device == 'cpu' or torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    files = sorted(input_dir.glob(pattern))
    if len(files) == 0:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return
    
    print(f"Found {len(files)} Z-stack file(s) to process")
    
    # Load model once
    model, zstack_context = load_model(checkpoint_path, device=device)
    
    # Process each file
    for i, input_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {input_path.name}")
        output_path = output_dir / f"{input_path.stem}_restored{input_path.suffix}"
        
        try:
            # Load Z-stack using memory mapping for efficiency
            zstack = tifffile.memmap(str(input_path), mode='r')[:]  # [:] loads into memory
            
            restored_zstack = restore_zstack(
                model, zstack, device=device, tile_size=tile_size,
                zstack_context=zstack_context,
                percentile_low=percentile_low,
                percentile_high=percentile_high,
                normalization_type=normalization_type,
                sliding_window_size=sliding_window_size
            )
            # Save restored Z-stack
            tifffile.imwrite(str(output_path), restored_zstack)
            print(f"✓ Saved restored Z-stack to {output_path}")
        except Exception as e:
            print(f"Error processing {input_path.name}: {e}")



def addnoise_batch_zstacks(checkpoint_path, input_dir, output_dir, device='cuda', tile_size=128, 
                        percentile_low=1.0, percentile_high=99.0,
                        sliding_window_size=20, pattern='*.tif'):
    """
    Add noise to all Z-stacks in a directory.
    
    Args:
        checkpoint_path: Path to model checkpoint
        input_dir: Directory containing input Z-stack files
        output_dir: Directory to save restored Z-stacks
        device: Device to use ('cuda' or 'cpu')
        tile_size: Tile size for large images (default 128)
        percentile_low: Lower percentile for normalization (default 1.0)
        percentile_high: Upper percentile for normalization (default 99.0)
        normalization_type: Type of normalization ('global', 'per_plane', or 'sliding_window', default 'global')
        sliding_window_size: Window size for sliding window normalization (default 20)
        pattern: File pattern to match (default '*.tif')
    
    Example:
        addnoise_batch_zstacks(
            checkpoint_path='outputs/checkpoints/best_model.pth',
            input_dir='zstack_noisy/',
            output_dir='zstack_restored/',
            pattern='*.tif'
        )
    """
    device = device if device == 'cpu' or torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    files = sorted(input_dir.glob(pattern))
    print('files found are :', files)
    if len(files) == 0:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return
    
    print(f"Found {len(files)} Z-stack file(s) to process")
    
    # Load model once
    model, zstack_context = load_model(checkpoint_path, device=device)
    
    # Process each file
    for i, input_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {input_path.name}")
        output_path = output_dir / f"{input_path.stem}_noisy{input_path.suffix}"
        zstack = tifffile.memmap(str(input_path), mode='r')[:]  # Load into memory
        
        degraded_zstack = degrade_image(model=model, 
                                        image=zstack, 
                                        device=device,
                                        tile_size=tile_size,
                                        zstack_context=zstack_context,
                                        use_percentile_norm=True,
                                        percentile_low=percentile_low,
                                        percentile_high=percentile_high,
        )


        # Save degraded Z-stack
        tifffile.imwrite(str(output_path), degraded_zstack)
        print(f"✓ Saved degraded Z-stack to {output_path}")


if __name__ == "__main__":


    import argparse

    timelapse = True
    batch = False

    if timelapse:
        # Set up command-line argument parser
        parser = argparse.ArgumentParser(
            description='Tiled 3D dsnoising (with 3 views) for large timelapses',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='''
                    Examples:
                    # Segment all timepoints with default settings
                    python simple_inference.py --video video.tif --output results/ --model /path/to/model
                            '''
                        )
        
        parser.add_argument('--checkpoint_path', type=str, help='Path to input checkpoint')
        parser.add_argument('--input_path'     , type=str, help='Path to input timelapse video (TIFF)')
        parser.add_argument('--output_path'    , type=str, help='Output path for results')
        parser.add_argument('--t_list', nargs='+', type=int, metavar='T', default=None,
                       help='Specific timepoints to process. If not provided, all timepoints will be processed. E.g., --t_list 0 5 10 15')
        parser.add_argument('--three_views'    , default=False,
                        help='whether to use three views denoising', action='store_true')
        parser.add_argument('--normalization_type', type=str, default='global',
                        help="Normalization type: 'global', 'per_plane', or 'sliding_window'")
        
        args = parser.parse_args()
        
        # Check if running with command-line arguments or using example code
        if args.input_path and args.output_path and args.checkpoint_path:
            # Command-line mode
            print("Running in command-line mode...")
            
            denoise_timelapse_stack(
                checkpoint_path   = args.checkpoint_path,
                input_path        = args.input_path,
                output_path       = args.output_path,
                t_list            = args.t_list,
                three_views       = args.three_views,
                normalization_type= args.normalization_type
            )

    elif batch:
        # Set up command-line argument parser
        parser = argparse.ArgumentParser(
            description='Batch denoising for multiple Z-stacks in a directory',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='''
                    Examples:
                    # Denoise all Z-stacks in input_dir and save to output_dir
                    python simple_inference.py --input_dir zstacks_noisy/ --output_dir zstacks_restored/ --model /path/to/model
                            '''
                        )
        
        parser.add_argument('--checkpoint_path', type=str, help='Path to input checkpoint')
        parser.add_argument('--input_dir'      , type=str, help='Directory with input Z-stacks (TIFF files)')
        parser.add_argument('--output_dir'     , type=str, help='Output directory for restored Z-stacks')
        parser.add_argument('--pattern'        , type=str, default='*.tif', help='File pattern to match (default: *.tif)')

        
        args = parser.parse_args()
        
        # Check if running with command-line arguments or using example code
        if args.input_dir and args.output_dir and args.checkpoint_path:
            # Command-line mode
            print("Running in command-line mode...")
            
            addnoise_batch_zstacks(
                checkpoint_path   = args.checkpoint_path,
                input_dir         = args.input_dir,
                output_dir        = args.output_dir,
                pattern           = args.pattern,
            )   