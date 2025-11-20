"""
Simple inference script for Cycle-CARE with tiling and Z-stack support.
Automatically tiles large images (>128x128) and processes Z-stacks plane-by-plane.
"""

import os
# Fix for macOS OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
try:
    import tifffile
except ImportError:
    tifffile = None
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


def preprocess_image(image_array, use_percentile_norm=True, percentile_low=0.0, percentile_high=99.0):
    """
    Convert image to tensor and normalize.
    Uses percentile-based normalization for fluorescence microscopy.
    
    Args:
        image_array: Input image as numpy array
        use_percentile_norm: If True, normalize to percentile instead of min/max (recommended for fluorescence)
        percentile_low: Lower percentile for normalization (default 0.0)
        percentile_high: Upper percentile for normalization (default 99.0)
    
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
    if use_percentile_norm:
        # Percentile-based normalization (robust to outliers in fluorescence)
        p_min = np.percentile(image_array, percentile_low)  # Lower percentile (default 1%)
        p_max = np.percentile(image_array, percentile_high)  # Upper percentile (default 99%)
        
        # Avoid division by zero
        if p_max - p_min < 1e-8:
            print(f"  Warning: Very low contrast image (p{percentile_low}={p_min:.3f}, p{percentile_high}={p_max:.3f})")
            image_array = np.zeros_like(image_array)
        else:
            # Normalize to [0, 1] based on percentiles
            image_array = (image_array - p_min) / (p_max - p_min)
            # Clip to [0, 1] (values outside percentile range get clipped)
            image_array = np.clip(image_array, 0.0, 1.0)
    else:
        # Simple min-max normalization (legacy behavior)
        img_min = image_array.min()
        img_max = image_array.max()
        if img_max - img_min < 1e-8:
            image_array = np.zeros_like(image_array)
        else:
            image_array = (image_array - img_min) / (img_max - img_min)
    
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
    Split image into overlapping tiles.
    
    Args:
        image_array: Input image (H, W) or (H, W, C)
        tile_size: Size of each tile (default 128)
        overlap: Overlap between tiles in pixels (default 16)
    
    Returns:
        tiles: List of tile arrays
        positions: List of (y, x) positions for each tile
        original_shape: Original image shape
    """
    if len(image_array.shape) == 3:
        h, w, c = image_array.shape
    else:
        h, w = image_array.shape
        c = 1
    
    stride = tile_size - overlap
    tiles = []
    positions = []
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Get tile boundaries
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            
            # Extract tile
            if c == 1:
                tile = image_array[y:y_end, x:x_end]
            else:
                tile = image_array[y:y_end, x:x_end, :]
            
            # Pad if necessary to reach tile_size
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                if c == 1:
                    padded = np.zeros((tile_size, tile_size), dtype=image_array.dtype)
                    padded[:tile.shape[0], :tile.shape[1]] = tile
                else:
                    padded = np.zeros((tile_size, tile_size, c), dtype=image_array.dtype)
                    padded[:tile.shape[0], :tile.shape[1], :] = tile
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


def restore_image(model, image_path, output_path, device='cuda', tile_size=128, zstack_context=1, 
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
    print(f"\nProcessing: {image_path}")
    
    # Load image
    try:
        import tifffile
        img_array = tifffile.imread(image_path)
    except:
        img = Image.open(image_path)
        img_array = np.array(img)
    
    # Handle different input shapes
    if len(img_array.shape) == 3:
        # Could be (H, W, C) or (C, H, W) - TIFF Z-stacks are usually (Z, H, W)
        if img_array.shape[0] == zstack_context and img_array.shape[0] < img_array.shape[1]:
            # Likely (C, H, W) format from TIFF
            c, h, w = img_array.shape
            print(f"  Image size: {w}×{h}, {c} channels")
        else:
            # Likely (H, W, C) format from PIL
            h, w, c = img_array.shape
            if c == zstack_context:
                # Transpose to (C, H, W) for processing
                img_array = np.transpose(img_array, (2, 0, 1))
                print(f"  Image size: {w}×{h}, {c} channels (transposed to C,H,W)")
            else:
                # Convert to grayscale if doesn't match expected channels
                print(f"Warning: Image has {c} channels but model expects {zstack_context}, converting to grayscale")
                img_array = np.mean(img_array, axis=2).astype(img_array.dtype)
                h, w = img_array.shape
                print(f"  Image size: {w}×{h}")
    else:
        h, w = img_array.shape
        print(f"  Image size: {w}×{h}")
    
    # Check if tiling is needed
    if h <= tile_size and w <= tile_size:
        print(f"  Using direct processing (image ≤ {tile_size}×{tile_size})")
        
        # Preprocess with percentile normalization
        tensor = preprocess_image(img_array, use_percentile_norm=use_percentile_norm, 
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
        tiles, positions, original_shape = tile_image(img_array, tile_size=tile_size, overlap=16)
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
    
    # Save result
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as float32 TIFF (preserves full precision) or convert to uint8/uint16 for other formats
    if output_path.suffix.lower() in ['.tif', '.tiff']:
        if tifffile is not None:
            tifffile.imwrite(output_path, restored)
        else:
            # Fallback: convert to uint16 for PIL
            restored_uint16 = (restored * 65535).astype(np.uint16)
            Image.fromarray(restored_uint16).save(output_path)
    else:
        # For PNG, JPG etc: convert to uint8
        restored_uint8 = (restored * 255).astype(np.uint8)
        Image.fromarray(restored_uint8).save(output_path)
    
    print(f"  ✓ Saved to: {output_path}")
    
    return restored  # Return float32 array


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


def restore_zstack(model, zstack, device='cuda', tile_size=128, zstack_context=1):
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
    
    # Process each plane
    restored_planes = []
    half_context = zstack_context // 2
    
    print(f"  Processing {num_planes} plane(s)...")
    for z in tqdm(range(num_planes), desc="Denoising planes"):
        
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
            # Each plane preprocessed individually, then stacked
            context_tensors = []
            for plane_ctx in context_planes:
                tensor_ctx = preprocess_image(plane_ctx)  # Returns [1, H, W]
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
            
            stacked_input = preprocess_image(plane)  # [1, H, W]
        
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
            # Tiled processing with context
            # Need to tile all context planes together
            restored_plane = _denoise_with_tiling_and_context(
                stacked_input, model, device, tile_size, overlap=16
            )
        
        restored_planes.append(restored_plane)
    
    # Stack planes back together
    restored_zstack = np.array(restored_planes)
    return restored_zstack


def denoise_single_image(checkpoint_path, input_path, output_path, device='cuda', tile_size=128,
                         use_percentile_norm=True, percentile_low=1.0, percentile_high=99.0):
    """
    Denoise a single 2D image.
    
    Args:
        checkpoint_path: Path to model checkpoint
        input_path: Path to input noisy image
        output_path: Path to save restored image (use .tif/.tiff to preserve float32 precision)
        device: Device to use ('cuda' or 'cpu')
        tile_size: Tile size for large images (default 128)
        use_percentile_norm: Use percentile-based normalization (recommended for fluorescence, default True)
        percentile_low: Lower percentile for normalization (default 1.0)
        percentile_high: Upper percentile for normalization (default 99.0)
    
    Returns:
        Restored image as float32 numpy array in [0, 1] range
    
    Example:
        img = denoise_single_image(
            checkpoint_path='outputs/checkpoints/best_model.pth',
            input_path='noisy_image.tif',
            output_path='restored_image.tif'  # Use .tif for float32
        )
    """
    device = device if device == 'cpu' or torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model, zstack_context = load_model(checkpoint_path, device=device)
    
    # Warn if model expects Z-stack but single image provided
    if zstack_context > 1:
        print(f"⚠️  Model was trained with {zstack_context}-plane Z-stack context.")
        print(f"   For single 2D images, output may be suboptimal.")
        print(f"   Consider using denoise_zstack() for best results.")
    
    restored = restore_image(model, input_path, output_path, device=device, tile_size=tile_size, 
                            zstack_context=zstack_context, use_percentile_norm=use_percentile_norm, 
                            percentile_low=percentile_low, percentile_high=percentile_high)
    
    print("✓ Done!")
    return restored


def denoise_zstack(checkpoint_path, zstack, device='cuda', tile_size=128):
    """
    Denoise a Z-stack (multi-plane TIFF) by processing each plane independently.
    
    Args:
        checkpoint_path: Path to model checkpoint
        zstack: Input Z-stack as a numpy array (Z, H, W)
        output_path: Path to save restored Z-stack (saved as float32 TIFF)
        device: Device to use ('cuda' or 'cpu')
        tile_size: Tile size for large images (default 128)
    
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
    
    model, zstack_context = load_model(checkpoint_path, device=device)
    restored = restore_zstack(model, zstack, device=device, tile_size=tile_size, zstack_context=zstack_context)
    
    print("✓ Done!")
    return restored


def denoise_batch(checkpoint_path, input_dir, output_dir, device='cuda', tile_size=128, pattern='*.tif'):
    """
    Denoise all images in a directory.
    
    Args:
        checkpoint_path: Path to model checkpoint
        input_dir: Directory containing input images
        output_dir: Directory to save restored images
        device: Device to use ('cuda' or 'cpu')
        tile_size: Tile size for large images (default 128)
        pattern: File pattern to match (default '*.tif')
    
    Example:
        denoise_batch(
            checkpoint_path='outputs/checkpoints/best_model.pth',
            input_dir='noisy_images/',
            output_dir='restored_images/',
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
    
    print(f"Found {len(files)} file(s) to process")
    
    # Load model once
    model, zstack_context = load_model(checkpoint_path, device=device)
    
    if zstack_context > 1:
        print(f"⚠️  Model was trained with {zstack_context}-plane Z-stack context.")
        print(f"   For best results, use denoise_zstack() for Z-stack files.")
    
    # Process each file
    for i, input_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {input_path.name}")
        output_path = output_dir / f"{input_path.stem}_restored{input_path.suffix}"
        
        try:
            restore_image(model, input_path, output_path, device=device, tile_size=tile_size)
        except Exception as e:
            print(f"  ✗ Error processing {input_path.name}: {e}")
    
    print(f"\n✓ Batch processing complete! Results in: {output_dir}")


def denoise_timelapse_stack(checkpoint_path, input_path, output_path, device='cuda', tile_size=128):
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

    Returns:
        Restored stack as float32 numpy array with same shape as input.
    """
    
    device = device if device == 'cpu' or torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model and obtain expected Z-context size
    model, zstack_context = load_model(checkpoint_path, device=device)
    print(f"Model Z-context: {zstack_context} channel(s)")

    # Load the image stack
    arr = tifffile.imread(str(input_path))
    orig_shape = arr.shape
    print(f"Loaded stack shape: {orig_shape}")

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

    # Prepare output array (float32 in [0,1])
    restored = np.zeros((T, Z, Y, X), dtype=np.float32)
    half_context = zstack_context // 2

    #please convert the input array to float32 for processing
    arr = arr.astype(np.float32)

    # Process each timepoint
    print("Beginning timelapse denoising...")
    for i in range(T):
        print(f"Processing timepoint {i+1}/{T}...")
        zstack = arr[i]  # (Z, Y, X)
        restored_zstack  = denoise_zstack(checkpoint_path=checkpoint_path,
                                            zstack=zstack,
                                            device=device,
                                            tile_size=tile_size)
        restored[i] = restored_zstack

    # Save output preserving shape
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(output_path), restored)

    print(f"✓ Saved restored timelapse stack to {output_path}")
    return restored

if __name__ == "__main__":
    # Example usage - uncomment the one you want to run:
    denoise_timelapse_stack(
        checkpoint_path= '/Users/ewheeler/Downloads/checkpoint_epoch_0080.pth',
        input_path     = '/Users/ewheeler/cycleCARE_HPC/test_data/b2-2a_2c_pos6-01_crop_t1_z50-359_y750-1262_x1000-1512.tif',
        output_path    = '/Users/ewheeler/cycleCARE_HPC/test_data/b2-2a_2c_pos6-01_crop_t1_z50-359_y750-1262_x1000-1512_restored.tif',
    )
