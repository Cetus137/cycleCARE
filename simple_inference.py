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
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    tifffile = None
    print("Warning: tifffile not available. Using PIL fallback for TIFF I/O.")
except Exception as e:
    HAS_TIFFFILE = False
    tifffile = None
    print(f"Warning: tifffile import failed ({e}). Using PIL fallback for TIFF I/O.")
from tqdm import tqdm


def _load_tiff_array(file_path):
    """
    Load a TIFF file as numpy array with fallback support.
    Supports both single and multi-page TIFFs.
    
    Args:
        file_path: Path to TIFF file
    
    Returns:
        numpy.ndarray: Array with shape (H, W) or (N, H, W)
    """
    if HAS_TIFFFILE:
        try:
            return tifffile.imread(file_path)
        except Exception as e:
            print(f"Warning: tifffile.imread failed: {e}")
            print("Falling back to PIL...")
    
    # Fallback to PIL
    img = Image.open(file_path)
    
    # Check if multi-page
    try:
        n_frames = img.n_frames
    except AttributeError:
        n_frames = 1
    
    if n_frames == 1:
        # Single page
        return np.array(img)
    else:
        # Multi-page TIFF
        planes = []
        for i in range(n_frames):
            img.seek(i)
            planes.append(np.array(img))
        return np.stack(planes, axis=0)  # Shape: (N, H, W)


def _save_tiff_array(file_path, array):
    """
    Save numpy array as TIFF with fallback support.
    
    Args:
        file_path: Path to save TIFF file
        array: numpy array with shape (H, W) or (N, H, W)
    """
    if HAS_TIFFFILE:
        try:
            tifffile.imwrite(file_path, array)
            return
        except Exception as e:
            print(f"Warning: tifffile.imwrite failed: {e}")
            print("Falling back to PIL...")
    
    # Fallback to PIL
    if len(array.shape) == 2:
        # Single plane
        img = Image.fromarray(array)
        img.save(file_path)
    elif len(array.shape) == 3:
        # Multi-page TIFF
        images = [Image.fromarray(array[i]) for i in range(array.shape[0])]
        images[0].save(file_path, save_all=True, append_images=images[1:])
    else:
        raise ValueError(f"Cannot save array with shape {array.shape}")


def load_model(checkpoint_path, device='cuda'):
    """
    Load the trained model from checkpoint.
    
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
    
    # Get config from checkpoint
    if 'config' in checkpoint:
        cfg = checkpoint['config']
        
        # Handle both dict (new format) and Config object (old format)
        if isinstance(cfg, dict):
            unet_depth = cfg.get('UNET_DEPTH', 2)
            unet_filters = cfg.get('UNET_FILTERS', 64)
            img_channels = cfg.get('IMG_CHANNELS', 1)
            zstack_context = cfg.get('ZSTACK_CONTEXT', img_channels)  # For backward compat
            print(f"Loaded config from checkpoint (dict): UNET_DEPTH={unet_depth}, UNET_FILTERS={unet_filters}, IMG_CHANNELS={img_channels}")
            if zstack_context > 1:
                print(f"  Z-stack mode: {zstack_context}-plane context")
        else:
            # Old format: Config object with class-level attributes
            unet_depth = getattr(cfg, 'UNET_DEPTH', 2)
            unet_filters = getattr(cfg, 'UNET_FILTERS', 64)
            img_channels = getattr(cfg, 'IMG_CHANNELS', 1)
            zstack_context = getattr(cfg, 'ZSTACK_CONTEXT', img_channels)
            print(f"Loaded config from checkpoint (object): UNET_DEPTH={unet_depth}, UNET_FILTERS={unet_filters}, IMG_CHANNELS={img_channels}")
            if zstack_context > 1:
                print(f"  Z-stack mode: {zstack_context}-plane context")
            print("  WARNING: Old checkpoint format detected. Config may not reflect actual training settings.")
            print("  Verifying against model weights...")
            
            # For old format, verify against actual weights
            state_dict = checkpoint['model_state_dict']
            max_encoder_idx = -1
            for key in state_dict.keys():
                if 'encoder_blocks.' in key:
                    idx = int(key.split('encoder_blocks.')[1].split('.')[0])
                    max_encoder_idx = max(max_encoder_idx, idx)
            
            if max_encoder_idx >= 0:
                actual_depth = max_encoder_idx + 1
                if actual_depth != unet_depth:
                    print(f"  ⚠️  Config mismatch detected! Using actual UNET_DEPTH={actual_depth} from weights (config says {unet_depth})")
                    unet_depth = actual_depth
    else:
        # Infer architecture from state_dict
        print("Config not in checkpoint, inferring from model weights...")
        state_dict = checkpoint['model_state_dict']
        
        # Find maximum encoder block index
        max_encoder_idx = -1
        for key in state_dict.keys():
            if 'encoder_blocks.' in key:
                # Extract index from keys like "G_AB.encoder_blocks.2.conv1..."
                idx = int(key.split('encoder_blocks.')[1].split('.')[0])
                max_encoder_idx = max(max_encoder_idx, idx)
        
        # UNET_DEPTH = max_encoder_idx + 1 (since indices start at 0)
        unet_depth = max_encoder_idx + 1 if max_encoder_idx >= 0 else 2
        
        # Infer filters from bottleneck size
        # Bottleneck has shape [filters * (2^depth), filters * (2^depth), 3, 3]
        for key in state_dict.keys():
            if 'bottleneck.0.block.0.weight' in key:
                bottleneck_size = state_dict[key].shape[0]
                unet_filters = bottleneck_size // (2 ** unet_depth)
                break
        else:
            unet_filters = 64  # default
        
        img_channels = 1  # default for grayscale
        zstack_context = img_channels  # Assume Z-stack context equals img_channels
        
        print(f"Inferred architecture: UNET_DEPTH={unet_depth}, UNET_FILTERS={unet_filters}, IMG_CHANNELS={img_channels}")
    
    # Create model
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
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded successfully from {checkpoint_path}")
    return model, zstack_context


def preprocess_image(image_array):
    """Convert image to tensor and normalize."""
    # Ensure float [0, 1] range
    if image_array.dtype == np.uint8:
        image_array = image_array.astype(np.float32) / 255.0
    elif image_array.dtype == np.uint16:
        image_array = image_array.astype(np.float32) / 65535.0
    
    # Convert to tensor and normalize to [-1, 1]
    tensor = torch.from_numpy(image_array).float()
    if len(tensor.shape) == 2:  # Grayscale
        tensor = tensor.unsqueeze(0)  # Add channel dimension
    tensor = (tensor - 0.5) / 0.5  # Normalize to [-1, 1]
    
    return tensor


def postprocess_image(tensor):
    """Convert tensor back to numpy image as float32 in [0, 1] range."""
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor * 0.5) + 0.5
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy as float32
    image_array = tensor.squeeze().cpu().numpy().astype(np.float32)
    
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


def restore_image(model, image_path, output_path, device='cuda', tile_size=128):
    """
    Restore a single image with automatic tiling for large images.
    
    Args:
        model: Trained CycleCARE model
        image_path: Path to input noisy image
        output_path: Path to save restored image (TIFF preserves float32, PNG/JPG converts to uint8)
        device: Device to use ('cuda' or 'cpu')
        tile_size: Tile size (default 128 to match training)
    
    Returns:
        Restored image as float32 numpy array in [0, 1] range
    """
    print(f"\nProcessing: {image_path}")
    
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Get original dimensions and ensure grayscale
    if len(img_array.shape) == 3:
        h, w, c = img_array.shape
        if c > 1:
            print("Warning: Converting multi-channel to grayscale")
            img_array = np.mean(img_array, axis=2).astype(img_array.dtype)
    else:
        h, w = img_array.shape
    
    print(f"  Image size: {w}×{h}")
    
    # Check if tiling is needed
    if h <= tile_size and w <= tile_size:
        print(f"  Using direct processing (image ≤ {tile_size}×{tile_size})")
        
        # Preprocess
        tensor = preprocess_image(img_array)
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
            # Preprocess
            tensor = preprocess_image(tile)
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
        _save_tiff_array(output_path, restored)
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
            restored_tensor = model.restore(tile_batch)
        
        restored_tile = postprocess_image(restored_tensor[0])
        restored_tiles.append(restored_tile)
    
    # Stitch tiles
    output = np.zeros((h, w), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)
    
    for tile, (y, x, y_end, x_end) in zip(restored_tiles, positions):
        tile_h = y_end - y
        tile_w = x_end - x
        tile_crop = tile[:tile_h, :tile_w]
        
        # Create weight map with soft edges
        tile_weights = np.ones((tile_h, tile_w), dtype=np.float32)
        
        if overlap > 0:
            if y > 0:
                tile_weights[:overlap, :] *= np.linspace(0, 1, overlap)[:, np.newaxis]
            if y_end < h:
                tile_weights[-overlap:, :] *= np.linspace(1, 0, overlap)[:, np.newaxis]
            if x > 0:
                tile_weights[:, :overlap] *= np.linspace(0, 1, overlap)
            if x_end < w:
                tile_weights[:, -overlap:] *= np.linspace(1, 0, overlap)
        
        output[y:y_end, x:x_end] += tile_crop * tile_weights
        weights[y:y_end, x:x_end] += tile_weights
    
    weights = np.maximum(weights, 1e-8)
    output = output / weights
    
    return output.astype(np.float32)


def restore_zstack(model, zstack_path, output_path, device='cuda', tile_size=128, zstack_context=1):
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
    print(f"\nProcessing Z-stack: {zstack_path}")
    
    if zstack_context > 1:
        print(f"  Using {zstack_context}-plane context window")
    
    # Load as multi-page TIFF (uses fallback if tifffile unavailable)
    try:
        zstack = _load_tiff_array(zstack_path)
    except Exception as e:
        print(f"Error loading Z-stack: {e}")
        print("Treating as single plane image")
        img = Image.open(zstack_path)
        zstack = np.array(img)
        if len(zstack.shape) == 2:
            zstack = zstack[np.newaxis, :, :]  # Add Z dimension
    
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
    
    # Save result
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if num_planes == 1:
        # Single plane
        if output_path.suffix.lower() in ['.tif', '.tiff']:
            _save_tiff_array(output_path, restored_zstack[0])
        else:
            # For PNG, JPG etc: convert to uint8
            restored_uint8 = (restored_zstack[0] * 255).astype(np.uint8)
            Image.fromarray(restored_uint8).save(output_path)
    else:
        # Z-stack - save as multi-page TIFF
        _save_tiff_array(output_path, restored_zstack)
    
    print(f"  ✓ Saved to: {output_path}")
    
    return restored_zstack


def denoise_single_image(checkpoint_path, input_path, output_path, device='cuda', tile_size=128):
    """
    Denoise a single 2D image.
    
    Args:
        checkpoint_path: Path to model checkpoint
        input_path: Path to input noisy image
        output_path: Path to save restored image (use .tif/.tiff to preserve float32 precision)
        device: Device to use ('cuda' or 'cpu')
        tile_size: Tile size for large images (default 128)
    
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
    
    restored = restore_image(model, input_path, output_path, device=device, tile_size=tile_size)
    
    print("✓ Done!")
    return restored


def denoise_zstack(checkpoint_path, input_path, output_path, device='cuda', tile_size=128):
    """
    Denoise a Z-stack (multi-plane TIFF) by processing each plane independently.
    
    Args:
        checkpoint_path: Path to model checkpoint
        input_path: Path to input Z-stack (multi-page TIFF)
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
    restored = restore_zstack(model, input_path, output_path, device=device, tile_size=tile_size, zstack_context=zstack_context)
    
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


if __name__ == "__main__":
    # Example usage - uncomment the one you want to run:
    
    denoise_zstack(
        checkpoint_path='/Users/ewheeler/cycleCARE_HPC/outputs/checkpoints/checkpoint_epoch_0090.pth',
        input_path ='/Users/ewheeler/cycleCARE_HPC/b2-2a_2c_pos6-01_deskew_cgt-cropped_for_segmentation0.tif',
        output_path='/Users/ewheeler/cycleCARE_HPC/b2-2a_2c_pos6-01_deskew_cgt-cropped_for_segmentation0_restored.tif',
    )
