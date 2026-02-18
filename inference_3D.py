"""
3D Inference script for Cycle-CARE with automatic model detection.
Automatically detects 2D vs 3D models from checkpoint and processes volumes accordingly.
"""

import os
import torch
import numpy as np
from pathlib import Path
import tifffile
from tqdm import tqdm


def get_device(device='cuda'):
    """
    Get available device with automatic CPU fallback.
    
    Args:
        device: Requested device ('cuda' or 'cpu')
    
    Returns:
        str: Actual device to use ('cuda' or 'cpu')
    """
    if device == 'cuda':
        if torch.cuda.is_available():
            print(f"✓ Using CUDA device: {torch.cuda.get_device_name(0)}")
            return 'cuda'
        else:
            print("⚠ CUDA not available, falling back to CPU")
            return 'cpu'
    else:
        print("Using CPU")
        return 'cpu'


def load_model(checkpoint_path, device='cuda'):
    """
    Load trained model from checkpoint with automatic 2D/3D detection.
    
    Detects model type by checking weight tensor dimensions:
    - 2D models: Conv layers have 4 dims [out_ch, in_ch, H, W]
    - 3D models: Conv layers have 5 dims [out_ch, in_ch, D, H, W]
    
    Returns:
        tuple: (model, is_3d, metadata_dict)
            - model: Loaded CycleCARE or CycleCARE3D model
            - is_3d: Boolean indicating if model is 3D
            - metadata: Dict with architecture parameters
    """
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    print("Detecting model architecture from checkpoint...")
    
    # Detect 2D vs 3D by checking weight dimensions
    is_3d = False
    for key in state_dict.keys():
        if 'G_AB' in key and 'weight' in key and ('conv' in key.lower() or 'initial' in key.lower()):
            weight_shape = state_dict[key].shape
            if len(weight_shape) == 5:  # [out_ch, in_ch, D, H, W]
                is_3d = True
                print(f"  Detected 3D model (weight shape: {weight_shape})")
                break
            elif len(weight_shape) == 4:  # [out_ch, in_ch, H, W]
                is_3d = False
                print(f"  Detected 2D model (weight shape: {weight_shape})")
                break
    
    # Infer img_channels from first conv layer
    img_channels = 1
    for key in state_dict.keys():
        if 'G_AB.initial_conv.0.block.0.weight' in key or 'G_AB.initial.0.conv1.block.0.weight' in key:
            img_channels = state_dict[key].shape[1]
            break
    
    # Infer unet_depth from encoder blocks
    max_encoder_idx = -1
    for key in state_dict.keys():
        if 'encoder_blocks.' in key:
            idx = int(key.split('encoder_blocks.')[1].split('.')[0])
            max_encoder_idx = max(max_encoder_idx, idx)
    unet_depth = max_encoder_idx + 1 if max_encoder_idx >= 0 else 3
    
    # Infer unet_filters from bottleneck
    unet_filters = 24 if is_3d else 64  # default
    for key in state_dict.keys():
        if 'bottleneck.0.block.0.weight' in key:
            bottleneck_channels = state_dict[key].shape[0]
            unet_filters = bottleneck_channels // (2 ** unet_depth)
            break
    
    # Infer disc_kernel_size from discriminator
    disc_kernel_size = 4  # default
    for key in state_dict.keys():
        if 'D_A.model.0.weight' in key or 'D_B.model.0.weight' in key:
            disc_kernel_size = state_dict[key].shape[2 if is_3d else 2]
            break
    
    print(f"  Model Type: {'3D' if is_3d else '2D'}")
    print(f"  UNET_DEPTH: {unet_depth}")
    print(f"  UNET_FILTERS: {unet_filters}")
    print(f"  IMG_CHANNELS: {img_channels}")
    print(f"  DISC_KERNEL_SIZE: {disc_kernel_size}")
    
    # Load appropriate model class
    if is_3d:
        from models.cycle_care_3d import CycleCARE3D
        model = CycleCARE3D(
            img_channels=img_channels,
            unet_depth=unet_depth,
            unet_filters=unet_filters,
            unet_kernel_size=3,
            disc_filters=unet_filters,  # Match generator filters
            disc_num_layers=3,
            disc_kernel_size=disc_kernel_size,
            use_batch_norm=True,
            use_dropout=True,
            dropout_rate=0.5
        ).to(device)
    else:
        from models import CycleCARE
        model = CycleCARE(
            img_channels=img_channels,
            unet_depth=unet_depth,
            unet_filters=unet_filters,
            unet_kernel_size=3,
            disc_filters=64,
            disc_num_layers=3,
            disc_kernel_size=disc_kernel_size,
            use_batch_norm=True,
            use_dropout=True,
            dropout_rate=0.5
        ).to(device)
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    metadata = {
        'is_3d': is_3d,
        'img_channels': img_channels,
        'unet_depth': unet_depth,
        'unet_filters': unet_filters,
        'disc_kernel_size': disc_kernel_size
    }
    
    print(f"✓ Model loaded from {checkpoint_path}")
    return model, is_3d, metadata


def preprocess_volume(volume, use_percentile_norm=True, percentile_low=0.0, percentile_high=99.0):
    """
    Normalize volume to [-1, 1] range for model input.
    
    Args:
        volume: Input volume as numpy array (D, H, W) or (C, D, H, W)
        use_percentile_norm: Use percentile-based normalization
        percentile_low: Lower percentile (default 0.0)
        percentile_high: Upper percentile (default 99.0)
    
    Returns:
        Normalized tensor in [-1, 1] range
    """
    # Convert to float32
    volume = volume.astype(np.float32)
    
    # Compute normalization parameters
    if use_percentile_norm:
        p_min = np.percentile(volume, percentile_low)
        p_max = np.percentile(volume, percentile_high)
    else:
        p_min = volume.min()
        p_max = volume.max()
    
    # Normalize to [0, 1]
    if p_max - p_min < 1e-8:
        print(f"  Warning: Very low contrast (min={p_min:.3f}, max={p_max:.3f})")
        volume = np.zeros_like(volume)
    else:
        volume = (volume - p_min) / (p_max - p_min)
        volume = np.clip(volume, 0.0, 1.0)
    
    # Convert to tensor
    tensor = torch.from_numpy(volume).float()
    
    # Ensure correct shape (C, D, H, W)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)  # Add channel dimension
    
    # Normalize to [-1, 1] (same as training)
    tensor = (tensor - 0.5) / 0.5
    
    return tensor, (p_min, p_max)


def postprocess_volume(tensor, norm_params=None):
    """
    Convert tensor back to numpy volume in [0, 1] range.
    
    Args:
        tensor: Output tensor from model
        norm_params: Optional (p_min, p_max) to denormalize to original range
    
    Returns:
        Volume as float32 numpy array
    """
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor * 0.5) + 0.5
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    volume = tensor.squeeze().cpu().numpy().astype(np.float32)
    
    # Optionally denormalize to original range
    if norm_params is not None:
        p_min, p_max = norm_params
        volume = volume * (p_max - p_min) + p_min
    
    return volume


def tile_volume_3d(volume, tile_size=(64, 128, 128), overlap=(16, 32, 32)):
    """
    Split a 3D volume into overlapping tiles.
    
    Args:
        volume: Input tensor (C, D, H, W)
        tile_size: Size of each tile (d, h, w)
        overlap: Overlap between tiles in each dimension (d, h, w)
    
    Returns:
        tiles: List of tile tensors
        positions: List of (d_start, h_start, w_start) positions
        original_shape: Original volume shape (C, D, H, W)
    """
    C, D, H, W = volume.shape
    td, th, tw = tile_size
    od, oh, ow = overlap
    
    # Calculate stride (tile_size - overlap)
    stride_d = td - od
    stride_h = th - oh
    stride_w = tw - ow
    
    tiles = []
    positions = []
    
    # Calculate number of tiles needed in each dimension
    n_tiles_d = max(1, int(np.ceil((D - td) / stride_d)) + 1) if D > td else 1
    n_tiles_h = max(1, int(np.ceil((H - th) / stride_h)) + 1) if H > th else 1
    n_tiles_w = max(1, int(np.ceil((W - tw) / stride_w)) + 1) if W > tw else 1
    
    print(f"  Tiling volume into {n_tiles_d}×{n_tiles_h}×{n_tiles_w} = {n_tiles_d*n_tiles_h*n_tiles_w} tiles")
    print(f"  Tile size: {tile_size}, Overlap: {overlap}, Stride: ({stride_d}, {stride_h}, {stride_w})")
    
    for i_d in range(n_tiles_d):
        for i_h in range(n_tiles_h):
            for i_w in range(n_tiles_w):
                # Calculate tile boundaries
                d_start = min(i_d * stride_d, D - td)
                h_start = min(i_h * stride_h, H - th)
                w_start = min(i_w * stride_w, W - tw)
                
                d_end = min(d_start + td, D)
                h_end = min(h_start + th, H)
                w_end = min(w_start + tw, W)
                
                # Extract tile
                tile = volume[:, d_start:d_end, h_start:h_end, w_start:w_end]
                
                # Pad if necessary (edge tiles might be smaller)
                if tile.shape[1:] != tile_size:
                    pad_d = td - tile.shape[1]
                    pad_h = th - tile.shape[2]
                    pad_w = tw - tile.shape[3]
                    tile = torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h, 0, pad_d), mode='reflect')
                
                tiles.append(tile)
                positions.append((d_start, h_start, w_start))
    
    return tiles, positions, (C, D, H, W)


def create_blend_weight_3d(tile_size, overlap):
    """
    Create a 3D weight mask for blending overlapping tiles.
    Uses linear ramp in overlap regions.
    
    Args:
        tile_size: (d, h, w) tile dimensions
        overlap: (od, oh, ow) overlap in each dimension
    
    Returns:
        weight: 3D weight mask (d, h, w) with values in [0, 1]
    """
    d, h, w = tile_size
    od, oh, ow = overlap
    
    # Create 1D ramps for each dimension
    def create_ramp(size, overlap):
        ramp = np.ones(size, dtype=np.float32)
        if overlap > 0:
            # Linear ramp from 0 to 1 in overlap region
            ramp_vals = np.linspace(0, 1, overlap)
            ramp[:overlap] = ramp_vals
            ramp[-overlap:] = ramp_vals[::-1]
        return ramp
    
    ramp_d = create_ramp(d, od // 2)
    ramp_h = create_ramp(h, oh // 2)
    ramp_w = create_ramp(w, ow // 2)
    
    # Create 3D weight by outer product
    weight = np.einsum('i,j,k->ijk', ramp_d, ramp_h, ramp_w)
    
    return torch.from_numpy(weight).float()


def stitch_tiles_3d(tiles, positions, original_shape, tile_size=(64, 128, 128), overlap=(16, 32, 32)):
    """
    Stitch processed tiles back into full volume with blending.
    
    Args:
        tiles: List of processed tile tensors (C, d, h, w)
        positions: List of (d_start, h_start, w_start) positions
        original_shape: Target shape (C, D, H, W)
        tile_size: Size of each tile
        overlap: Overlap between tiles
    
    Returns:
        Stitched volume tensor (C, D, H, W)
    """
    C, D, H, W = original_shape
    
    # Initialize output volume and weight accumulator
    output = torch.zeros(original_shape, dtype=torch.float32)
    weights = torch.zeros((D, H, W), dtype=torch.float32)
    
    # Create blending weight once
    blend_weight = create_blend_weight_3d(tile_size, overlap)
    
    print(f"  Stitching {len(tiles)} tiles back into volume {original_shape}")
    
    for tile, (d_start, h_start, w_start) in zip(tiles, positions):
        td, th, tw = tile_size
        
        # Get actual tile size (might be smaller for padded edge tiles)
        d_end = min(d_start + td, D)
        h_end = min(h_start + th, H)
        w_end = min(w_start + tw, W)
        
        actual_d = d_end - d_start
        actual_h = h_end - h_start
        actual_w = w_end - w_start
        
        # Crop tile if it was padded
        tile_crop = tile[:, :actual_d, :actual_h, :actual_w]
        weight_crop = blend_weight[:actual_d, :actual_h, :actual_w]
        
        # Accumulate weighted tile
        output[:, d_start:d_end, h_start:h_end, w_start:w_end] += tile_crop * weight_crop
        weights[d_start:d_end, h_start:h_end, w_start:w_end] += weight_crop
    
    # Normalize by accumulated weights (avoid division by zero)
    weights = torch.clamp(weights, min=1e-8)
    output = output / weights.unsqueeze(0)
    
    return output


def denoise_volume(model, volume, device='cuda', is_3d=True, use_percentile_norm=True, 
                   percentile_low=0.0, percentile_high=99.0, return_original_range=False,
                   tile_size=(64, 128, 128), overlap=(16, 32, 32), use_tiling=True):
    """
    Denoise a single volume using the loaded model with automatic tiling for large volumes.
    
    Args:
        model: Loaded CycleCARE3D or CycleCARE model
        volume: Input volume as numpy array (D, H, W) or (C, D, H, W)
        device: Device to use ('cuda' or 'cpu')
        is_3d: Whether model is 3D (uses denoise() method) or 2D (uses restore())
        use_percentile_norm: Use percentile-based normalization
        percentile_low: Lower percentile for normalization
        percentile_high: Upper percentile for normalization
        return_original_range: If True, denormalize output to original input range
        tile_size: Size of tiles for processing large volumes (d, h, w)
        overlap: Overlap between tiles (d, h, w)
        use_tiling: Whether to use tiling (auto-enabled for volumes larger than tile_size)
    
    Returns:
        Denoised volume as float32 numpy array
    """
    print(f"Processing volume: {volume.shape}")
    
    # Preprocess
    tensor, norm_params = preprocess_volume(
        volume, 
        use_percentile_norm=use_percentile_norm,
        percentile_low=percentile_low,
        percentile_high=percentile_high
    )
    
    # Check if tiling is needed
    C, D, H, W = tensor.shape
    needs_tiling = (D > tile_size[0] or H > tile_size[1] or W > tile_size[2]) and use_tiling
    
    if needs_tiling:
        print(f"  Volume {(D, H, W)} exceeds tile size {tile_size}, using tiled processing")
        
        # Tile the volume
        tiles, positions, original_shape = tile_volume_3d(tensor, tile_size=tile_size, overlap=overlap)
        
        # Process each tile
        processed_tiles = []
        with torch.no_grad():
            for idx, tile in enumerate(tqdm(tiles, desc="Processing tiles", disable=len(tiles) < 5)):
                # Add batch dimension: (1, C, D, H, W)
                tile_batch = tile.unsqueeze(0).to(device)
                
                if is_3d and hasattr(model, 'denoise'):
                    # 3D model
                    output_tile = model.denoise(tile_batch)
                elif hasattr(model, 'restore'):
                    # 2D model: process plane-by-plane
                    B, C, D_tile, H_tile, W_tile = tile_batch.shape
                    output_planes = []
                    for d in range(D_tile):
                        plane = tile_batch[:, :, d, :, :]
                        restored_plane = model.restore(plane)
                        output_planes.append(restored_plane)
                    output_tile = torch.stack(output_planes, dim=2)
                else:
                    raise ValueError("Model has neither denoise() nor restore() method")
                
                # Remove batch dimension and move to CPU
                processed_tiles.append(output_tile[0].cpu())
        
        # Stitch tiles back together
        output = stitch_tiles_3d(processed_tiles, positions, original_shape, 
                                tile_size=tile_size, overlap=overlap)
        
        print(f"  Output volume shape: {output.shape}")
        
    else:
        print(f"  Volume fits in memory, processing without tiling")
        
        # Add batch dimension: (1, C, D, H, W)
        tensor = tensor.unsqueeze(0).to(device)
        
        print(f"  Input tensor shape: {tensor.shape}")
        
        # Denoise
        with torch.no_grad():
            if is_3d and hasattr(model, 'denoise'):
                # 3D model: expect (B, C, D, H, W)
                output = model.denoise(tensor)
            elif hasattr(model, 'restore'):
                # 2D model: process plane-by-plane
                print("  Warning: Using 2D model on 3D volume - processing planes independently")
                B, C, D, H, W = tensor.shape
                output_planes = []
                for d in range(D):
                    plane = tensor[:, :, d, :, :]  # (B, C, H, W)
                    restored_plane = model.restore(plane)
                    output_planes.append(restored_plane)
                output = torch.stack(output_planes, dim=2)  # (B, C, D, H, W)
            else:
                raise ValueError("Model has neither denoise() nor restore() method")
        
        print(f"  Output tensor shape: {output.shape}")
        
        # Remove batch dimension
        output = output[0]
    
    # Postprocess
    if return_original_range:
        restored = postprocess_volume(output, norm_params=norm_params)
    else:
        restored = postprocess_volume(output, norm_params=None)
    
    return restored


def denoise_batch_volumes(checkpoint_path, input_dir, output_dir, device='cuda',
                          percentile_low=0.0, percentile_high=99.0, pattern='*.tif',
                          tile_size=(64, 128, 128), overlap=(16, 32, 32), use_tiling=True):
    """
    Denoise all volumes in a directory.
    
    Args:
        checkpoint_path: Path to model checkpoint
        input_dir: Directory containing input volume files
        output_dir: Directory to save restored volumes
        device: Device to use ('cuda' or 'cpu')
        percentile_low: Lower percentile for normalization
        percentile_high: Upper percentile for normalization
        pattern: File pattern to match (default '*.tif')
        tile_size: Size of tiles for processing large volumes (d, h, w)
        overlap: Overlap between tiles (d, h, w)
        use_tiling: Whether to use tiling
    """
    device = get_device(device)
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    files = sorted(input_dir.glob(pattern))
    if len(files) == 0:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return
    
    print(f"Found {len(files)} volume file(s) to process")
    
    # Load model once
    model, is_3d, metadata = load_model(checkpoint_path, device=device)
    
    # Process each file
    for i, input_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {input_path.name}")
        output_path = output_dir / f"{input_path.stem}_restored{input_path.suffix}"
        
        try:
            # Load volume
            volume = tifffile.imread(str(input_path))
            
            # Denoise
            restored = denoise_volume(
                model, volume, device=device, is_3d=is_3d,
                use_percentile_norm=True,
                percentile_low=percentile_low,
                percentile_high=percentile_high,
                return_original_range=True,
                tile_size=tile_size,
                overlap=overlap,
                use_tiling=use_tiling
            )
            
            # Save
            tifffile.imwrite(str(output_path), restored)
            print(f"✓ Saved restored volume to {output_path}")
            
        except Exception as e:
            print(f"✗ Error processing {input_path.name}: {e}")
            import traceback
            traceback.print_exc()


def denoise_timelapse(checkpoint_path, input_path, output_path, device='cuda',
                      percentile_low=0.0, percentile_high=99.0, t_list=None,
                      normalization_type='global', tile_size=(64, 128, 128), 
                      overlap=(16, 32, 32), use_tiling=True):
    """
    Denoise a 3D volume or 4D timelapse (T, Z, Y, X) or 5D (T, C, Z, Y, X).
    Automatically handles single volumes by treating them as T=1 timelapse.
    
    Args:
        checkpoint_path: Path to model checkpoint
        input_path: Path to input volume/timelapse file
        output_path: Path to save restored volume/timelapse
        device: Device to use ('cuda' or 'cpu')
        percentile_low: Lower percentile for normalization
        percentile_high: Upper percentile for normalization
        t_list: List of specific timepoints to process (None = all)
        normalization_type: 'global' (across all timepoints) or 'per_timepoint'
        tile_size: Size of tiles for large volumes (d, h, w)
        overlap: Overlap between tiles (d, h, w)
        use_tiling: Whether to use tiling for large volumes
    """
    device = get_device(device)
    
    # Load model
    model, is_3d, metadata = load_model(checkpoint_path, device=device)
    
    # Load data (memory-mapped for efficiency)
    arr = tifffile.memmap(str(input_path), mode='r')
    orig_shape = arr.shape
    print(f"Loaded data shape: {orig_shape}")
    
    # Normalize dimensions - handle 3D, 4D, and 5D arrays
    if arr.ndim == 5:  # (T, C, Z, Y, X)
        T, C, Z, Y, X = arr.shape
        if C != 1:
            raise ValueError(f"Multi-channel (C={C}) not supported - expected C=1")
        arr = arr.reshape(T, Z, Y, X)
    elif arr.ndim == 4:  # (T, Z, Y, X)
        T, Z, Y, X = arr.shape
    elif arr.ndim == 3:  # (Z, Y, X) - Single volume
        Z, Y, X = arr.shape
        T = 1
        arr = arr.reshape(1, Z, Y, X)  # Add time dimension
        print(f"  Single volume detected, treating as T=1 timelapse")
    else:
        raise ValueError(f"Expected 3D, 4D or 5D array, got shape {orig_shape}")
    
    print(f"Detected {T} timepoint(s) with volume shape: ({Z}, {Y}, {X})")
    
    # Determine timepoints to process
    if t_list is None:
        timepoints_to_process = list(range(T))
        print(f"Processing all {T} timepoints")
    else:
        timepoints_to_process = list(t_list)
        if any(t < 0 or t >= T for t in timepoints_to_process):
            raise ValueError(f"Invalid timepoint indices. Valid range: 0-{T-1}")
        print(f"Processing {len(timepoints_to_process)} specific timepoints: {timepoints_to_process}")
    
    # Compute global normalization if requested
    if normalization_type == 'global':
        print("Computing global normalization parameters...")
        sample_size = min(10000000, arr.size)  # Sample up to 10M voxels
        sample_indices = np.random.choice(arr.size, sample_size, replace=False)
        flat_arr = arr.reshape(-1)
        sample_data = flat_arr[sample_indices].astype(np.float32)
        global_p_min = np.percentile(sample_data, percentile_low)
        global_p_max = np.percentile(sample_data, percentile_high)
        print(f"  Global p{percentile_low}={global_p_min:.3f}, p{percentile_high}={global_p_max:.3f}")
    else:
        global_p_min = None
        global_p_max = None
    
    # Process timepoints
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    restored_volumes = []
    
    for idx, t in enumerate(tqdm(timepoints_to_process, desc="Processing timepoints", disable=T==1)):
        if T > 1:
            print(f"\nTimepoint {t} ({idx+1}/{len(timepoints_to_process)})")
        
        # Load volume for this timepoint
        volume = arr[t].astype(np.float32)  # (Z, Y, X)
        
        # Use denoise_volume with tiling support
        # Handle global vs per-timepoint normalization
        if global_p_min is not None and normalization_type == 'global':
            # Pre-normalize with global parameters then denoise
            volume_norm = (volume - global_p_min) / (global_p_max - global_p_min)
            volume_norm = np.clip(volume_norm, 0.0, 1.0)
            
            # Process without additional normalization (already normalized)
            restored = denoise_volume(
                model, volume_norm, device=device, is_3d=is_3d,
                use_percentile_norm=False,  # Skip normalization
                return_original_range=False,  # Keep in [0, 1]
                tile_size=tile_size,
                overlap=overlap,
                use_tiling=use_tiling
            )
            
            # Denormalize to original range
            restored = restored * (global_p_max - global_p_min) + global_p_min
        else:
            # Per-timepoint normalization (handled by denoise_volume)
            restored = denoise_volume(
                model, volume, device=device, is_3d=is_3d,
                use_percentile_norm=True,
                percentile_low=percentile_low,
                percentile_high=percentile_high,
                return_original_range=True,
                tile_size=tile_size,
                overlap=overlap,
                use_tiling=use_tiling
            )
        
        restored_volumes.append(restored)
        
        # Save individual timepoint
        if T > 1:
            tp_path = output_dir / f"restored_timepoint_{t:04d}.tif"
            tifffile.imwrite(str(tp_path), restored)
            print(f"  ✓ Saved to {tp_path}")
    
    # Save full output (timelapse or single volume)
    if T == 1:
        # Single volume - save directly
        tifffile.imwrite(str(output_path), restored_volumes[0])
        print(f"\n✓ Saved restored volume to {output_path}")
    elif len(restored_volumes) > 1:
        restored_stack = np.array(restored_volumes)
        tifffile.imwrite(str(output_path), restored_stack)
        print(f"\n✓ Saved full restored timelapse to {output_path}")
    
    print("Done!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='3D Inference for Cycle-CARE with automatic mode detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Automatic Mode Detection:
  - If INPUT is a directory  → Batch mode (process all volumes in directory)
  - If INPUT is a file       → Volume/Timelapse mode (3D or 4D data)
    * 3D volumes (Z,Y,X) are treated as T=1 timelapse
    * 4D timelapses (T,Z,Y,X) process multiple timepoints

Examples:
  # Single volume (auto-detected)
  python inference_3D.py --checkpoint models/epoch_3.pth --input volume.tif --output restored.tif
  
  # Batch directory (auto-detected)
  python inference_3D.py --checkpoint models/epoch_3.pth --input data/volumes/ --output data/restored/
  
  # Timelapse with custom tile size
  python inference_3D.py --checkpoint models/epoch_3.pth --input timelapse.tif --output restored.tif --tile_size 64 128 128
  
  # Process specific timepoints
  python inference_3D.py --checkpoint models/epoch_3.pth --input timelapse.tif --output restored.tif --t_list 0 5 10 15 20
        '''
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input path (file for single/timelapse, directory for batch)')
    parser.add_argument('--output', type=str, required=True, 
                       help='Output path (file for single/timelapse, directory for batch)')
    
    # Device settings
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use (cuda/cpu, auto-fallback enabled)')
    
    # Normalization settings
    parser.add_argument('--percentile_low', type=float, default=0.0, 
                       help='Lower percentile for normalization (default: 0.0)')
    parser.add_argument('--percentile_high', type=float, default=99.0, 
                       help='Upper percentile for normalization (default: 99.0)')
    
    # Tiling settings
    parser.add_argument('--tile_size', nargs=3, type=int, default=[64, 128, 128], 
                       help='Tile size (D H W) for processing large volumes (default: 64 128 128)')
    parser.add_argument('--overlap', nargs=3, type=int, default=[16, 32, 32],
                       help='Overlap (D H W) between tiles (default: 16 32 32)')
    parser.add_argument('--no_tiling', action='store_true', 
                       help='Disable tiling (process entire volume at once, may cause OOM)')
    
    # Batch mode settings
    parser.add_argument('--pattern', type=str, default='*.tif', 
                       help='File pattern for batch mode (default: *.tif)')
    
    # Timelapse mode settings
    parser.add_argument('--t_list', nargs='+', type=int, 
                       help='Specific timepoints to process (default: all)')
    parser.add_argument('--normalization', type=str, default='global', 
                       choices=['global', 'per_timepoint'], 
                       help='Normalization for timelapse: global or per_timepoint (default: global)')
    
    args = parser.parse_args()
    
    # Detect available device with CPU fallback
    device = get_device(args.device)
    
    # Auto-detect mode based on input type
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_dir():
        # Batch mode: process directory of volumes
        print(f"{'='*70}")
        print(f"BATCH MODE: Processing directory")
        print(f"{'='*70}")
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print(f"Pattern: {args.pattern}")
        print(f"{'='*70}\n")
        
        denoise_batch_volumes(
            checkpoint_path=args.checkpoint,
            input_dir=str(input_path),
            output_dir=str(output_path),
            device=device,
            percentile_low=args.percentile_low,
            percentile_high=args.percentile_high,
            pattern=args.pattern,
            tile_size=tuple(args.tile_size),
            overlap=tuple(args.overlap),
            use_tiling=not args.no_tiling
        )
        
    elif input_path.is_file():
        # Volume/Timelapse mode: process single file (handles 3D and 4D)
        print(f"{'='*70}")
        print(f"VOLUME/TIMELAPSE MODE: Processing file")
        print(f"{'='*70}")
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print(f"Tile size: {args.tile_size}, Overlap: {args.overlap}")
        if args.t_list:
            print(f"Timepoints: {args.t_list}")
        print(f"Normalization: {args.normalization}")
        print(f"{'='*70}\n")
        
        denoise_timelapse(
            checkpoint_path=args.checkpoint,
            input_path=str(input_path),
            output_path=str(output_path),
            device=device,
            percentile_low=args.percentile_low,
            percentile_high=args.percentile_high,
            t_list=args.t_list,
            normalization_type=args.normalization,
            tile_size=tuple(args.tile_size),
            overlap=tuple(args.overlap),
            use_tiling=not args.no_tiling
        )
        
    else:
        print(f"ERROR: Input path does not exist: {input_path}")
        print("Please provide a valid file or directory path")
        parser.print_help()
