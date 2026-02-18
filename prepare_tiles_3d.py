
"""
Prepare 3D training tiles from 256×256×256 volumes for CycleCARE depth restoration.

Workflow:
1. Load all TIFF files from input directory
2. Extract Z=0-128 (first 128 layers)
3. Split into non-overlapping 128×128×128 cubes
4. Save Z=0-64 of each cube to Domain A (surface)
5. Save Z=64-128 of each cube to Domain B (deep)

Configuration is passed via command line arguments from the SLURM script.
"""

import argparse
import numpy as np
import tifffile
from pathlib import Path


def load_and_validate_volume(filepath, expected_shape):
    """Load TIFF and validate it's 256³."""
    volume = tifffile.imread(filepath).astype(np.float32)
    
    # Squeeze out singleton dimensions (e.g., (1, 256, 256, 256) -> (256, 256, 256))
    original_shape = volume.shape
    volume = np.squeeze(volume)
    
    if original_shape != volume.shape:
        print(f"  Squeezed shape: {original_shape} → {volume.shape}")
    
    if volume.shape != expected_shape:
        print(f"  ⚠️  Warning: Expected shape {expected_shape}, got {volume.shape}")
        print(f"      Will attempt to process anyway...")
    
    return volume


def extract_domain_cubes(volume, z_start, z_end, tile_size_z=64, tile_size_yx=128):
    """
    Extract non-overlapping tiles from a specific Z-range of the volume.
    
    Args:
        volume: Full 3D volume (D, H, W)
        z_start: Starting Z-layer (inclusive)
        z_end: Ending Z-layer (exclusive)
        tile_size_z: Depth of tiles in Z dimension (default: 64)
        tile_size_yx: Size of tiles in Y and X dimensions (default: 128 for 64×128×128)
    
    Returns:
        cubes: List of extracted tiles
        positions: List of (d, h, w) positions in the original volume
    """
    # Extract the Z-range
    volume_subset = volume[z_start:z_end, :, :]
    D, H, W = volume_subset.shape
    
    cubes = []
    positions = []
    
    # Calculate how many tiles fit in each dimension
    n_d = D // tile_size_z
    n_h = H // tile_size_yx
    n_w = W // tile_size_yx
    
    print(f"  Extracting {n_d}×{n_h}×{n_w} = {n_d*n_h*n_w} non-overlapping {tile_size_z}×{tile_size_yx}×{tile_size_yx} tiles from Z={z_start}-{z_end}")
    
    for d_idx in range(n_d):
        for h_idx in range(n_h):
            for w_idx in range(n_w):
                d_start = d_idx * tile_size_z
                h_start = h_idx * tile_size_yx
                w_start = w_idx * tile_size_yx
                
                cube = volume_subset[
                    d_start:d_start + tile_size_z,
                    h_start:h_start + tile_size_yx,
                    w_start:w_start + tile_size_yx
                ]
                
                cubes.append(cube)
                # Store absolute position in original volume
                positions.append((z_start + d_idx, h_idx, w_idx))
    
    return cubes, positions


def process_volume(filepath, volume_idx, args, domain_a_dir, domain_b_dir):
    """Process a single 256³ volume."""
    print(f"\n{'='*70}")
    print(f"Processing volume {volume_idx}: {filepath.name}")
    print(f"{'='*70}")
    
    # Load volume
    print(f"Loading {filepath}...")
    volume = load_and_validate_volume(filepath, args.expected_shape)
    print(f"  Loaded shape: {volume.shape}")
    print(f"  Value range: [{volume.min():.2f}, {volume.max():.2f}]")
    
    # Extract cubes from Domain A Z-range
    print(f"\nDomain A: Extracting Z-layers {args.domain_a_z_start}-{args.domain_a_z_end}...")
    cubes_a, positions_a = extract_domain_cubes(
        volume, args.domain_a_z_start, args.domain_a_z_end, 
        tile_size_z=args.tile_size_z, tile_size_yx=args.tile_size_yx
    )
    
    # Extract cubes from Domain B Z-range
    print(f"\nDomain B: Extracting Z-layers {args.domain_b_z_start}-{args.domain_b_z_end}...")
    cubes_b, positions_b = extract_domain_cubes(
        volume, args.domain_b_z_start, args.domain_b_z_end,
        tile_size_z=args.tile_size_z, tile_size_yx=args.tile_size_yx
    )
    
    # Verify we have the same number of cubes
    if len(cubes_a) != len(cubes_b):
        print(f"  ⚠️  Warning: Domain A has {len(cubes_a)} cubes, Domain B has {len(cubes_b)} cubes")
        print(f"      Using minimum count: {min(len(cubes_a), len(cubes_b))}")
        num_cubes = min(len(cubes_a), len(cubes_b))
    else:
        num_cubes = len(cubes_a)
    
    # Save cubes (with filtering for edge tiles)
    basename = filepath.stem
    saved_count = 0
    skipped_count = 0
    
    for i in range(num_cubes):
        cube_a = cubes_a[i]
        cube_b = cubes_b[i]
        d, h, w = positions_a[i]  # Use domain A positions for naming
        
        # Check if either cube has >1% zeros (edge/background tiles)
        zero_pct_a = (np.count_nonzero(cube_a == 0) / cube_a.size) * 100
        zero_pct_b = (np.count_nonzero(cube_b == 0) / cube_b.size) * 100
        
        if zero_pct_a > 1.0 or zero_pct_b > 1.0:
            skipped_count += 1
            print(f"  Skipping tile d={d} h={h} w={w}: {zero_pct_a:.2f}% zeros (A), {zero_pct_b:.2f}% zeros (B)")
            continue
        
        # Generate filenames with position info
        filename = f"{basename}_d{d}_h{h}_w{w}.tif"
        
        # Save domain A
        domain_a_path = domain_a_dir / filename
        tifffile.imwrite(domain_a_path, cube_a)
        
        # Save domain B
        domain_b_path = domain_b_dir / filename
        tifffile.imwrite(domain_b_path, cube_b)
        
        saved_count += 1
    
    if skipped_count > 0:
        print(f"\n✓ Saved {saved_count} cube pairs (skipped {skipped_count} with >1% zeros)")
    else:
        print(f"\n✓ Saved {saved_count} cube pairs")
    print(f"  Domain A (Z={args.domain_a_z_start}-{args.domain_a_z_end}): {domain_a_dir}")
    print(f"  Domain B (Z={args.domain_b_z_start}-{args.domain_b_z_end}): {domain_b_dir}")
    
    return num_cubes


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare 3D training tiles from 256³ volumes for CycleCARE depth restoration"
    )
    
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory containing 256³ TIFF volumes")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for processed tiles")
    parser.add_argument("--domain_a_name", type=str, default="surface",
                        help="Name for domain A directory")
    parser.add_argument("--domain_b_name", type=str, default="depth",
                        help="Name for domain B directory")
    parser.add_argument("--expected_shape", type=int, nargs=3, default=[256, 256, 256],
                        help="Expected volume shape (D H W)")
    parser.add_argument("--domain_a_z_start", type=int, default=0,
                        help="Domain A: Starting Z-layer (inclusive, default: 0)")
    parser.add_argument("--domain_a_z_end", type=int, default=64,
                        help="Domain A: Ending Z-layer (exclusive, default: 64)")
    parser.add_argument("--domain_b_z_start", type=int, default=64,
                        help="Domain B: Starting Z-layer (inclusive, default: 64)")
    parser.add_argument("--domain_b_z_end", type=int, default=128,
                        help="Domain B: Ending Z-layer (exclusive, default: 128)")
    parser.add_argument("--tile_size_z", type=int, default=64,
                        help="Depth of tiles in Z dimension (default: 64)")
    parser.add_argument("--tile_size_yx", type=int, default=128,
                        help="Size of tiles in Y and X dimensions (default: 128 for 64×128×128)")
    
    return parser.parse_args()


def main():
    """Main processing function."""
    args = parse_args()
    
    # Convert to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    domain_a_dir = output_dir / args.domain_a_name
    domain_b_dir = output_dir / args.domain_b_name
    expected_shape = tuple(args.expected_shape)
    args.expected_shape = expected_shape
    
    print("="*70)
    print("3D Tile Preparation for CycleCARE Depth Restoration")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Expected volume shape: {expected_shape}")
    print(f"  Domain A Z-range: {args.domain_a_z_start}-{args.domain_a_z_end} ({args.domain_a_z_end - args.domain_a_z_start} slices)")
    print(f"  Domain B Z-range: {args.domain_b_z_start}-{args.domain_b_z_end} ({args.domain_b_z_end - args.domain_b_z_start} slices)")
    print(f"  Tile size: {args.tile_size_z}×{args.tile_size_yx}×{args.tile_size_yx}")
    print(f"  Domain A: {domain_a_dir}")
    print(f"  Domain B: {domain_b_dir}")
    
    # Create output directories
    domain_a_dir.mkdir(parents=True, exist_ok=True)
    domain_b_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n Output directories created")
    
    # Find all TIFF files
    print(f"\nSearching for TIFF files in {input_dir}...")
    tiff_files = sorted(list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff")))
    
    if not tiff_files:
        print(f"\n No TIFF files found in {input_dir}")
        print(f"   Please add your 256³ volumes to this directory")
        return
    
    print(f"✓ Found {len(tiff_files)} TIFF file(s)")
    
    # Process each volume
    total_cubes = 0
    successful_volumes = 0
    
    for idx, filepath in enumerate(tiff_files, 1):
        try:
            cube_count = process_volume(filepath, idx, args, domain_a_dir, domain_b_dir)
            total_cubes += cube_count
            successful_volumes += 1
        except Exception as e:
            print(f"\n Error processing {filepath.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"\nSummary:")
    print(f"  Volumes processed: {successful_volumes}/{len(tiff_files)}")
    print(f"  Total cube pairs: {total_cubes}")
    print(f"  Domain A tiles: {total_cubes} × {args.tile_size_z}×{args.tile_size_yx}×{args.tile_size_yx} (Z={args.domain_a_z_start}-{args.domain_a_z_end})")
    print(f"  Domain B tiles: {total_cubes} × {args.tile_size_z}×{args.tile_size_yx}×{args.tile_size_yx} (Z={args.domain_b_z_start}-{args.domain_b_z_end})")
    
    # Count output files
    domain_a_files = list(domain_a_dir.glob("*.tif"))
    domain_b_files = list(domain_b_dir.glob("*.tif"))
    
    print(f"\nOutput verification:")
    print(f"  Domain A: {len(domain_a_files)} files")
    print(f"  Domain B: {len(domain_b_files)} files")
    
    if len(domain_a_files) > 0 and len(domain_b_files) > 0:
        print(f"\n✓ SUCCESS! Ready for training")
        print(f"\nNext steps:")
        print(f"  1. Verify output: ls -lh {domain_a_dir}")
        print(f"  2. Check setup: python verify_3d_setup.py")
        print(f"  3. Start training: python train_3d.py")
    else:
        print(f"\n  Warning: Output directories may be empty")
    
    print("="*70)


if __name__ == "__main__":
    main()
