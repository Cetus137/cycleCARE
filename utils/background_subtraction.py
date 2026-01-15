from pathlib import Path
import numpy as np
from skimage import io, img_as_float
from skimage.restoration import rolling_ball
from skimage.util import img_as_ubyte
import tifffile as tiff

def rolling_ball_dir(
    input_dir,
    output_dir,
    radius=50,
    clip=True,
):
    """
    Apply rolling-ball background subtraction to all TIFF files in a directory.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing input .tif/.tiff files
    output_dir : str or Path
        Directory to save corrected images
    radius : int
        Rolling-ball radius in pixels
    clip : bool
        Clip result to [0, 1] before saving
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiff_files = sorted(
        list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    )

    if not tiff_files:
        raise FileNotFoundError("No TIFF files found in input directory.")

    for tif in tiff_files:
        print(f"Processing: {tif.name}")

        img = tiff.imread(tif)

        print(f" - Original shape: {img.shape}, dtype: {img.dtype}")

        # Convert to float [0, 1]
        img_f = (img - img.min()) / (img.max() - img.min())
        img_f = img_f.astype(np.float32)

        print('image min/max after normalization:', img_f.min(), img_f.max())


        # Handle 2D vs stack
        if img_f.ndim == 2:
            bg = rolling_ball(
                img_f,
                radius=radius,
            )
            corrected = img_f - bg

        elif img_f.ndim == 3:
            corrected = np.empty_like(img_f)
            for i in range(img_f.shape[0]):
                bg = rolling_ball(
                    img_f[i],
                    radius=radius,
                )
                corrected[i] = img_f[i] - bg
        else:
            raise ValueError(f"Unsupported image shape: {img_f.shape}")

        if clip:
            print(" - Clipping to [0, 1]")
            corrected = np.clip(corrected, 0, 1)

        print("image min/max before saving:", corrected.min(), corrected.max())

        out_path = output_dir / tif.name
        tiff.imwrite(out_path, corrected.astype(np.float32))

    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply rolling-ball background subtraction to all TIFF files in a directory."
    )
    parser.add_argument("--input_dir", type=str, help="Input directory with TIFF files.")
    parser.add_argument("--output_dir", type=str, help="Output directory for corrected images.")
    parser.add_argument(
        "--radius", type=int, default=25, help="Rolling-ball radius in pixels."
    )
    parser.add_argument(
        "--no_clip",
        action="store_true",
        help="Do not clip result to [0, 1].",
    )

    args = parser.parse_args()

    rolling_ball_dir(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        radius=args.radius,
        clip=not args.no_clip,
    )