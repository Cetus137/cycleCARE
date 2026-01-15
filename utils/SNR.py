import numpy as np
import os
from pathlib import Path
import tifffile



def compute_snr(image):
	"""
	Compute Signal-to-Noise Ratio (SNR) for a 2D image.

	Accepts a 256x256 image, but works for any HxW.
	Image must be a NumPy array.

	SNR definition used:
	- Signal: mean intensity of the image (or foreground)
	- Noise: standard deviation of the image (or background)

	This implementation uses a robust, unsupervised estimate:
	- Signal = median of the top 10% intensities
	- Noise = standard deviation of the bottom 10% intensities

	Returns SNR in linear scale only.

	Args:
		image (np.ndarray): 2D image, shape (H, W) or (1, H, W)
			Values can be float in [0,1] or arbitrary range.

	Returns:
		float: snr_linear
	"""

	# Ensure NumPy array
	image = np.asarray(image)

	# Squeeze channel if provided as (1, H, W)
	if image.ndim == 3 and image.shape[0] == 1:
		image = image[0]

	if image.ndim != 2:
		raise ValueError("compute_snr expects a 2D image (H, W) or (1, H, W)")

	# Flatten for percentile-based robust stats
	flat = image.reshape(-1).astype(np.float64)

	# Handle NaNs/Infs safely
	flat = flat[np.isfinite(flat)]
	if flat.size == 0:
		raise ValueError("Image contains no finite values")

	# Robust signal/noise estimation via percentiles
	# Bottom 10% approximates background; top 10% approximates foreground
	p10 = np.percentile(flat, 10)
	p90 = np.percentile(flat, 90)

	background = flat[flat <= p10]
	foreground = flat[flat >= p90]

	# Fallbacks if percentiles collapse
	if background.size < 10:
		background = flat
	if foreground.size < 10:
		foreground = flat

	noise_std = float(np.std(background))
	signal_mean = float(np.median(foreground))

	# Avoid division by zero
	if noise_std == 0.0:
		# If the image is perfectly uniform, SNR is theoretically infinite
		return float('inf')

	snr_linear = (p90 - p10) / noise_std #signal_mean / noise_std
	return float(snr_linear)


def compute_snr_directory(directory, snr_threshold, output_dir=None, save=True):
	"""
	Compute SNR for all TIFF images in a directory.

	Args:
		directory (str | os.PathLike | Path): Folder to scan
		output_path (str | os.PathLike | None): Where to save images that pass the SNR threshold.
			Required if save=True.
		save (bool): If True, save each image whose SNR >= threshold to output_path.

	Returns:
		list[tuple[str, float, bool]]: List of (filepath, snr, saved_flag) entries
	"""
	dir_path = Path(directory)
	if not dir_path.is_dir():
		raise ValueError(f"Not a directory: {directory}")

	# Find .tif / .tiff files
	files = sorted(list(dir_path.glob("**/*.tif")) + list(dir_path.glob("**/*.tiff")))
	if len(files) == 0:
		return []

	results = []
	snr_fn = compute_snr

	out_dir = None
	if save:
		if output_dir is None:
			out_dir = dir_path / "snr_filtered"
		else:
			out_dir = Path(output_dir)
		out_dir.mkdir(parents=True, exist_ok=True)

	for f in files:
		# Load image; tifffile handles microscopy TIFFs well
		img_all = tifffile.imread(str(f))
		img = img_all
		if img.ndim > 2:
			zshape = img.shape[0]
			# If multi-channel/stack, evaluate SNR on middle channel/plane when available
			img = img[zshape // 2, ...] if zshape > 1 else img[0, ...]
		img = img.astype(np.float32)

		snr = snr_fn(img)
		print(snr)

		if save and out_dir is not None and snr >= snr_threshold:
			out_path = out_dir / f.name
			# Preserve original data when saving
			tifffile.imwrite(str(out_path), img_all)

		results.append((str(f), float(snr)))

	return results


if __name__ == "__main__":
		import argparse

		parser = argparse.ArgumentParser(description="Compute SNR for TIFF images in a directory and optionally save those above a threshold.")
		parser.add_argument("--input_dir", type=str, help="Input directory to scan for .tif/.tiff files")
		parser.add_argument("--threshold", type=float, default=5.0, help="SNR threshold for saving (default: 5.0)")
		parser.add_argument("--output", type=str, default=None, help="Output directory to save passing images (default: <directory>/snr_filtered)")
		parser.add_argument("--save", action="store_true", help="Save images with SNR above the threshold")

		args = parser.parse_args()

		save = args.save
		results = compute_snr_directory(directory=args.input_dir, snr_threshold=args.threshold, output_dir=args.output, save=save)

		# Print summary
		total = len(results)
		saved_count = 0
		for path, snr in results:
			print(f"{path}\tSNR={snr:.4f}")
			if save and snr >= args.threshold:
				saved_count += 1

		if save:
			print(f"Saved {saved_count}/{total} images with SNR >= {args.threshold}")

