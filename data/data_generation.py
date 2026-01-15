from aicsimageio import AICSImage
import dask.array as da
import numpy as np
import tifffile as tiff
import os

def load_czi_image(czi_path) :
    """
    Load CZI image using AICSImageIO as a Dask array for out-of-memory processing.

    Parameters:
    -----------
    czi_path : str
        Path to the CZI file

    Returns:
    --------
    Tuple[da.Array, dict]
        Dask array with shape (C, 1, Z, Y, X) and metadata dictionary
    """

    # Load image with AICSImageIO
    img = AICSImage(czi_path)

    # Get Dask array - this doesn't load the full image into memory
    dask_array = img.dask_data

    # Get metadata
    metadata = {
        'original_shape': dask_array.shape,
        'dtype': dask_array.dtype,
        'physical_pixel_sizes': img.physical_pixel_sizes,
        'channel_names': img.channel_names,
        'dims': img.dims,
        'size_gb': dask_array.nbytes / (1024**3)
    }
    print(metadata)
    return dask_array, metadata

def save_czi_image_as_tif(dask_array, save_path, T , C, Z):
    """
    Save Dask array as CZI file using AICSImageIO.

    Parameters:
    -----------
    dask_array : da.Array
        Dask array to be saved.
    save_path : str
        Path to save the CZI file.
    metadata : dict
        Metadata dictionary for the image.
    """
    # Convert Dask array to NumPy array (this may require sufficient memory)
    numpy_array = dask_array[T , C, ...]
    print(numpy_array.shape)
    numpy_array = numpy_array.compute()

    # Save as CZI
    print('computed and now saving')
    tiff.imwrite(save_path, numpy_array.astype(np.float32))
    print(f"Image saved to {save_path}")
    return


def tile_2D_image(image, tile_size, z, overlap , output_path=None):
    """
    Tile a 2D image into smaller overlapping patches.

    Parameters:
    -----------
    image : numpy.ndarray
        2D input image to be tiled.
    tile_size : int
        Size of each tile (tile_size x tile_size).
    overlap : int
        Number of pixels to overlap between tiles.

    Returns:
    --------
    List[numpy.ndarray]
        List of tiled image patches.
    """
    tiles = []
    step = tile_size - overlap
    if len(image.shape) == 2:
        h, w = image.shape
    elif len(image.shape) == 3:
        z, h, w = image.shape

    for y in range(0, h - tile_size + 1, step):
        for x in range(0, w - tile_size + 1, step):
            tile = image[:, y:y + tile_size, x:x + tile_size]
            tiles.append(tile)
            #print(tile.shape)
            if output_path is not None:
                tile_filename = f"{output_path}_z{z}_y{y}_x{x}.tif"
                #remove a tile is it more than 1% zeros
                print(tile.size)
                non_zero_ratio = np.count_nonzero(tile) / tile.size
                print(f"non_zero_ratio: {non_zero_ratio}")
                if non_zero_ratio < 0.99:
                    print(f"Removing tile: {tile_filename}")
                    continue
                else:
                    print('tile shape:', tile.shape)
                    tiff.imwrite(tile_filename, tile.astype(np.float32))
                #print(f"Saved tile: {tile_filename}")

    return tiles

def tile_3D_image(image, tile_size, overlap, output_path=None):
    """
    Tile a 3D image into smaller overlapping patches.

    Parameters:
    -----------
    image : numpy.ndarray
        3D input image to be tiled.
    tile_size : int
        Size of each tile (tile_size x tile_size x tile_size).
    overlap : int
        Number of pixels to overlap between tiles.

    Returns:
    --------
    List[numpy.ndarray]
        List of tiled image patches.
    """
    tiles = []
    step = tile_size - overlap
    d, h, w = image.shape

    for z in range(0, d - tile_size + 1, step):
        for y in range(0, h - tile_size + 1, step):
            for x in range(0, w - tile_size + 1, step):
                tile = image[z:z + tile_size, y:y + tile_size, x:x + tile_size]
                tiles.append(tile)
                if output_path is not None:
                    tile_filename = f"{output_path}_z{z}_y{y}_x{x}.tif"
                    tiff.imwrite(tile_filename, tile.astype(np.float32))
                    print(f"Saved tile: {tile_filename}")

    return tiles

    """
    Load a time-lapse CZI image, extract tiles for specified time points, and save them as TIFF files.

    Parameters:
    -----------
    image_path : str
        Path to the CZI file.
    output_path : str
        Directory to save the tiled TIFF files.
    T_start : int
        Starting time point index.
    T_end : int
        Ending time point index.
    C : int
        Channel index to extract.
    Z : int
        Z-depth index to extract.
    tile_size : int
        Size of each tile (tile_size x tile_size).
    overlap : int
        Number of pixels to overlap between tiles.
    mask_bounds : tuple, optional
        Tuple of (Y1, Y2, X1, X2) to crop the image before tiling.
    """
    dask_array, metadata = load_czi_image(image_path)



    for t in T_values:
        print(f"processing time point: {t}")
        img = dask_array[t , C, ...]
        img = img.compute()
        print(img.shape)

        # Find the valid region within the image
        if mask_bounds is not None:
            Y1, Y2, X1, X2 = mask_bounds
            print(f"applying mask bounds - Y:({Y1},{Y2}) X:({X1},{X2})")
            img = img[:, Y1:Y2 , X1:X2]
        else:
            img = img[...]
            #Z1, Y1, X1, Z2, Y2, X2 = find_maximimal_deskewed_region(img, Z1=Z_values[0], Z2=Z_values[1])
            #print(f"valid region - Z:({Z1},{Z2}) Y:({Y1},{Y2}) X:({X1},{X2})")
            #img = img[:, Y1:Y2 , X1:X2]

        print(img.shape)
        for z in Z_values:
            print(f"processing z slice: {z}")
            img_slice = img[z , :, :]
            print(img_slice.shape)
            tile_parent = f"{output_path}/T{t}_C{C}_Z{z}/"
            os.makedirs(tile_parent, exist_ok=True)
            tile_path = f"{tile_parent}/tile_T{t}_C{C}_Z{z}"
            tiles = tile_2D_image(img_slice, tile_size, overlap , output_path=tile_path)
    return

def find_maximimal_deskewed_region(image, Z1 , Z2):
    """
    Finds the rectangular region within a deskewed image that contains valid data (non-zero pixels).
    Parameters:
    ----------- 
    image : numpy.ndarray
        3D deskewed image array with shape (Z, Y, X).
    Z1 : int
        Starting Z index for the region of interest.    
    Z2 : int
        Ending Z index for the region of interest.
    Returns:
    --------
    Tuple[int, int, int, int]
        (Z1, Y1, X1, Z2, Y2, X2) coordinates of the valid region.

    """
    Z, Y, X = image.shape

    # Initialize boundaries
    Y1, X1 = Y, X
    Y2, X2 = 0, 0

    for z in (Z1 , Z2):
        slice_ = image[z , ...]
        ys, xs = np.where(slice_ > 0)

        if ys.size > 0 and xs.size > 0:
            Y1 = min(Y1, ys.min())
            Y2 = max(Y2, ys.max())
            X1 = min(X1, xs.min())
            X2 = max(X2, xs.max())

    return Z1, Y1, X1, Z2, Y2, X2

def load_tile_czi_timelapse(image_path , output_path , T_values , C , Z_values , Y_range , tile_size , overlap, mask_bounds=None):
    """
    Load a time-lapse CZI image, extract tiles for specified time points, and save them as TIFF files.

    Parameters:
    -----------
    image_path : str
        Path to the CZI file.
    output_path : str
        Directory to save the tiled TIFF files.
    T_start : int
        Starting time point index.
    T_end : int
        Ending time point index.
    C : int
        Channel index to extract.
    Z : int
        Z-depth index to extract.
    tile_size : int
        Size of each tile (tile_size x tile_size).
    overlap : int
        Number of pixels to overlap between tiles.
    mask_bounds : tuple, optional
        Tuple of (Y1, Y2, X1, X2) to crop the image before tiling.
    """
    dask_array, metadata = load_czi_image(image_path)



    for t in T_values:
        print(f"processing time point: {t}")
        img = dask_array[t , C, Z_values[0]:Z_values[1], ...]
        img = img.compute()
        print(img.shape)

        if Y_range is not None:
            print(f"Y_range specified: {Y_range}")
            Y1 = Y_range[0]
            Y2 = Y_range[1] if Y_range[1] is not None else img.shape[-2]
            img = img[..., Y1:Y2 , :]
            print(f"Adjusted image shape after applying Y_range: {img.shape}")

        # Find the valid region within the image
        if mask_bounds is not None:
            Y1, Y2, X1, X2 = mask_bounds
            print(f"applying mask bounds - Y:({Y1},{Y2}) X:({X1},{X2})")
            img = img[:, Y1:Y2 , X1:X2]
        else:
            img = img[...]
            #Z1, Y1, X1, Z2, Y2, X2 = find_maximimal_deskewed_region(img, Z1=Z_values[0], Z2=Z_values[1])
            #print(f"valid region - Z:({Z1},{Z2}) Y:({Y1},{Y2}) X:({X1},{X2})")
            #img = img[:, Y1:Y2 , X1:X2]

        print(img.shape)
        tile_parent = f"{output_path}" #/T{t}_C{C}/"
        os.makedirs(tile_parent, exist_ok=True)
        tile_path = f"{output_path}/tile_T{t}_C{C}"
        tiles = tile_2D_image(img, tile_size, Z_values[0], overlap , output_path=tile_path)
    return

def load_tile_tif_timelapse(image_path , output_path , T_values , C , Z_values , tile_size , overlap, Y_range=None, mask_bounds=None):
    """
    Load a time-lapse TIFF image, extract tiles for specified time points, and save them as TIFF files.

    Parameters:
    -----------
    image_path : str
        Path to the TIFF file.
    output_path : str
        Directory to save the tiled TIFF files.
    T_start : int
        Starting time point index.
    T_end : int
        Ending time point index.
    C : int
        Channel index to extract.
    Z : int
        Z-depth index to extract.
    tile_size : int
        Size of each tile (tile_size x tile_size).
    overlap : int
        Number of pixels to overlap between tiles.
    mask_bounds : tuple, optional
        Tuple of (Y1, Y2, X1, X2) to crop the image before tiling.
    """


    img = tiff.imread(image_path)
    print(f"Loaded image shape: {img.shape}")

    if Y_range is not None:
        print(f"Y_range specified: {Y_range}")
        Y1 = Y_range[0]
        Y2 = Y_range[1] if Y_range[1] is not None else img.shape[-2]
        img = img[..., Y1:Y2 , :]
        print(f"Adjusted image shape after applying Y_range: {img.shape}")

    #check and adjust for channel dimension
    if len(img.shape) == 4:
        #assume shape is (T, C, Z, Y, X)
        pass
    elif len(img.shape) == 5:
        #assume shape is (T, C, Z, Y, X)
        img = img[:, C, ...]
        print(f"Adjusted image shape after selecting channel {C}: {img.shape}")

    for t in T_values:
        print(f"processing time point: {t}")
        if Z_values is not None:
            img_t = img[t , Z_values[0]:Z_values[1], ...]
        else:
            img_t = img[t, ...]
        print(img_t.shape)

        # Find the valid region within the image
        if mask_bounds is not None:
            Y1, Y2, X1, X2 = mask_bounds
            print(f"applying mask bounds - Y:({Y1},{Y2}) X:({X1},{X2})")
            img_t = img_t[:, Y1:Y2 , X1:X2]
        else:
            img_t = img_t[...]

        print(img_t.shape)
        tile_path = f"{output_path}/tile_T{t}_C{C}"
        tiles = tile_2D_image(img_t, tile_size, Z_values[0], overlap , output_path=tile_path)
    return

if __name__ == "__main__":

    
    path        = r'/users/kir-fritzsche/aif490/devel/raw_data/videos/b2-2a_2c_pos6-01_deskew_cgt.czi'
    output_path = r'/users/kir-fritzsche/aif490/devel/tissue_analysis/CARE/cycleCARE/data/node1_z220_256_cropped'
    T_values = range(0,80)
    C = 0
    Z_values = [220, 221]
    Y_range  = [550, None]
    tile_size = 256
    overlap = 16
    #mask_bounds = (186, 186+1860, 363, 363+2088)  # Y1, Y2, X1, X2
    load_tile_czi_timelapse(path , output_path , T_values , C , Z_values , Y_range , tile_size , overlap, mask_bounds=None)
    