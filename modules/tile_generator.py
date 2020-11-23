import numpy as np
import random
import pathlib
import pandas as pd
import rasterio
import geopandas
from collections import Counter

from modules.helpers import *


def resize_sat_img(src, rescale_factor = (1.0, 1.0), resampling = 'nearest'):
    # resample data to target shape
    height = int(src.height * rescale_factor[0])
    width = int(src.width * rescale_factor[1])
    count = src.count
    
    if resampling == 'nearest':
        resampling = rasterio.enums.Resampling.nearest
    elif resampling == 'bicubic':
        resampling = rasterio.enums.Resampling.cubic
    elif resampling == 'bilinear':
        resampling = rasterio.enums.Resampling.bilinear 
        
    img = src.read(out_shape = (count,
                  height,
                  width),
                  resampling = resampling)
                   
    t = src.transform

    # rescale the metadata 
    # https://gis.stackexchange.com/questions/329434/creating-an-in-memory-rasterio-dataset-from-numpy-array/329439#329439
    transform = rasterio.Affine(t.a / rescale_factor[1], t.b, t.c, t.d, t.e / rescale_factor[0], t.f)

    return img, transform

def resize_sat_img_to_new_pixel_size(row, save_dir, new_pixel_size_pan = (1.0, 1.0), 
                                     sr_factor = 4, resampling = 'nearest'):
    image_string_UID = get_string_uid(row, row['int_uid'])
    print(image_string_UID)
    image_dir = pathlib.Path(save_dir, image_string_UID)
    image_dir.mkdir(exist_ok = True)
    ms_dir = pathlib.Path(image_dir, 'ms')
    ms_dir.mkdir(exist_ok = True)
    pan_dir = pathlib.Path(image_dir, 'pan')
    pan_dir.mkdir(exist_ok = True)
    
    # For resizing of image to new pixel size only the pan pixel size is used directly,
    # while the new ms pixel size is calculated from pan and the sr_factor.
    # This is done to avoid rounding errors and prioritize sr_factor to be exact
    old_pan_pixel_size = (row['pan_pixelHeight'], row['pan_pixelWidth'])
    old_ms_pixel_size = (row['ms_pixelHeight'], row['ms_pixelWidth'])
    #assert pan_pixel_size[0] == pan_pixel_size[1] 
    pan_resize_factor = (old_pan_pixel_size[0]/new_pixel_size_pan[0], 
                         old_pan_pixel_size[1]/new_pixel_size_pan[1])
    #print(pan_resize_factor)
    ms_resize_factor = (old_ms_pixel_size[0] / (sr_factor*new_pixel_size_pan[0]), 
                        old_ms_pixel_size[1] / (sr_factor*new_pixel_size_pan[1]))
    #print(old_ms_pixel_size[0]/old_pan_pixel_size[0])
    #print(old_ms_pixel_size[0]*ms_resize_factor[0]/(old_pan_pixel_size[0]*pan_resize_factor[0]))
    
    with rasterio.open(row['ms_tif_path'], 'r') as ms_src, rasterio.open(row['pan_tif_path'], 'r') as pan_src:
        print('Dimensions before resize', (ms_src.count, ms_src.shape[0], ms_src.shape[1]), 
              (pan_src.count, pan_src.shape[0], pan_src.shape[1]))
        print('Resize by factors (height, width):')
        print('pan', pan_resize_factor, ', ms:', ms_resize_factor)
        ms_img, ms_transform = resize_sat_img(ms_src, 
                                              rescale_factor = ms_resize_factor, 
                                              resampling = resampling)
        pan_img, pan_transform = resize_sat_img(pan_src, 
                                                rescale_factor = pan_resize_factor, 
                                                resampling = resampling)
        print('Dimensions after resize', ms_img.shape, pan_img.shape)
        ms_path = pathlib.Path(ms_dir, str(image_string_UID + '.tif'))
        pan_path = pathlib.Path(pan_dir, str(image_string_UID + '.tif'))
        with rasterio.open(
                ms_path, 'w',
                driver='GTiff',
                width=ms_img.shape[2],
                height=ms_img.shape[1],
                count=ms_img.shape[0],
                dtype=ms_img.dtype,
                crs=ms_src.crs,
                transform=ms_transform) as ms_dst:
                    ms_dst.write(ms_img)
        with rasterio.open(
                pan_path, 'w',
                driver='GTiff',
                width=pan_img.shape[2],
                height=pan_img.shape[1],
                count=pan_img.shape[0],
                dtype=pan_img.dtype,
                crs=pan_src.crs,
                transform=pan_transform) as pan_dst:
                    pan_dst.write(pan_img)
    print()
    
    # Update paths to the new resized versions
    row['ms_tif_path'] = ms_path.absolute()
    row['pan_tif_path'] = pan_path.absolute()
    return row

def resize_all_sat_imgs_to_new_pixel_size(meta, save_dir, new_pixel_size_pan=(1.0,1.0), 
                                          sr_factor=4, resampling='nearest'):
    meta = meta.apply(resize_sat_img_to_new_pixel_size, axis = 1, 
                      save_dir = save_dir, 
                      new_pixel_size_pan = new_pixel_size_pan, 
                      sr_factor = sr_factor, resampling = resampling)

def allocating_tiles(meta, n_tiles_train, n_tiles_val, n_tiles_test):
    counts_df = pd.DataFrame(index=meta.index)
    counts_df['n_tiles'] = None

    for p in ['train', 'val', 'test']:
        if p == 'train' and n_tiles_train > 0:
            n_tiles = n_tiles_train
        elif p == 'val' and n_tiles_val > 0:
            n_tiles = n_tiles_val
        elif p == 'test' and n_tiles_test > 0:
            n_tiles = n_tiles_test
        else:
            continue # If n_tiles_part is 0
            
        l = list(meta[meta['train_val_test'] == p]['area_ratio'].index)
        w = list(meta[meta['train_val_test'] == p]['area_ratio'].values)

        sampling = random.choices(l, weights = w, k = n_tiles)
        print('Allocated', n_tiles, 'tiles across the', 
              meta['train_val_test'].value_counts()[p], 
              'images in the', p, 'partition.' )
        counts = pd.DataFrame.from_dict(dict(Counter(sampling)), 
                                        orient = 'index', 
                                        columns = ['n_tiles'])
        counts_df.update(counts)        
    
    meta['n_tiles'] = counts_df
    meta['n_tiles'] = meta['n_tiles'].astype('int32')
    return meta

def get_random_box(img_shape, crop_size):
    """Samples a smaller window from within the boundaries of a larger image

    Samples a window uniformly from a rectangular array defined 
    by its shape. Useful if you later want to use this window to 
    extract/crop from a larger image.

    Args:
        img_shape: shape of the larger image that window will be
            sampled from. 
            A list on the form [height, width, channels]
            Note: channels is required.
        crop_size: size of the smaller window
            list on the form [height, width]

    Returns:
        A list defining the window:
        [upper_left_y, upper_left_x, height, width, channels]
    """
    
    #print('img_shape', img_shape)
    maxval_y, maxval_x, channels = img_shape
    
    # Ensure that window does not go beyond image dimensions
    maxval_y -= crop_size[0]
    maxval_x -= crop_size[1]
    
    # Random sampling
    rng = np.random.default_rng()
    upper_left_yx = rng.integers(0, high=[maxval_y, maxval_x], dtype='int32')
    
    # Concatenating results
    box = [upper_left_yx[0], upper_left_yx[1], crop_size[0], crop_size[1], channels]
    return box

def get_hr_box(lr_box, resize_factor, hr_channels):
    hr_box = np.copy(lr_box)
    #print(lr_box)
    #print(resize_factor)
    #print(hr_channels)
    hr_box[:4] = lr_box[:4] * resize_factor
    hr_box[4] = hr_channels
    return hr_box

def is_border_pixel_in_image(img):
    if 0 in img:
        return True
    else:
        return False
    
def is_img_all_cloud_or_sea(img, sd_treshold):
    # Very simplified categorization at the moment
    if np.std(img) < sd_treshold:
        return True
    else:
        return False

def generate_tiles(row, save_dir, ms_height_width=(32,32), sr_factor=4, print_tile_info=False):
    image_string_UID = get_string_uid(row, row['int_uid'])
    print(row['train_val_test'], image_string_UID, '- Generating', row['n_tiles'], 'tiles')
    partition_dir = pathlib.Path(save_dir, row['train_val_test'])
    partition_dir.mkdir(exist_ok = True)
    image_dir = pathlib.Path(partition_dir, image_string_UID)
    image_dir.mkdir(exist_ok = True)
    ms_dir = pathlib.Path(image_dir, 'ms')
    ms_dir.mkdir(exist_ok = True)
    pan_dir = pathlib.Path(image_dir, 'pan')
    pan_dir.mkdir(exist_ok = True)
    
    discard_count_border_pixel = 0
    discard_count_cloud_sea = 0

    with rasterio.open(row['ms_tif_path'], 'r') as ms_src, rasterio.open(row['pan_tif_path'], 'r') as pan_src:
        print('Shapes', (ms_src.count, ms_src.shape[0], ms_src.shape[1]), 
              (pan_src.count, pan_src.shape[0], pan_src.shape[1]))

        for i in range(row['n_tiles']):
            #print(i)
            while True:
                ms_shape = [ms_src.shape[0], ms_src.shape[1], ms_src.count]
                ms_box = np.array(get_random_box(ms_shape, ms_height_width))
                #print('random_box', ms_box)
                # Box list on the form [upper_left_y, upper_left_x, height, width, channels]
                ms_win = rasterio.windows.Window(ms_box[1], ms_box[0], ms_box[3], ms_box[2])
                ms_tile = ms_src.read(window=ms_win)
                ms_win_transform = ms_src.window_transform(ms_win)
                #print(ms_tile.shape)
                
                pan_box = get_hr_box(ms_box, sr_factor, pan_src.count)

                pan_win = rasterio.windows.Window(pan_box[1], pan_box[0], pan_box[3], pan_box[2])
                pan_tile = pan_src.read(window=pan_win)
                pan_win_transform = pan_src.window_transform(pan_win)
                
                # CLOUD/SEA DETECTOR. TODO: REPLACE WITH PROPER DETECTOR
                is_ms_cloud_or_sea = is_img_all_cloud_or_sea(ms_tile, sd_treshold=90.0)
                is_pan_cloud_or_sea = is_img_all_cloud_or_sea(pan_tile, sd_treshold=10.0)
                
                if is_border_pixel_in_image(ms_tile):
                    if print_tile_info:
                        print('Border area detected in ms tile', i, 
                              'from image', image_string_UID)
                        print('Discarding current tile and resampling new tile')
                    discard_count_border_pixel += 1
                elif is_border_pixel_in_image(pan_tile):
                    if print_tile_info:
                        print('Border area detected in pan tile', i, 
                              'from image', image_string_UID)
                        print('Discarding current tile and resampling new tile')
                    discard_count_border_pixel += 1
                ########################################################################
                # CLOUD/SEA TILE REMOVAL!!!
                ########################################################################
                elif is_ms_cloud_or_sea or is_pan_cloud_or_sea:
                    if print_tile_info:
                        print('Tile', i, 'from image', image_string_UID, 
                              'is probably only sea or clouds')
                        print('ms cloud? ', is_ms_cloud_or_sea, ', pan cloud?', is_pan_cloud_or_sea)
                        print('Discarding current tile and resampling new tile')
                    discard_count_cloud_sea += 1
                ########################################################################    
                else:
                    break
    
            with rasterio.open(
                pathlib.Path(ms_dir, str(str(i).zfill(5) + '.tif')),
                'w',
                driver='GTiff',
                width=ms_box[3],
                height=ms_box[2],
                count=ms_box[4],
                dtype=ms_tile.dtype,
                crs=ms_src.crs,
                transform=ms_win_transform) as ms_dst:
                    ms_dst.write(ms_tile)
                
            with rasterio.open(
                pathlib.Path(pan_dir, str(str(i).zfill(5) + '.tif')),
                'w',
                driver='GTiff',
                width=pan_box[3],
                height=pan_box[2],
                count=pan_box[4],
                dtype=pan_tile.dtype,
                crs=pan_src.crs,
                transform=pan_win_transform) as pan_dst:
                    pan_dst.write(pan_tile)
    print('Number of tiles discarded due to border pixels:', discard_count_border_pixel)
    print('Number of tiles discarded due to only clouds or sea:', discard_count_cloud_sea)
    print()

def generate_all_tiles(meta, save_dir):
    n_tiles_train = count_images_in_partition(meta, train_val_test='train')
    n_tiles_val = count_images_in_partition(meta, train_val_test='val')
    n_tiles_test = count_images_in_partition(meta, train_val_test='test')
    
    print('Generating', n_tiles_train, 'training,', 
          n_tiles_val, 'validation and', n_tiles_test, 'test tiles:')
    meta.apply(generate_tiles, axis = 1, save_dir = save_dir)
    print('Tile generation finished')
    
    