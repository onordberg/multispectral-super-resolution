import numpy as np
import random
import pathlib
import pandas as pd
import rasterio
import rasterio.windows
import geopandas
from collections import Counter

from modules.helpers import *
from modules.cloudsea_classifier import *


def resize_sat_img(src, rescale_factor=(1.0, 1.0), resampling='nearest'):
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

    img = src.read(out_shape=(count, height, width),
                   resampling=resampling)

    t = src.transform

    # rescale the metadata 
    # https://gis.stackexchange.com/questions/329434/creating-an-in-memory-rasterio-dataset-from-numpy-array/329439#329439
    transform = rasterio.Affine(t.a / rescale_factor[1], t.b, t.c, t.d, t.e / rescale_factor[0], t.f)
    return img, transform


def resize_sat_img_to_new_pixel_size(row, save_dir, new_pixel_size_pan=(1.0, 1.0),
                                     sr_factor=4, resampling='nearest'):
    image_string_UID = get_string_uid(row, row['int_uid'])
    print(image_string_UID)
    image_dir = pathlib.Path(save_dir, image_string_UID)
    image_dir.mkdir(exist_ok=True)
    ms_dir = pathlib.Path(image_dir, 'ms')
    ms_dir.mkdir(exist_ok=True)
    pan_dir = pathlib.Path(image_dir, 'pan')
    pan_dir.mkdir(exist_ok=True)

    # For resizing of image to new pixel size only the pan pixel size is used directly,
    # while the new ms pixel size is calculated from pan and the sr_factor.
    # This is done to avoid rounding errors and prioritize sr_factor to be exact
    old_pan_pixel_size = (row['pan_pixelHeight'], row['pan_pixelWidth'])
    old_ms_pixel_size = (row['ms_pixelHeight'], row['ms_pixelWidth'])
    # assert pan_pixel_size[0] == pan_pixel_size[1]
    pan_resize_factor = (old_pan_pixel_size[0] / new_pixel_size_pan[0],
                         old_pan_pixel_size[1] / new_pixel_size_pan[1])
    # print(pan_resize_factor)
    ms_resize_factor = (old_ms_pixel_size[0] / (sr_factor * new_pixel_size_pan[0]),
                        old_ms_pixel_size[1] / (sr_factor * new_pixel_size_pan[1]))
    # print(old_ms_pixel_size[0]/old_pan_pixel_size[0])
    # print(old_ms_pixel_size[0]*ms_resize_factor[0]/(old_pan_pixel_size[0]*pan_resize_factor[0]))

    with rasterio.open(row['ms_tif_path'], 'r') as ms_src, rasterio.open(row['pan_tif_path'], 'r') as pan_src:
        print('Dimensions before resize', (ms_src.count, ms_src.shape[0], ms_src.shape[1]),
              (pan_src.count, pan_src.shape[0], pan_src.shape[1]))
        print('Resize by factors (height, width):')
        print('pan', pan_resize_factor, ', ms:', ms_resize_factor)
        ms_img, ms_transform = resize_sat_img(ms_src,
                                              rescale_factor=ms_resize_factor,
                                              resampling=resampling)
        pan_img, pan_transform = resize_sat_img(pan_src,
                                                rescale_factor=pan_resize_factor,
                                                resampling=resampling)
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


def resize_all_sat_imgs_to_new_pixel_size(meta, save_dir, new_pixel_size_pan=(1.0, 1.0),
                                          sr_factor=4, resampling='nearest'):
    meta.apply(resize_sat_img_to_new_pixel_size, axis=1,
               save_dir=save_dir,
               new_pixel_size_pan=new_pixel_size_pan,
               sr_factor=sr_factor, resampling=resampling)


def allocate_tiles_by_expected(meta,
                               pan_tile_size=128,
                               tiles_per_m2=1.0,
                               override_pan_pixel_size=False,
                               by_partition=False,
                               tiles_per_m2_train_val_test=(1.0, 1.0, 1.0),
                               pan_tile_size_train_val_test=(128, 128, 128),
                               new_column_name='n_tiles'):
    counts_df = pd.DataFrame(index=meta.index)
    counts_df[new_column_name] = 0

    if not override_pan_pixel_size:
        # Check that all pixel sizes are square. Anything else is very unusual in UTM sat images.
        pd.testing.assert_series_equal(meta['pan_pixelHeight'], meta['pan_pixelWidth'], check_names=False)
        # Since all are square the following simplification is OK
        counts_df['pan_pixel_size'] = meta.loc[:, 'pan_pixelHeight']
    else:
        if not isinstance(override_pan_pixel_size, float):
            raise ValueError('override_pan_pixel_size must either be False or a float, not VALUE:'
                             + str(override_pan_pixel_size) + ' TYPE: ' + str(type(override_pan_pixel_size)))
        if override_pan_pixel_size < 0.0:
            raise ValueError('override_pan_pixel_size must be positive')
        counts_df['pan_pixel_size'] = override_pan_pixel_size

    if by_partition:
        counts_df['train_val_test'] = meta.loc[:, 'train_val_test']
        for i, p in enumerate(['train', 'val', 'test']):
            meta_p = meta.loc[meta['train_val_test'] == p, :]
            counts_df_p = counts_df.loc[counts_df['train_val_test'] == p, :].copy()

            # The actual calculation of number of tiles
            counts_df_p[new_column_name] = ((tiles_per_m2_train_val_test[i] * meta_p['area_m2'])
                                            / (counts_df['pan_pixel_size'] ** 2 * pan_tile_size_train_val_test[i] ** 2))
            counts_df.update(counts_df_p)

    else:
        # The actual calculation of number of tiles
        counts_df[new_column_name] = ((tiles_per_m2 * meta['area_m2'])
                                      / (counts_df['pan_pixel_size'] ** 2 * pan_tile_size ** 2))

    meta[new_column_name] = counts_df[new_column_name].astype(int)
    return meta


def allocate_tiles_by_fixed_n_tiles(meta, by_partition=True, n_tiles_train=0, n_tiles_val=0, n_tiles_test=0,
                                    n_tiles_total=None, new_column_name='n_tiles'):
    # n_tiles_total only to be used when by_partition=False
    counts_df = pd.DataFrame(index=meta.index)
    counts_df[new_column_name] = 0

    if by_partition and n_tiles_total is not None:
        raise ValueError('If by_partition=True, n_tiles_total must not be specified.')
    if not by_partition and n_tiles_total is None:
        raise ValueError('If by_partition=False, n_tiles_total must be specified.')

    # when allocating by partition they are looped through
    if by_partition:
        for p in ['train', 'val', 'test']:
            if p == 'train' and n_tiles_train > 0:
                n_tiles = n_tiles_train
            elif p == 'val' and n_tiles_val > 0:
                n_tiles = n_tiles_val
            elif p == 'test' and n_tiles_test > 0:
                n_tiles = n_tiles_test
            else:
                continue  # If n_tiles_part is 0

            # list of image names (index names) that will be allocated
            image_names = list(meta[meta['train_val_test'] == p]['area_ratio'].index)
            # list of weights based on the area_ratio of an image
            # (small images will get less tiles allocated than large images)
            image_area_weights = list(meta[meta['train_val_test'] == p]['area_ratio'].values)

            # the actual sampling
            sampling = random.choices(image_names, weights=image_area_weights, k=n_tiles)

            # collecting the samples in a dataframe and updating for every partition
            counts = pd.DataFrame.from_dict(dict(Counter(sampling)),
                                            orient='index',
                                            columns=[new_column_name])
            print('Allocated', int(sum(counts[new_column_name])), 'tiles across the',
                  meta['train_val_test'].value_counts()[p],
                  'images in the', p, 'partition.')
            counts_df.update(counts)

    # when allocating without partition it is straight-forward
    else:
        image_names = list(meta['area_ratio'].index)
        image_area_weights = list(meta['area_ratio'].values)
        sampling = random.choices(image_names, weights=image_area_weights, k=n_tiles_total)
        counts = pd.DataFrame.from_dict(dict(Counter(sampling)),
                                        orient='index',
                                        columns=[new_column_name])
        counts_df.update(counts)
        print('Allocated', int(sum(counts_df[new_column_name])), 'tiles across the', count_images(meta), 'images')

    # "merge" tile counts dataframe with main metadata dataframe
    meta[new_column_name] = counts_df
    meta[new_column_name] = meta[new_column_name].astype('int32')
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

    # print('img_shape', img_shape)
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
    # print(lr_box)
    # print(resize_factor)
    # print(hr_channels)
    hr_box[:4] = lr_box[:4] * resize_factor
    hr_box[4] = hr_channels
    return hr_box


def is_border_pixel_in_image(img):
    if 0 in img:
        return True
    else:
        return False


def is_img_all_cloud_or_sea(img, model, pred_cutoff=0.95):
    # Since classification will be done on single images .predict is NOT recommended, but rather directly calling
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
    cloud_sea_prob = model(img, training=False)
    # cloud_sea_prob = model.predict(img)
    # print(cloud_sea_prob.numpy()))
    if cloud_sea_prob > pred_cutoff:
        return True
    else:
        return False


def keep_cloud_sea(keep_rate):
    return random.choices(population=(False, True), weights=(1-keep_rate, keep_rate), k=1)[0]


def generate_tiles(row,
                   save_dir,
                   ms_tile_size=32,
                   sr_factor=4,
                   by_partition=True,
                   ms_tile_size_train_val_test=(32, 32, 32),
                   cloud_sea_removal=False,
                   cloud_sea_model=None,
                   cloud_sea_pred_cutoff=0.95,
                   cloud_sea_keep_rate=0.1):
    if isinstance(save_dir, str):
        save_dir = pathlib.Path(save_dir)

    image_string_UID = get_string_uid(row, row['int_uid'])
    if by_partition:
        save_dir = pathlib.Path(save_dir, row['train_val_test'])
        partition = row['train_val_test']
        if partition == 'train':
            ms_tile_size = ms_tile_size_train_val_test[0]
        elif partition == 'val':
            ms_tile_size = ms_tile_size_train_val_test[1]
        elif partition == 'test':
            ms_tile_size = ms_tile_size_train_val_test[2]
        else:
            raise ValueError(
                'Partition string in metadata dataframe is neither "train", "val" or "test". Value:', partition)
        print(partition, 'set -', 'From image', image_string_UID, '- Generating', row['n_tiles'], 'tiles')
    else:
        print('From image', image_string_UID, '- Generating', row['n_tiles'], 'tiles')
    save_dir.mkdir(parents=True, exist_ok=True)

    image_dir = pathlib.Path(save_dir, image_string_UID)
    image_dir.mkdir(exist_ok=True)
    ms_dir = pathlib.Path(image_dir, 'ms')
    ms_dir.mkdir(exist_ok=True)
    pan_dir = pathlib.Path(image_dir, 'pan')
    pan_dir.mkdir(exist_ok=True)

    count_border_pixel = 0
    count_cloud_sea = 0
    count_cloud_sea_keep = 0
    tile_count = 0

    with rasterio.open(row['ms_tif_path'], 'r') as ms_src, rasterio.open(row['pan_tif_path'], 'r') as pan_src:
        print('Shapes', (ms_src.count, ms_src.shape[0], ms_src.shape[1]),
              (pan_src.count, pan_src.shape[0], pan_src.shape[1]))

        for i in range(row['n_tiles']):
            tile_filename = str(str(i).zfill(5) + '.tif')
            while True:
                ms_shape = [ms_src.shape[0], ms_src.shape[1], ms_src.count]
                ms_box = np.array(get_random_box(ms_shape, (ms_tile_size, ms_tile_size)))  # square tile h == w

                # Box list on the form [upper_left_y, upper_left_x, height, width, channels]
                ms_win = rasterio.windows.Window(ms_box[1], ms_box[0], ms_box[3], ms_box[2])
                ms_tile = ms_src.read(window=ms_win)
                ms_win_transform = ms_src.window_transform(ms_win)

                pan_box = get_hr_box(ms_box, sr_factor, pan_src.count)

                pan_win = rasterio.windows.Window(pan_box[1], pan_box[0], pan_box[3], pan_box[2])
                pan_tile = pan_src.read(window=pan_win)
                pan_win_transform = pan_src.window_transform(pan_win)

                write_to_disk = False
                resample = False
                if is_border_pixel_in_image(ms_tile):
                    count_border_pixel += 1
                    resample = True
                    write_to_disk = False
                elif is_border_pixel_in_image(pan_tile):  # elif so that the tile is not counted twice
                    count_border_pixel += 1
                    resample = True
                    write_to_disk = False

                if not resample:
                    if cloud_sea_removal:
                        if is_img_all_cloud_or_sea(cloudsea_preprocess(pan_tile),
                                                   cloud_sea_model,
                                                   pred_cutoff=cloud_sea_pred_cutoff):
                            count_cloud_sea += 1
                            # Do we keep or discard the tile?
                            if isinstance(cloud_sea_keep_rate, float) and (0 <= cloud_sea_keep_rate <= 1):
                                if keep_cloud_sea(keep_rate=cloud_sea_keep_rate):
                                    count_cloud_sea_keep += 1
                                    write_to_disk = True
                                    resample = False
                                else:
                                    write_to_disk = False
                                    resample = False  # equivalent to discarding of tile
                            else:
                                raise ValueError('cloud_sea_keep_rate must be float in range [0, 1]')
                        else:  # when the tile has been classified as NOT being only cloud and/or sea
                            write_to_disk = True
                            resample = False
                    else:  # the case where no cloud_sea_removal is done
                        write_to_disk = True
                        resample = False

                if write_to_disk:
                    with rasterio.open(
                            pathlib.Path(ms_dir, tile_filename),
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
                            pathlib.Path(pan_dir, tile_filename),
                            'w',
                            driver='GTiff',
                            width=pan_box[3],
                            height=pan_box[2],
                            count=pan_box[4],
                            dtype=pan_tile.dtype,
                            crs=pan_src.crs,
                            transform=pan_win_transform) as pan_dst:
                        pan_dst.write(pan_tile)
                    tile_count += 1
                    break  # Tile written to disk -> Stop the loop
                if not resample:  # equivalent to discarding of tile
                    break
    print('Number of tiles discarded and resampled due to border pixels:', count_border_pixel)
    if cloud_sea_removal:
        print('Number of tiles classified as being only clouds or sea:', count_cloud_sea)
        print('Number of tiles kept despite being only clouds or sea:', count_cloud_sea_keep)
    print('Tiles actually generated to disk:', tile_count)

    # Storing both allocated and actual number of tiles in the meta dataframe
    row['n_tiles_allocated'] = row['n_tiles']
    row['n_tiles'] = tile_count
    print()
    return row


def generate_all_tiles(meta, save_dir,
                       ms_tile_size=32,
                       sr_factor=4,
                       by_partition=True,
                       ms_tile_size_train_val_test=(32, 32, 32),
                       cloud_sea_removal=True,
                       cloud_sea_weights_path=None,
                       cloud_sea_pred_cutoff=0.95,
                       cloud_sea_keep_rate=0.1,
                       save_meta_to_disk=True):
    cloud_sea_model = None
    if cloud_sea_removal:
        CLOUD_SEA_INPUT_SIZE = 224
        cloud_sea_model = build_model(augment=False, input_shape=(CLOUD_SEA_INPUT_SIZE, CLOUD_SEA_INPUT_SIZE, 1))
        if cloud_sea_weights_path is None:
            raise ValueError('Provide path to cloud/sea classifier model weights.', cloud_sea_weights_path, 'given.')
        if isinstance(cloud_sea_weights_path, str):
            cloud_sea_weights_path = pathlib.Path(cloud_sea_weights_path)
            cloud_sea_model.load_weights(cloud_sea_weights_path)

    if by_partition:
        n_tiles_train = count_tiles_in_partition(meta, train_val_test='train')
        n_tiles_val = count_tiles_in_partition(meta, train_val_test='val')
        n_tiles_test = count_tiles_in_partition(meta, train_val_test='test')
        print('Generating', n_tiles_train, 'training,', n_tiles_val, 'validation and', n_tiles_test, 'test tiles:')
    else:
        n_tiles = count_tiles(meta)
        print('Generating', n_tiles, 'without separating by train/val/test partition.')
    meta = meta.apply(generate_tiles, axis=1,
                      save_dir=save_dir,
                      ms_tile_size=ms_tile_size,
                      sr_factor=sr_factor,
                      by_partition=by_partition,
                      ms_tile_size_train_val_test=ms_tile_size_train_val_test,
                      cloud_sea_removal=cloud_sea_removal,
                      cloud_sea_model=cloud_sea_model,
                      cloud_sea_pred_cutoff=cloud_sea_pred_cutoff,
                      cloud_sea_keep_rate=cloud_sea_keep_rate)
    print('Tile generation finished')

    if save_meta_to_disk:
        save_meta_pickle_csv(meta, save_dir, 'metadata_tile_allocation', to_pickle=True, to_csv=True)
        print('Metadata dataframe saved as pickle and csv @', save_dir)
    return meta
