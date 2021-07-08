import numpy as np
import pathlib
import os
import rasterio
import rasterio.plot
import tensorflow as tf

from modules.image_utils import *


# TODO: Preserve georeference information through the process

class GeotiffDataset:
    def __init__(self,
                 tiles_path,
                 batch_size=16,
                 ms_tile_shape=(32, 32, 8),
                 pan_tile_shape=(128, 128, 1),
                 sensor='WV02',  # 'WV02', 'GE01', 'WV03_VNIR'
                 band_selection='all',  # (1, 2, 4, 6)
                 mean_correction=None,
                 cache_memory=True,
                 cache_file=None,
                 repeat=True,
                 shuffle=True,
                 shuffle_buffer_size=1000,
                 build=True,
                 include_file_paths=False,
                 augment_flip=False,
                 augment_rotate=False):
        self.tiles_path = tiles_path
        self.batch_size = batch_size
        self.ms_tile_shape = ms_tile_shape
        self.pan_tile_shape = pan_tile_shape
        self.sensor = sensor
        self.band_selection = band_selection
        self.mean_correction = mean_correction
        self.cache_memory = cache_memory
        self.cache_file = cache_file
        self.repeat = repeat
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.include_file_paths = include_file_paths
        self.augment_flip = augment_flip
        self.augment_rotate = augment_rotate

        if isinstance(self.band_selection, tuple) or isinstance(self.band_selection, list):
            self.band_selection_bool = True
        elif self.band_selection == 'all':
            self.band_selection_bool = False

        if isinstance(self.mean_correction, float) or isinstance(self.mean_correction, int):
            self.mean_correction_bool = True
        else:
            self.mean_correction_bool = False

        if build:
            self.dataset = self.build_dataset()

    def get_dataset(self):
        return self.dataset

    def build_dataset(self):
        # Find and list all MS patches in the specified directory
        file_pattern = str(pathlib.Path(self.tiles_path).joinpath(
            self.sensor + '*', 'ms*', '*.tif'))
        if self.shuffle:
            # Shuffle the list of files during training
            ds = tf.data.Dataset.list_files(file_pattern, shuffle=True)
        else:
            # Option to not shuffle during inference
            ds = tf.data.Dataset.list_files(file_pattern, shuffle=False)

        # Turn list of files into paired arrays of MS and PAN patches
        # This mapping calls several methods that read GeoTIFFs, select MS bands,
        # scale the pixels from int16 to float32 and massage the arrays in different ways
        # to comply with tf.data shape requirements
        ds = ds.map(self.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # File caching. Turning on enables only performing time consuming above operations once.
        if isinstance(self.cache_file, str):
            ds = ds.cache(self.cache_file)
        # Memory caching (both file and memory can be combined).
        # System RAM limits the size of the memory cache.
        # The size is defined by the shuffle buffer size (5 lines down)
        if self.cache_memory:
            ds = ds.cache()
        # Shuffle in memory. Necessary during training.
        if self.shuffle:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer_size)

        # Repeat forever. Necessary if epochs during training is set to something else
        # than the whole training set
        if self.repeat:
            ds = ds.repeat()

        # Augmentation (after caching). Only using highly optimized tf functions.
        if self.augment_flip:
            ds = ds.map(self.random_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.random_rotate:
            ds = ds.map(self.random_rotate, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Concatenate into a mini-batch
        ds = ds.batch(self.batch_size)

        # Prefetch lets the dataset fetch batches in the background using CPU during training.
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    def decode_geotiff(self, image_path, ms_or_pan, image_path_is_tensor=True):
        if image_path_is_tensor:
            image_path = pathlib.Path(image_path.numpy().decode())
        with rasterio.open(image_path) as src:
            img = src.read()
        img = rasterio.plot.reshape_as_image(img)  # from channels first to channels last

        if self.band_selection_bool and ms_or_pan == 'ms':
            img = select_bands(img, self.band_selection)

        img = input_scaler(img,
                           radius=1.0,
                           output_dtype='float32',
                           uint_bit_depth=11,
                           mean_correction=self.mean_correction_bool,
                           mean=self.mean_correction,
                           print_ranges=False,
                           return_range_only=False)
        return img

    def random_flip(self, ms, pan):
        if self.include_file_paths:
            ms_tile_path, ms_img = ms
            pan_tile_path, pan_img = pan
        else:
            ms_img = ms
            pan_img = pan
        flip_left_right = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
        flip_up_down = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)

        if flip_left_right == 1:
            ms_img = tf.image.flip_left_right(ms_img)
            pan_img = tf.image.flip_left_right(pan_img)
        if flip_up_down == 1:
            ms_img = tf.image.flip_up_down(ms_img)
            pan_img = tf.image.flip_up_down(pan_img)
        if self.include_file_paths:
            return (ms_tile_path, ms_img), (pan_tile_path, pan_img)
        else:
            return ms_img, pan_img

    def random_rotate(self, ms, pan):
        if self.include_file_paths:
            ms_tile_path, ms_img = ms
            pan_tile_path, pan_img = pan
        else:
            ms_img = ms
            pan_img = pan
        rotation = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        ms_img = tf.image.rot90(ms_img, k=rotation)
        pan_img = tf.image.rot90(pan_img, k=rotation)
        if self.include_file_paths:
            return (ms_tile_path, ms_img), (pan_tile_path, pan_img)
        else:
            return ms_img, pan_img

    def process_path(self, ms_tile_path):
        # img_string_UID = tf.strings.split(ms_tile_path, os.sep)[-3]
        # tile_UID = tf.strings.split(tf.strings.split(ms_tile_path, os.sep)[-1], '.')[0]

        ms_img = tf.py_function(self.decode_geotiff, [ms_tile_path, 'ms'], [tf.float32], name='decode_geotiff_ms')
        pan_tile_path = tf.strings.regex_replace(ms_tile_path, '\\\\ms\\\\', '\\\\pan\\\\')
        pan_img = tf.py_function(self.decode_geotiff, [pan_tile_path, 'pan'], [tf.float32], name='decode_geotiff_pan')

        # Removing first axis as this will create problems when batching later
        ms_img = tf.squeeze(ms_img, [0])
        pan_img = tf.squeeze(pan_img, [0])

        if self.include_file_paths:
            return (ms_tile_path, ms_img), (pan_tile_path, pan_img)
        else:
            return ms_img, pan_img

    # # https://www.tensorflow.org/tutorials/load_data/images
    # def prepare_for_training(self, ds):
    #     # File caching
    #     if isinstance(self.cache_file, str):
    #         ds = ds.cache(self.cache_file)
    #     # Memory caching (both file and memory can be combined)
    #     if self.cache_memory:
    #         ds = ds.cache()
    #     if self.shuffle:
    #         ds = ds.shuffle(buffer_size=self.shuffle_buffer_size)
    #     # Repeat forever
    #     if self.repeat:
    #         ds = ds.repeat()
    #     ds = ds.batch(self.batch_size)
    #     # `prefetch` lets the dataset fetch batches in the background while the model is training.
    #     ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    #     return ds

    def get_scaler_output_range(self, print_ranges=False):
        dummy_arr = np.zeros(1)
        output_range = input_scaler(dummy_arr,
                                    radius=1.0,
                                    output_dtype='float32',
                                    uint_bit_depth=11,
                                    mean_correction=self.mean_correction_bool,
                                    mean=self.mean_correction,
                                    print_ranges=print_ranges,
                                    return_range_only=True)
        return output_range
