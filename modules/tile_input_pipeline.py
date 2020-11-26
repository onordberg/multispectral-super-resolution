import numpy as np
import pathlib
import os
import rasterio
import rasterio.plot
import tensorflow as tf

#TODO: Preserve filename through the process
#TODO: Preserve georeference information through the process

def wv02_imitate_ge01(img):
    img = np.take(img, IMITATION_BANDS, -1)
    return img

def decode_geotiff(image_path):
    image_path = pathlib.Path(image_path.numpy().decode())
    with rasterio.open(image_path) as src:
        img = src.read()
    img = rasterio.plot.reshape_as_image(img) # from channels first to channels last
    
    # Imitate GE01 sensor by dropping WV02 bands?
    if IMITATE_GE01:
        img = wv02_imitate_ge01(img)
    return img

def preprocess_images(img, ms_or_pan):
    if ms_or_pan == 'ms':
        h, w = MS_TILE_SHAPE[0], MS_TILE_SHAPE[1]
    elif ms_or_pan == 'pan':
        h, w = PAN_TILE_SHAPE[0], PAN_TILE_SHAPE[1]
        
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.reshape(img, [h, w, -1]) # To avoid issue with extra dimension
    return img

def process_path(ms_tile_path):
    img_string_UID = tf.strings.split(ms_tile_path, os.sep)[-3]
    tile_UID = tf.strings.split(tf.strings.split(ms_tile_path, os.sep)[-1], '.')[0]
    
    ms_img = tf.py_function(decode_geotiff, [ms_tile_path], [tf.int16], name = 'decode_geotiff_ms')
    pan_tile_path = tf.strings.regex_replace(ms_tile_path, '\\\\ms\\\\', '\\\\pan\\\\')
    pan_img = tf.py_function(decode_geotiff, [pan_tile_path], [tf.int16], name = 'decode_geotiff_pan')
    
    ms_img = preprocess_images(ms_img, 'ms')
    pan_img = preprocess_images(pan_img, 'pan')
    
    return ms_img, pan_img

# https://www.tensorflow.org/tutorials/load_data/images
def prepare_for_training(ds, batch_size, cache_memory=True, cache_file=None, shuffle_buffer_size=100):
    # File caching
    if isinstance(cache_file, str):
        ds = ds.cache(cache_file)
    # Memory caching (both can be combined)
    if cache_memory:
        ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(batch_size)
    
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds

def dataset_from_tif_tiles(tiles_path, batch_size,
                           ms_tile_shape=(32,32,8), pan_tile_shape=(128,128,1),
                           imitate_ge01=False, imitation_bands=(1, 2, 4, 6),
                           cache_memory=True, cache_file=None, shuffle_buffer_size = 1000):
    
    # MS_TILE_SHAPE = (HEIGHT, WIDTH, BANDS), PAN_TILE_SHAPE = (HEIGHT, WIDTH, BANDS)
    global MS_TILE_SHAPE
    global PAN_TILE_SHAPE
    global IMITATE_GE01
    global IMITATION_BANDS
    
    MS_TILE_SHAPE, PAN_TILE_SHAPE = ms_tile_shape, pan_tile_shape
    IMITATE_GE01, IMITATION_BANDS = imitate_ge01, imitation_bands
    
    ds = tf.data.Dataset.list_files(str(pathlib.Path(tiles_path)/'*/ms*.tif'))
    ds = ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds = prepare_for_training(ds, batch_size, cache_memory, cache_file, shuffle_buffer_size)
    return ds