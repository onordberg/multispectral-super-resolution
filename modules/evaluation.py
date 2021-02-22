import pandas as pd
import pathlib
import tensorflow as tf
import rasterio
import numpy as np
import shutil

from modules.tile_input_pipeline import GeotiffDataset
from modules.image_utils import *


def esrgan_evaluate(model, dataset, steps=4, per_image=True):
    results = {}
    step = 0
    for batch in dataset:
        if step == steps:
            break
        batch_size = batch[0].shape[0]

        # Do forward passes for every single image, i.e. batch_size=1, instead of the whole mini-batch:
        if per_image:
            for i in range(batch_size):
                if step == steps:
                    break
                x, y = tf.expand_dims(batch[0][i], 0), tf.expand_dims(batch[1][i], 0)
                res = model.test_on_batch(x=x, y=y,
                                          reset_metrics=True, return_dict=True)
                for metric, value in res.items():
                    if metric not in results:
                        results[metric] = []
                    results[metric].append(value)
                step += 1

        # Do forward passes for the whole mini-batch:
        else:
            x, y = batch
            res = model.test_on_batch(x=x, y=y,
                                      reset_metrics=True, return_dict=True)
            for metric, value in res.items():
                if metric not in results:
                    results[metric] = []
                results[metric].append(value)
            step += 1

    return pd.DataFrame.from_dict(results)


def esrgan_epoch_evaluator(esrgan_model,
                           model_weights_dir,
                           model_weight_prefix,
                           dataset,
                           n_epochs,
                           first_epoch,
                           steps_per_epoch,
                           csv_dir,
                           per_image=True):
    if isinstance(model_weights_dir, str):
        model_weights_dir = pathlib.Path(model_weights_dir)
    if isinstance(csv_dir, str):
        csv_dir = pathlib.Path(csv_dir)
    csv_dir.mkdir(exist_ok=True, parents=True)

    if first_epoch == 1:
        n_epochs += first_epoch

    for i in range(first_epoch, n_epochs):
        filename = model_weight_prefix + str(i).zfill(2)
        model_weights_path = model_weights_dir.joinpath(filename + '.h5')
        esrgan_model.G.load_weights(model_weights_path)
        print('Start evaluation of epoch', i, ', model weights', model_weights_path)
        results = esrgan_evaluate(esrgan_model, dataset, steps=steps_per_epoch, per_image=per_image)
        csv_path = csv_dir.joinpath(filename + '.csv')
        results.to_csv(csv_path)
        print('Saved evaluation csv for epoch', i, '@', csv_path)


def esrgan_predict(model,
                   ms_img_path,
                   result_dir,
                   sensor,
                   band_indices,
                   mean_correction,
                   pre_or_gan,
                   sr_factor=4,
                   copy_pan_img=False,
                   pan_img_path=None,
                   output_dtype='uint16'):
    if isinstance(ms_img_path, str):
        ms_img_path = pathlib.Path(ms_img_path)
    if isinstance(result_dir, str):
        result_dir = pathlib.Path(result_dir)
    if copy_pan_img and isinstance(pan_img_path, str):
        pan_img_path = pathlib.Path(pan_img_path)
    filename = ms_img_path.stem
    result_dir.mkdir(parents=True, exist_ok=True)
    # copy ms image file to the results dir for easier comparison
    shutil.copy2(src=ms_img_path, dst=result_dir.joinpath(filename + '-ms.tif'))
    if copy_pan_img:
        if not isinstance(pan_img_path, pathlib.Path):
            pan_img_path = ms_img_path.parents[1].joinpath('pan').joinpath(filename + '.tif')
        shutil.copy2(src=pan_img_path, dst=result_dir.joinpath(filename + '-pan.tif'))
    print(pan_img_path)

    # Grab the geo-profile from the ms file. Needed to be able to write the sr file to disk.
    # Preserves georeference information
    with rasterio.open(ms_img_path, 'r') as img_ds:
        out_profile = img_ds.profile
    out_profile['width'] = out_profile['width'] * sr_factor
    out_profile['height'] = out_profile['height'] * sr_factor
    out_profile['count'] = 1
    # scale image transform
    out_profile['transform'] = out_profile['transform'] * out_profile['transform'].scale(1/sr_factor, 1/sr_factor)

    ds = GeotiffDataset(tiles_path=ms_img_path,
                        batch_size=1,
                        ms_tile_shape=(None, None, None),
                        pan_tile_shape=(None, None, None),
                        sensor=sensor,
                        band_selection=band_indices,
                        mean_correction=mean_correction,
                        cache_memory=False,
                        cache_file=False,
                        repeat=False,
                        shuffle=False,
                        shuffle_buffer_size=0,
                        build=False)
    ms_img = ds.decode_geotiff(ms_img_path, ms_or_pan='ms', image_path_is_tensor=False)
    ms_img = np.expand_dims(ms_img, 0)
    sr_img = model.predict(ms_img)
    if output_dtype == 'uint16':
        sr_img = output_scaler(sr_img,
                               radius=1.0,
                               output_dtype='uint16',
                               uint_bit_depth=11,
                               mean_correction=isinstance(mean_correction, float),
                               mean=mean_correction)
    elif output_dtype == 'float32':
        out_profile['dtype'] = 'float32'
    else:
        raise ValueError('only output_dtype uint16 and float32 supported')
    sr_img = sr_img[0,:,:,:]  # remove batch dimension
    ndarray_to_geotiff(sr_img,
                       geotiff_path=result_dir.joinpath(filename + '-sr-' + pre_or_gan + '.tif'),
                       rasterio_profile=out_profile)
    return sr_img
