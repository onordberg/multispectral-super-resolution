import pandas as pd
import pathlib
import tensorflow as tf
import numpy as np


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
