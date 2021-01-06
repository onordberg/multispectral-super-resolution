import tensorflow as tf
import datetime
import pathlib


class EsrganLogger:
    def __init__(self,
                 model_name,
                 tensorboard=True,
                 tensorboard_log_dir='logs/tb/',
                 model_save=True,
                 model_save_dir='logs/models/'
                 ):
        self.model_name = model_name
        if tensorboard:
            self.tensorboard_log_dir = tensorboard_log_dir
            self.build_tb_callback()
        if model_save:
            self.model_save_dir = model_save_dir
        self.callbacks = []

    def build_tb_callback(self):
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        log_subdir = pathlib.Path(self.tensorboard_log_dir).joinpath(self.model_name + '_' + timestamp)
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_subdir,
            histogram_freq=0,
            write_graph=False,
            write_images=True,
            update_freq='epoch',
            profile_batch=False,
            embeddings_freq=0,
            embeddings_metadata=None)
        self.callbacks.append(tb_callback)

    def get_callbacks(self):
        return self.callbacks
