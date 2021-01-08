import tensorflow as tf
import datetime
import pathlib

from modules.image_utils import *


class EsrganLogger:
    def __init__(self,
                 model_name,
                 log_tensorboard=True,
                 tensorboard_logs_dir='logs/tb/',
                 save_models=True,
                 models_save_dir='logs/models/',
                 save_weights_only=True,
                 log_train_images=False,
                 n_train_image_batches=1,
                 train_image_dataset=None,
                 log_val_images=False,
                 n_val_image_batches=1,
                 val_image_dataset=None,  # This can also be dict with different sensors
                 model=None
                 ):
        self.callbacks = []
        self.model_name = model_name
        if save_models:
            self.model_save_dir = models_save_dir
            if isinstance(self.model_save_dir, str):
                self.model_save_dir = pathlib.Path(self.model_save_dir)
            self.save_weights_only = save_weights_only
            self.build_checkpoint_callback()
        if log_tensorboard:
            self.tensorboard_logs_dir = tensorboard_logs_dir
            if isinstance(self.tensorboard_logs_dir, str):
                self.tensorboard_logs_dir = pathlib.Path(self.tensorboard_logs_dir)
            self.tensorboard_logs_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir = self.build_tb_callback()
            if log_train_images or log_val_images:
                self.model = model
            if log_train_images:
                self.ds_train = train_image_dataset
                self.n_train_image_batches = n_train_image_batches
                self.n_train_images = 0
                self.train_file_writer, self.train_image_batches = self.build_train_image_logger()
                self.callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_begin=self.build_train_image_logger))

    def build_train_image_logger(self):
        train_file_writer = tf.summary.create_file_writer(self.log_dir.joinpath('train').as_posix())
        train_image_batches = []
        for i, batch in enumerate(self.ds_train):
            if i == self.n_train_image_batches - 1:
                break
            self.n_train_images += batch[0].shape[0]
            train_image_batches.append(batch)
        print(self.n_train_images, 'train images will be logged at each epoch')
        return train_file_writer, train_image_batches

    def train_image_logger(self, epoch, logs):
        lr, hr, sr = [], [], []
        for batch in self.train_image_batches:
            lr.append(stretch_batch(ms_to_rgb_batch(batch[0])))
            hr.append(stretch_batch(batch[1]))
            sr.append(stretch_batch(self.model.predict(batch)))
        # From list to ndarrays
        lr = np.concatenate(lr, axis=0)
        hr = np.concatenate(hr, axis=0)
        sr = np.concatenate(sr, axis=0)
        assert lr.shape[0] == hr.shape[0] == sr.shape[0] == self.n_train_images
        with self.train_file_writer.as_default():
            # LR and HR only need to be written on first epoch
            tf.summary.image('train-LR(MS)', lr, step=0, max_outputs=self.n_train_images)
            tf.summary.image('train-HR(PAN)', hr, step=0, max_outputs=self.n_train_images)
            # SR written every epoch
            tf.summary.image('train-SR', sr, step=epoch, max_outputs=self.n_train_images)

    def build_tb_callback(self):
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = self.tensorboard_logs_dir.joinpath(self.model_name + '_' + timestamp)
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir.as_posix(),
            histogram_freq=0,
            write_graph=False,
            write_images=True,
            update_freq='epoch',
            profile_batch=False,
            embeddings_freq=0,
            embeddings_metadata=None)
        self.callbacks.append(tb_callback)
        return log_dir

    def build_checkpoint_callback(self):
        filepath = self.model_save_dir.joinpath(self.model_name)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath.as_posix() + '-{epoch:02d}-{val_loss:.6f}.h5',
            monitor="val_loss",
            verbose=0,
            save_best_only=False,
            save_weights_only=True,
            mode='auto',
            save_freq='epoch',
            options=None)
        self.callbacks.append(cp_callback)

    def get_callbacks(self):
        return self.callbacks
