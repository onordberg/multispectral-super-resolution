import tensorflow as tf
import datetime
import pathlib

from modules.image_utils import *


class EsrganLogger:
    def __init__(self,
                 model_name,
                 tag,  # Can be anything and useful for customizations
                 log_tensorboard=True,
                 tensorboard_logs_dir='logs/tb/',
                 save_models=True,
                 models_save_dir='logs/models/',
                 save_weights_only=True,
                 validate=True,
                 val_steps=10,
                 ds_val_dict=None,  # dict of dataset(s)
                 log_train_images=False,
                 n_train_image_batches=1,
                 ds_train_dict=None,  # dict of dataset(s)
                 log_val_images=False,
                 n_val_image_batches=1
                 ):
        self.callbacks = []
        self.model_name = model_name
        self.tag = tag
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

        if save_models:
            self.model_save_dir = models_save_dir
            if isinstance(self.model_save_dir, str):
                self.model_save_dir = pathlib.Path(self.model_save_dir)
            self.model_save_dir.mkdir(parents=True, exist_ok=True)
            self.save_weights_only = save_weights_only
            self.build_checkpoint_callback()

        if log_tensorboard:
            self.tensorboard_logs_dir = tensorboard_logs_dir
            if isinstance(self.tensorboard_logs_dir, str):
                self.tensorboard_logs_dir = pathlib.Path(self.tensorboard_logs_dir)
            self.tensorboard_logs_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir = self.build_tb_callback()

            if validate:
                self.ds_val_dict = ds_val_dict
                self.val_steps = val_steps
                self.callbacks.append(MultipleValSetsCallback(ds_val_dict=self.ds_val_dict,
                                                              steps=self.val_steps,
                                                              log_dir=self.log_dir))
            if log_train_images:
                self.ds_train_dict = ds_train_dict
                self.n_train_image_batches = n_train_image_batches
                self.callbacks.append(LrHrSrImageCallback(ds_dict=self.ds_train_dict,
                                                          log_dir=self.log_dir,
                                                          n_batches=self.n_train_image_batches,
                                                          train_val='train',
                                                          write_ds_name=False,
                                                          tag=self.tag))
            if log_val_images:
                self.n_val_image_batches = n_val_image_batches
                self.callbacks.append(LrHrSrImageCallback(ds_dict=self.ds_val_dict,
                                                          log_dir=self.log_dir,
                                                          n_batches=self.n_val_image_batches,
                                                          train_val='val',
                                                          write_ds_name=True,
                                                          tag=self.tag))

    def build_tb_callback(self):
        log_dir = self.tensorboard_logs_dir.joinpath(self.model_name + '_' + self.timestamp)
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
        save_dir = self.model_save_dir.joinpath(self.model_name + '_' + self.timestamp)
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir.joinpath(self.model_name)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath.as_posix() + '-{epoch:02d}.h5',
            monitor="loss",
            verbose=0,
            save_best_only=False,
            save_weights_only=self.save_weights_only,
            mode='auto',
            save_freq='epoch',
            options=None)
        self.callbacks.append(cp_callback)

    def get_callbacks(self):
        return self.callbacks


class LrHrSrImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, ds_dict, log_dir, n_batches, train_val='train', write_ds_name=True, tag=None):
        super(LrHrSrImageCallback, self).__init__()
        self.ds_dict = ds_dict
        self.log_dir = log_dir
        self.n_batches = n_batches
        self.train_val = train_val
        self.write_ds_name = write_ds_name
        self.tag = tag
        self.batches = {}
        self.lr = {}
        self.hr = {}
        self.sr = {}
        self.n_images = {}
        self.lr_hr()

    def on_train_begin(self, logs=None):
        for ds_name in self.lr.keys():
            self.sr[ds_name] = self.predict_sr(self.batches[ds_name])
            if self.write_ds_name:
                subdir = self.train_val + '-' + ds_name
            else:
                subdir = self.train_val
            file_writer = tf.summary.create_file_writer(self.log_dir.joinpath(subdir).as_posix())
            with file_writer.as_default():
                tf.summary.image(self.train_val + '-LR(MS)',
                                 self.lr[ds_name],
                                 step=0,
                                 max_outputs=self.n_images[ds_name])
                tf.summary.image(self.train_val + '-HR(PAN)',
                                 self.hr[ds_name],
                                 step=0,
                                 max_outputs=self.n_images[ds_name])
                # step=-1 to indicate SR images before training:
                tf.summary.image(self.train_val + '-SR',
                                 self.sr[ds_name],
                                 step=-1,
                                 max_outputs=self.n_images[ds_name])

    def on_epoch_end(self, epoch, logs=None):
        for ds_name, batches in self.batches.items():
            self.sr[ds_name] = self.predict_sr(batches)

            if self.write_ds_name:
                subdir = self.train_val + '-' + ds_name
            else:
                subdir = self.train_val
            file_writer = tf.summary.create_file_writer(self.log_dir.joinpath(subdir).as_posix())
            with file_writer.as_default():
                tf.summary.image(self.train_val + '-SR',
                                 self.sr[ds_name],
                                 step=epoch,
                                 max_outputs=self.n_images[ds_name])

    def lr_hr(self):
        for ds_name, ds in self.ds_dict.items():
            self.n_images[ds_name] = 0
            self.batches[ds_name] = []
            self.lr[ds_name] = []
            self.hr[ds_name] = []
            for i, batch in enumerate(ds):
                if i == self.n_batches:
                    break
                self.batches[ds_name].append(batch)  # Save batches in "raw" form to be used when predicting sr

                # Custom code needed to handle particulars of experiment-01 since this experiment manipulates the
                # band config of the ms images:
                if self.tag[:3] == 'e01':
                    sensor, rgb_bands = self.experiment_01_rgb_bands(batch[0])
                    print(sensor, rgb_bands)
                else:
                    sensor = ds_name
                    rgb_bands = None
                print(sensor, rgb_bands)
                self.lr[ds_name].append(stretch_batch(ms_to_rgb_batch(batch[0],
                                                                      sensor=sensor,
                                                                      rgb_bands=rgb_bands)))
                self.hr[ds_name].append(stretch_batch(batch[1]))

                self.n_images[ds_name] += batch[0].shape[0]  # Count images

            # From list to ndarrays (convenient when writing with tf.summary.image)
            self.lr[ds_name] = np.concatenate(self.lr[ds_name], axis=0)
            self.hr[ds_name] = np.concatenate(self.hr[ds_name], axis=0)

            print(self.n_images[ds_name], 'images from', self.train_val + '-' + ds_name, 'will be logged at each epoch')

    def predict_sr(self, batches):
        sr = []
        for batch in batches:
            # Predict SR image and append a stretched version (good for display) in list
            sr.append(stretch_batch(self.model.predict(batch)))

        # From list to ndarray (convenient when writing with tf.summary.image)
        sr = np.concatenate(sr, axis=0)
        return sr

    @staticmethod
    def experiment_01_rgb_bands(ms_batch):
        ms_bands = ms_batch[0].shape[-1]
        if ms_bands == 8:
            sensor = 'WV02'
            rgb_bands = None
        elif ms_bands == 6:
            sensor = None
            rgb_bands = (3, 1, 0)
        elif ms_bands == 4:
            sensor = 'GE01'
            rgb_bands = None
        elif ms_bands == 3:
            sensor = None
            rgb_bands = (2, 1, 0)
        else:
            raise ValueError('MS image must have either 8, 6, 4 or 3 bands to be part of experiment-01.',
                             'Number of MS bands provided:', ms_bands)
        return sensor, rgb_bands


class MultipleValSetsCallback(tf.keras.callbacks.Callback):
    def __init__(self, ds_val_dict, steps, log_dir):
        super(MultipleValSetsCallback, self).__init__()
        self.ds_val_dict = ds_val_dict
        self.steps = steps
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs=None):
        for ds_val_name, ds_val in self.ds_val_dict.items():
            val_file_writer = tf.summary.create_file_writer(
                self.log_dir.joinpath('val-' + ds_val_name).as_posix())
            metrics = self.model.evaluate(ds_val, steps=self.steps, return_dict=True, verbose=0)
            with val_file_writer.as_default():
                for metric_name, metric_value in metrics.items():
                    tf.summary.scalar('epoch_' + metric_name, data=metric_value, step=epoch)
