import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, LeakyReLU, Input, Flatten, Lambda
import functools

from modules.losses_metrics import *
from modules.matlab_metrics import *


def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


def _kernel_init(scale=1.0, seed=None):
    """He normal initializer with scale."""
    scale = 2. * scale
    return tf.keras.initializers.VarianceScaling(
        scale=scale, mode='fan_in', distribution="truncated_normal", seed=seed)


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """

    def __init__(self, axis=-1, momentum=0.9, epsilon=1e-5, center=True,
                 scale=True, name=None, **kwargs):
        super(BatchNormalization, self).__init__(
            axis=axis, momentum=momentum, epsilon=epsilon, center=center,
            scale=scale, name=name, **kwargs)

    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class ResDenseBlock5C(tf.keras.layers.Layer):
    """Residual Dense Block"""

    def __init__(self, n_filters=64, growth_channels=32, res_beta=0.2, weight_decay=0., name='RDB5C',
                 **kwargs):
        super(ResDenseBlock5C, self).__init__(name=name, **kwargs)
        # growth_channels i.e. intermediate channels
        self.res_beta = res_beta
        lrelu_f = functools.partial(LeakyReLU, alpha=0.2)
        _Conv2DLayer = functools.partial(
            Conv2D, kernel_size=3, padding='same',
            kernel_initializer=_kernel_init(0.1), bias_initializer='zeros',
            kernel_regularizer=_regularizer(weight_decay))
        self.conv1 = _Conv2DLayer(filters=growth_channels, activation=lrelu_f())
        self.conv2 = _Conv2DLayer(filters=growth_channels, activation=lrelu_f())
        self.conv3 = _Conv2DLayer(filters=growth_channels, activation=lrelu_f())
        self.conv4 = _Conv2DLayer(filters=growth_channels, activation=lrelu_f())
        self.conv5 = _Conv2DLayer(filters=n_filters, activation=lrelu_f())

    def call(self, x, **kwargs):
        x1 = self.conv1(x)
        x2 = self.conv2(tf.concat([x, x1], 3))
        x3 = self.conv3(tf.concat([x, x1, x2], 3))
        x4 = self.conv4(tf.concat([x, x1, x2, x3], 3))
        x5 = self.conv5(tf.concat([x, x1, x2, x3, x4], 3))
        return x5 * self.res_beta + x


class ResInResDenseBlock(tf.keras.layers.Layer):
    """Residual in Residual Dense Block"""

    def __init__(self, n_filters=64, growth_channels=32, res_beta=0.2, weight_decay=0., name='RRDB',
                 **kwargs):
        super(ResInResDenseBlock, self).__init__(name=name, **kwargs)
        self.res_beta = res_beta
        self.rdb_1 = ResDenseBlock5C(n_filters, growth_channels, res_beta=res_beta, weight_decay=weight_decay)
        self.rdb_2 = ResDenseBlock5C(n_filters, growth_channels, res_beta=res_beta, weight_decay=weight_decay)
        self.rdb_3 = ResDenseBlock5C(n_filters, growth_channels, res_beta=res_beta, weight_decay=weight_decay)

    def call(self, x, **kwargs):
        out = self.rdb_1(x)
        out = self.rdb_2(out)
        out = self.rdb_3(out)
        return out * self.res_beta + x


def rrdb_model(size_in, channels_in, channels_out, n_blocks=23, n_filters=64, growth_channels=32, weight_decay=0.,
               name='RRDB_model'):
    """Residual-in-Residual Dense Block based Model """
    lrelu_f = functools.partial(LeakyReLU, alpha=0.2)
    rrdb_f = functools.partial(ResInResDenseBlock, n_filters=n_filters, growth_channels=growth_channels,
                               weight_decay=weight_decay)
    conv_f = functools.partial(Conv2D, kernel_size=3, padding='same',
                               bias_initializer='zeros',
                               kernel_initializer=_kernel_init(),
                               kernel_regularizer=_regularizer(weight_decay))
    rrdb_truck_f = tf.keras.Sequential(
        [rrdb_f(name="RRDB_{}".format(i)) for i in range(n_blocks)],
        name='RRDB_trunk')

    # extraction
    x = inputs = Input([size_in, size_in, channels_in], name='input_image')
    fea = conv_f(filters=n_filters, name='conv_first')(x)
    fea_rrdb = rrdb_truck_f(fea)
    trunk = conv_f(filters=n_filters, name='conv_trunk')(fea_rrdb)
    fea = fea + trunk

    # upsampling
    size_fea_h = tf.shape(fea)[1] if size_in is None else size_in
    size_fea_w = tf.shape(fea)[2] if size_in is None else size_in
    fea_resize = tf.image.resize(fea, [size_fea_h * 2, size_fea_w * 2],
                                 method='nearest', name='upsample_nn_1')
    fea = conv_f(filters=n_filters, activation=lrelu_f(), name='upconv_1')(fea_resize)
    fea_resize = tf.image.resize(fea, [size_fea_h * 4, size_fea_w * 4],
                                 method='nearest', name='upsample_nn_2')
    fea = conv_f(filters=n_filters, activation=lrelu_f(), name='upconv_2')(fea_resize)
    fea = conv_f(filters=n_filters, activation=lrelu_f(), name='conv_hr')(fea)
    out = conv_f(filters=channels_out, name='conv_last')(fea)

    return tf.keras.Model(inputs, out, name=name)


def build_generator(pretrain_or_gan='pretrain', pretrain_learning_rate=5e-5, pretrain_loss_l1_l2='l1',
                    pretrain_beta_1=0.9, pretrain_beta_2=0.999, pretrain_metrics=('PSNR', 'SSIM'),
                    scaled_range=2.0,
                    n_channels_in=4, n_channels_out=1,
                    height_width_in=None,  # None will make network image size agnostic
                    n_filters=64, n_blocks=23):
    rrdb = rrdb_model(height_width_in, n_channels_in, n_channels_out, n_blocks=n_blocks, n_filters=n_filters)

    metrics_compile = []
    if 'PSNR' in pretrain_metrics or 'psnr' in pretrain_metrics:
        metrics_compile.append(PSNR(range=scaled_range))
    if 'SSIM' in pretrain_metrics or 'ssim' in pretrain_metrics:
        metrics_compile.append(SSIM(range=scaled_range))

    if pretrain_or_gan == 'pretrain':
        if pretrain_loss_l1_l2 == 'l1':
            loss = 'mean_absolute_error'
        elif pretrain_loss_l1_l2 == 'l2':
            loss = 'mean_squared_error'
        else:
            raise ValueError('pretrain_l1_l2 argument must be either "l1" or "l2"')
        rrdb.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=pretrain_learning_rate,
                                                        beta_1=pretrain_beta_1, beta_2=pretrain_beta_2),
                     loss=loss, metrics=metrics_compile)
    return rrdb


def build_discriminator(size, channels, n_filters=64, weight_decay=0., name='Discriminator_VGG'):
    """Discriminator VGG"""
    lrelu_f = functools.partial(LeakyReLU, alpha=0.2)
    conv_k3s1_f = functools.partial(Conv2D,
                                    kernel_size=3, strides=1, padding='same',
                                    kernel_initializer=_kernel_init(),
                                    kernel_regularizer=_regularizer(weight_decay))
    conv_k4s2_f = functools.partial(Conv2D,
                                    kernel_size=4, strides=2, padding='same',
                                    kernel_initializer=_kernel_init(),
                                    kernel_regularizer=_regularizer(weight_decay))
    dense_f = functools.partial(Dense, kernel_regularizer=_regularizer(weight_decay))

    x = inputs = Input(shape=(size, size, channels))

    x = conv_k3s1_f(filters=n_filters, name='conv0_0')(x)
    x = conv_k4s2_f(filters=n_filters, use_bias=False, name='conv0_1')(x)
    x = lrelu_f()(BatchNormalization(name='bn0_1')(x))

    x = conv_k3s1_f(filters=n_filters * 2, use_bias=False, name='conv1_0')(x)
    x = lrelu_f()(BatchNormalization(name='bn1_0')(x))
    x = conv_k4s2_f(filters=n_filters * 2, use_bias=False, name='conv1_1')(x)
    x = lrelu_f()(BatchNormalization(name='bn1_1')(x))

    x = conv_k3s1_f(filters=n_filters * 4, use_bias=False, name='conv2_0')(x)
    x = lrelu_f()(BatchNormalization(name='bn2_0')(x))
    x = conv_k4s2_f(filters=n_filters * 4, use_bias=False, name='conv2_1')(x)
    x = lrelu_f()(BatchNormalization(name='bn2_1')(x))

    x = conv_k3s1_f(filters=n_filters * 8, use_bias=False, name='conv3_0')(x)
    x = lrelu_f()(BatchNormalization(name='bn3_0')(x))
    x = conv_k4s2_f(filters=n_filters * 8, use_bias=False, name='conv3_1')(x)
    x = lrelu_f()(BatchNormalization(name='bn3_1')(x))

    x = conv_k3s1_f(filters=n_filters * 8, use_bias=False, name='conv4_0')(x)
    x = lrelu_f()(BatchNormalization(name='bn4_0')(x))
    x = conv_k4s2_f(filters=n_filters * 8, use_bias=False, name='conv4_1')(x)
    x = lrelu_f()(BatchNormalization(name='bn4_1')(x))

    x = Flatten()(x)
    x = dense_f(units=100, activation=lrelu_f(), name='linear1')(x)
    out = dense_f(units=1, name='linear2')(x)

    return tf.keras.Model(inputs, out, name=name)


def build_bicubic_model(upsample_factor, shape_in=(32, 32, 3),
                        loss='mean_absolute_error', metrics=('PSNR', 'SSIM'), scaled_range=2.0):
    inputs = Input(shape_in, name='input_image')
    x = Lambda(lambda z:
               tf.expand_dims(tf.math.reduce_mean(z, axis=-1), axis=-1))(inputs)
    x = Lambda(lambda z:
               tf.image.resize(z, [shape_in[0] * upsample_factor, shape_in[1] * upsample_factor],
                               method=tf.image.ResizeMethod.BICUBIC, preserve_aspect_ratio=False,
                               antialias=False, name=None))(x)
    model = tf.keras.Model(inputs, x, name='bicubic_upsample_model')

    metrics_compile = []
    if 'PSNR' in metrics or 'psnr' in metrics:
        metrics_compile.append(PSNR(range=scaled_range))
    if 'SSIM' in metrics or 'ssim' in metrics:
        metrics_compile.append(SSIM(range=scaled_range))

    model.compile(loss=loss, metrics=metrics_compile)
    return model


class EsrganModel(tf.keras.Model):
    def __init__(self, generator, discriminator, **kwargs):
        super().__init__(self, **kwargs)

        self.D = discriminator
        self.G = generator

        self.G_optimizer, self.D_optimizer = None, None

        self.G_loss_pixel_f, self.G_loss_pixel_mean = None, None
        self.G_loss_percep_f, self.G_loss_percep_mean = None, None
        self.G_loss_generator_f, self.G_loss_generator_mean = None, None
        # self.G_loss_total_f,
        self.G_loss_reg_mean = None
        self.G_loss_total_mean = None

        self.D_loss_f, self.D_loss_mean = None, None
        self.D_loss_reg_mean = None
        self.D_loss_total_mean = None

        self.G_metric_psnr_f = None
        self.G_metric_psnr_mean = None
        self.G_metric_ssim_f = None
        self.G_metric_ssim_mean = None
        self.G_metric_ma_f = None
        self.G_metric_ma_mean = None
        self.G_metric_niqe_f = None
        self.G_metric_niqe_mean = None
        self.G_metric_pi_f = None
        self.G_metric_pi_mean = None

        self.matlab_engine = None

    def get_config(self):
        d = {'generator': self.G,
             'discriminator': self.D}
        return d

    def call(self, inputs, **kwargs):
        return self.G(inputs, **kwargs)

    def compile(self, **kwargs):
        raise NotImplementedError("Please use special_compile()")

    # https://towardsdatascience.com/tensorflow-2-2-and-a-custom-training-logic-16fa72934ac3
    def special_compile(self, G_optimizer, D_optimizer,
                        G_loss_pixel_w=0.01, G_loss_pixel_l1_l2='l1',

                        G_loss_percep_w=1.0, G_loss_percep_l1_l2='l1', G_loss_percep_layer=54,
                        G_loss_percep_before_act=True,

                        G_loss_generator_w=0.005,
                        metric_reg=False, metric_ma=True, metric_niqe=True,
                        matlab_wd_path='modules/matlab', scale_mean=0, scaled_range=2.0, shave_width=4, **kwargs):
        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer

        self.G_loss_pixel_f = PixelLoss(l1_l2=G_loss_pixel_l1_l2, weight=G_loss_pixel_w)
        self.G_loss_pixel_mean = tf.keras.metrics.Mean(self.G_loss_pixel_f.name)

        self.G_loss_percep_f = PerceptualLoss(l1_l2=G_loss_percep_l1_l2,
                                              output_layer=G_loss_percep_layer,
                                              before_act=G_loss_percep_before_act,
                                              weight=G_loss_percep_w)
        self.G_loss_percep_mean = tf.keras.metrics.Mean(self.G_loss_percep_f.name)

        self.G_loss_generator_f = GeneratorLoss(weight=G_loss_generator_w)
        self.G_loss_generator_mean = tf.keras.metrics.Mean(self.G_loss_generator_f.name)

        self.G_loss_total_mean = tf.keras.metrics.Mean('G_loss_total')

        self.D_loss_f = DiscriminatorLoss()
        self.D_loss_total_mean = tf.keras.metrics.Mean('D_loss_total')

        if metric_reg:
            self.G_loss_reg_mean = tf.keras.metrics.Mean('G_reg_loss')
            self.D_loss_reg_mean = tf.keras.metrics.Mean('D_reg_loss')
            # If regularization is reported, discriminator loss without reg is also reported:
            self.D_loss_mean = tf.keras.metrics.Mean(self.D_loss_f.name)

        self.G_metric_psnr_f = PSNR(range=scaled_range)
        self.G_metric_ssim_f = SSIM(range=scaled_range)

        self.G_metric_psnr_mean = tf.keras.metrics.Mean(self.G_metric_psnr_f.name)
        self.G_metric_ssim_mean = tf.keras.metrics.Mean(self.G_metric_ssim_f.name)

        if metric_ma and metric_niqe:
            self.matlab_engine = MatLabEngine(wd_path=matlab_wd_path, ma=True, niqe=True,
                                              input_range=(-1,1), output_dtype='uint16', output_bit_depth=11,
                                              scale_mean=scale_mean, stretch=False, shave_width=shave_width)
        if metric_ma and not metric_niqe:
            self.matlab_engine = MatLabEngine(wd_path=matlab_wd_path, ma=True, niqe=False,
                                              input_range=(-1, 1), output_dtype='uint16', output_bit_depth=11,
                                              scale_mean=scale_mean, stretch=False, shave_width=shave_width)
        if not metric_ma and metric_niqe:
            self.matlab_engine = MatLabEngine(wd_path=matlab_wd_path, ma=False, niqe=True,
                                              input_range=(-1, 1), output_dtype='uint16', output_bit_depth=11,
                                              scale_mean=scale_mean, stretch=False, shave_width=shave_width)
        if metric_ma:
            self.G_metric_ma_f = self.matlab_engine.matlab_ma_metric
            self.G_metric_ma_mean = tf.keras.metrics.Mean(name='Ma')
        if metric_niqe:
            self.G_metric_niqe_f = self.matlab_engine.matlab_niqe_metric
            self.G_metric_niqe_mean = tf.keras.metrics.Mean(name='NIQE')
        if metric_ma and metric_niqe:
            self.G_metric_pi_f = perceptual_index
            self.G_metric_pi_mean = tf.keras.metrics.Mean(name='PI')

        super().compile(**kwargs)

    def generator_losses(self, hr, sr, hr_D_output, sr_D_output):
        G_loss_pixel = self.G_loss_pixel_f(hr, sr)
        self.G_loss_pixel_mean.update_state(G_loss_pixel)

        G_loss_percep = self.G_loss_percep_f(hr, sr)
        self.G_loss_percep_mean.update_state(G_loss_percep)

        G_loss_generator = self.G_loss_generator_f(hr_D_output, sr_D_output)
        self.G_loss_generator_mean.update_state(G_loss_generator)

        G_loss_reg = tf.math.add_n(self.G.losses)
        if self.G_loss_reg_mean is not None:
            self.G_loss_reg_mean.update_state(G_loss_reg)

        G_loss_total = G_loss_pixel + G_loss_percep + G_loss_generator + G_loss_reg
        self.G_loss_total_mean.update_state(G_loss_total)
        return G_loss_total

    def discriminator_losses(self, hr_D_output, sr_D_output):
        D_loss = self.D_loss_f(hr_D_output, sr_D_output)

        D_loss_reg = tf.math.add_n(self.D.losses)
        if self.D_loss_reg_mean is not None:
            self.D_loss_reg_mean.update_state(D_loss_reg)
            self.D_loss_mean.update_state(D_loss)

        D_loss_total = D_loss + D_loss_reg
        self.D_loss_total_mean.update_state(D_loss_total)
        return D_loss_total

    def train_step(self, data):
        lr, hr = data

        with tf.GradientTape(persistent=True) as tape:
            # Forward pass through generator
            sr = self.G(lr, training=True)

            # Forward pass through the discriminator
            hr_D_output = self.D(hr, training=True)
            sr_D_output = self.D(sr, training=True)

            # Generator losses:
            G_loss_total = self.generator_losses(hr, sr, hr_D_output, sr_D_output)

            # Discriminator losses:
            D_loss = self.discriminator_losses(hr_D_output, sr_D_output)

        G_grads = tape.gradient(G_loss_total, self.G.trainable_variables)
        D_grads = tape.gradient(D_loss, self.D.trainable_variables)
        self.G_optimizer.apply_gradients(zip(G_grads, self.G.trainable_variables))
        self.D_optimizer.apply_gradients(zip(D_grads, self.D.trainable_variables))
        del tape  # https://www.tensorflow.org/api_docs/python/tf/GradientTape

        # Metrics
        G_metric_psnr = self.G_metric_psnr_f(hr, sr)
        self.G_metric_psnr_mean.update_state(G_metric_psnr)

        G_metric_ssim = self.G_metric_ssim_f(hr, sr)
        self.G_metric_ssim_mean.update_state(G_metric_ssim)

        metrics_to_report = {m.name: m.result() for m in self.metrics}

        # Don't report Ma's SR and NIQE metrics since they is not being evaluated in the training step
        if self.G_metric_ma_f is not None:
            assert not tf.executing_eagerly()  # Checks that the graph is static
            metrics_to_report.pop('Ma')
        if self.G_metric_niqe_f is not None:
            metrics_to_report.pop('NIQE')
        if self.G_metric_pi_f is not None:
            metrics_to_report.pop('PI')
        assert not tf.executing_eagerly()  # Checks that the graph is static
        return metrics_to_report

    def test_step(self, data):
        assert not tf.executing_eagerly()  # Checks that the graph is static
        lr, hr = data

        # Forward pass through generator
        sr = self.G(lr, training=False)

        # Forward pass through the discriminator
        hr_D_output = self.D(hr, training=False)
        sr_D_output = self.D(sr, training=False)

        # Generator losses:
        self.generator_losses(hr, sr, hr_D_output, sr_D_output)

        # Discriminator losses:
        self.discriminator_losses(hr_D_output, sr_D_output)

        # Metrics
        G_metric_psnr = self.G_metric_psnr_f(hr, sr)
        self.G_metric_psnr_mean.update_state(G_metric_psnr)

        G_metric_ssim = self.G_metric_ssim_f(hr, sr)
        self.G_metric_ssim_mean.update_state(G_metric_ssim)

        G_metric_ma, G_metric_niqe = tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32)
        if self.G_metric_ma_f is not None:
            G_metric_ma = tf.py_function(self.G_metric_ma_f, [sr], [tf.float32])
            assert not tf.executing_eagerly()  # Checks that the graph is static
            self.G_metric_ma_mean.update_state(G_metric_ma)
            assert not tf.executing_eagerly()  # Checks that the graph is static

        if self.G_metric_niqe_f is not None:
            G_metric_niqe = tf.py_function(self.G_metric_niqe_f, [sr], [tf.float32])
            assert not tf.executing_eagerly()  # Checks that the graph is static
            self.G_metric_niqe_mean.update_state(G_metric_niqe)

        if self.G_metric_pi_f is not None:
            G_metric_pi = self.G_metric_pi_f(G_metric_ma, G_metric_niqe)
            assert not tf.executing_eagerly()  # Checks that the graph is static
            self.G_metric_pi_mean.update_state(G_metric_pi)

        metrics_to_report = {m.name: m.result() for m in self.metrics}
        return metrics_to_report

    @property
    def metrics(self):
        metrics = super().metrics
        metrics.append(self.G_loss_pixel_mean)
        metrics.append(self.G_loss_percep_mean)
        metrics.append(self.G_loss_generator_mean)
        metrics.append(self.G_loss_total_mean)
        # Generator regularization loss:
        if self.G_loss_reg_mean is not None:
            metrics.append(self.G_loss_reg_mean)

        metrics.append(self.D_loss_total_mean)
        # Discriminator regularization loss:
        if self.D_loss_reg_mean is not None:
            metrics.append(self.D_loss_mean)
            metrics.append(self.D_loss_reg_mean)

        metrics.append(self.G_metric_psnr_mean)
        metrics.append(self.G_metric_ssim_mean)

        # Ma et al. SR-metric
        if self.G_metric_ma_mean is not None:
            metrics.append(self.G_metric_ma_mean)
        if self.G_metric_niqe_mean is not None:
            metrics.append(self.G_metric_niqe_mean)
        if self.G_metric_pi_mean is not None:
            metrics.append(self.G_metric_pi_mean)
        return metrics


def build_esrgan_model(pretrain_weights_path,
                       n_channels_in=4, n_channels_out=1, n_filters=64, n_blocks=23, pan_shape=(128, 128, 1),
                       G_lr=0.00005, D_lr=0.00005, G_beta_1=0.9, G_beta_2=0.999, D_beta_1=0.9, D_beta_2=0.999,
                       G_loss_pixel_w=0.01, G_loss_pixel_l1_l2='l1',
                       G_loss_percep_w=1.0, G_loss_percep_l1_l2='l1', G_loss_percep_layer=54,
                       G_loss_percep_before_act=True,
                       G_loss_generator_w=0.005,
                       metric_reg=False, metric_ma=False, metric_niqe=False, matlab_wd_path='modules/matlab',
                       scale_mean=0, scaled_range=2.0, shave_width=4):

    generator = build_generator(pretrain_or_gan='gan',
                                n_channels_in=n_channels_in, n_channels_out=n_channels_out,
                                height_width_in=None,  # None will make network image size agnostic
                                n_filters=n_filters, n_blocks=n_blocks)
    # Loading pre-trained weights from pretrain_model
    if isinstance(pretrain_weights_path, str):
        pretrain_weights_path = pathlib.Path(pretrain_weights_path)
    generator.load_weights(pretrain_weights_path)

    assert pan_shape[0] == pan_shape[1]
    discriminator = build_discriminator(pan_shape[0], pan_shape[2])

    gan_model = EsrganModel(generator, discriminator)
    gan_model.special_compile(tf.keras.optimizers.Adam(learning_rate=G_lr, beta_1=G_beta_1, beta_2=G_beta_2),
                              tf.keras.optimizers.Adam(learning_rate=D_lr, beta_1=D_beta_1, beta_2=D_beta_2),
                              G_loss_pixel_w=G_loss_pixel_w, G_loss_pixel_l1_l2=G_loss_pixel_l1_l2,

                              G_loss_percep_w=G_loss_percep_w, G_loss_percep_l1_l2=G_loss_percep_l1_l2,
                              G_loss_percep_layer=G_loss_percep_layer,  # 22 or 54
                              G_loss_percep_before_act=G_loss_percep_before_act,  # True

                              G_loss_generator_w=G_loss_generator_w,
                              metric_reg=metric_reg, metric_ma=metric_ma, metric_niqe=metric_niqe,
                              matlab_wd_path=matlab_wd_path, scale_mean=scale_mean, scaled_range=scaled_range,
                              shave_width=shave_width)
    gan_model.built = True  # TODO: Do this properly by implementing all abstract classes in EsrganModel
    return gan_model
