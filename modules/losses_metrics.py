import tensorflow as tf


class PSNR:
    def __init__(self, range, name='PSNR'):
        self.range = range
        self.name = name

    def __call__(self, hr, sr):
        max_val = self.range
        return tf.image.psnr(hr, sr, max_val=max_val)


class SSIM:
    def __init__(self, range, name='SSIM'):
        self.range = range
        self.name = name

    def __call__(self, hr, sr):
        max_val = self.range
        return tf.image.ssim(hr, sr, max_val=max_val)


def perceptual_index(ma, niqe):
    return 0.5 * ((10 * tf.ones_like(ma) - ma) + niqe)


class DiscriminatorLoss(tf.keras.losses.Loss):
    def __init__(self, name='D_discr_loss'):
        super().__init__(name=name)

    # ragan loss
    def call(self, hr, sr):  # hr == y_true, sr == y_pred
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        sigma = tf.sigmoid
        return 0.5 * (
                cross_entropy(tf.ones_like(hr), sigma(hr - tf.reduce_mean(sr))) +
                cross_entropy(tf.zeros_like(sr), sigma(sr - tf.reduce_mean(hr))))


class GeneratorLoss(tf.keras.losses.Loss):
    def __init__(self, weight=0.005, name='G_generator_loss'):
        super().__init__(name=name)
        self.weight = weight

    # ragan loss
    def call(self, hr, sr):  # hr == y_true, sr == y_pred
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        sigma = tf.sigmoid
        return self.weight * 0.5 * (
                cross_entropy(tf.ones_like(sr), sigma(sr - tf.reduce_mean(hr))) +
                cross_entropy(tf.zeros_like(hr), sigma(hr - tf.reduce_mean(sr))))


class PixelLoss(tf.keras.losses.Loss):
    def __init__(self, l1_l2='l1', weight=0.01, name='G_pixel_loss'):
        super().__init__(name=name)
        self.loss_func = self.l1_l2_loss(l1_l2)
        self.weight = weight

    def l1_l2_loss(self, l1_l2):
        if l1_l2 == 'l1':
            loss_func = tf.keras.losses.MeanAbsoluteError()
        elif l1_l2 == 'l2':
            loss_func = tf.keras.losses.MeanSquaredError()
        else:
            raise NotImplementedError(
                'Loss type {} is not recognized.'.format(l1_l2))
        return loss_func

    def call(self, hr, sr):  # hr == y_true, sr == y_pred
        return self.weight * self.loss_func(hr, sr)


class PerceptualLoss(tf.keras.losses.Loss):
    def __init__(self, l1_l2='l1', output_layer=54, before_act=True, weight=1.0, name='G_perceptual_loss'):
        super().__init__(name=name)
        self.loss_func = self.l1_l2_loss(l1_l2)
        self.feature_extractor = self.build_feature_extractor(output_layer, before_act)
        self.weight = weight

    def l1_l2_loss(self, l1_l2):
        if l1_l2 == 'l1':
            loss_func = tf.keras.losses.MeanAbsoluteError()
        elif l1_l2 == 'l2':
            loss_func = tf.keras.losses.MeanSquaredError()
        else:
            raise NotImplementedError(
                'Loss type {} is not recognized.'.format(l1_l2))
        return loss_func

    def build_feature_extractor(self, output_layer, before_act):
        vgg = tf.keras.applications.VGG19(input_shape=(None, None, 3), weights='imagenet',
                                          include_top=False)
        if output_layer == 22:  # Low level feature
            pick_layer = 5
        elif output_layer == 54:  # Hight level feature
            pick_layer = 20
        else:
            raise NotImplementedError(
                'VGG output layer {} is not recognized.'.format(output_layer))

        if before_act:
            vgg.layers[pick_layer].activation = None

        return tf.keras.Model(vgg.input, vgg.layers[pick_layer].output)

    def feature_extraction(self, img):
        # scaling from [-1,1] to [0,255]
        # img = ((img + 1.) / 2.) * 255.
        img = ((img + tf.ones_like(img)) / 2.) * 255.

        # duplicate 1 channel to 3 channel image
        img = tf.image.grayscale_to_rgb(img)

        # official preprocessing of the image
        # https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg19/preprocess_input
        img = tf.keras.applications.vgg19.preprocess_input(img)

        # feature extraction:
        img_features = self.feature_extractor(img)

        # loss function weight for vgg featuremaps as presented in function (5) in SRGAN paper
        # features_shape = img_features.get_shape()
        # h = features_shape[1]
        # w = features_shape[2]
        # weight = h*w

        return img_features

    def call(self, hr, sr):  # hr == y_true, sr == y_pred
        hr_features = self.feature_extraction(hr)
        sr_features = self.feature_extraction(sr)
        return self.weight * self.loss_func(hr_features, sr_features)
