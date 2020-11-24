import tensorflow as tf

class PsnrMetric(tf.keras.losses.Loss):
    def __init__(self, name = 'PSNR'):
        super().__init__(name=name)
        
    def call(self, hr, sr):
        return tf.image.psnr(hr, sr, max_val = 1.0)
    
class SsimMetric(tf.keras.losses.Loss):
    def __init__(self, name = 'SSIM'):
        super().__init__(name=name)
        
    def call(self, hr, sr):
        return tf.image.ssim(hr, sr, max_val = 1.0)

class DiscriminatorLoss(tf.keras.losses.Loss):
    def __init__(self, name='D_discr_loss'):
        super().__init__(name=name)
    
    # ragan loss
    def call(self, hr, sr): #hr == y_true, sr == y_pred
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        sigma = tf.sigmoid
        return 0.5 * (
            cross_entropy(tf.ones_like(hr), sigma(hr - tf.reduce_mean(sr))) +
            cross_entropy(tf.zeros_like(sr), sigma(sr - tf.reduce_mean(hr))))
    
class GeneratorLoss(tf.keras.losses.Loss):
    def __init__(self, name='G_generator_loss'):
        super().__init__(name=name)
    
    # ragan loss
    def call(self, hr, sr): #hr == y_true, sr == y_pred
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        sigma = tf.sigmoid
        return 0.5 * (
            cross_entropy(tf.ones_like(sr), sigma(sr - tf.reduce_mean(hr))) +
            cross_entropy(tf.zeros_like(hr), sigma(hr - tf.reduce_mean(sr))))
    
class PixelLoss(tf.keras.losses.Loss):
    def __init__(self, l1_l2 = 'l1', name='G_pixel_loss'):
        super().__init__(name=name)
        self.loss_func = self.l1_l2_loss(l1_l2)
    
    def l1_l2_loss(self, l1_l2):
        if l1_l2 == 'l1':
            loss_func = tf.keras.losses.MeanAbsoluteError()
        elif l1_l2 == 'l2':
            loss_func = tf.keras.losses.MeanSquaredError()
        else:
            raise NotImplementedError(
                'Loss type {} is not recognized.'.format(l1_l2))
        return loss_func
    
    def call(self, hr, sr): #hr == y_true, sr == y_pred
        return self.loss_func(hr, sr)

class PerceptualLoss(tf.keras.losses.Loss):
    def __init__(self, l1_l2='l1', output_layer=54, before_act=True, name='G_perceptual_loss'):
        super().__init__(name=name)
        self.loss_func = self.l1_l2_loss(l1_l2)
        self.feature_extractor = self.build_feature_extractor(output_layer, before_act)
    
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

    def call(self, hr, sr): #hr == y_true, sr == y_pred
        #print(sr.shape, hr.shape)
        sr_rgb = tf.image.grayscale_to_rgb(sr)
        hr_rgb = tf.image.grayscale_to_rgb(hr)
        #print(sr_rgb.shape, hr_rgb.shape)
        # the input scale range is [0, 1] (vgg is [0, 255]).
        # 12.75 is rescale factor for vgg featuremaps.
        preprocess_sr = tf.keras.applications.vgg19.preprocess_input(sr_rgb * 255.) / 12.75
        preprocess_hr = tf.keras.applications.vgg19.preprocess_input(hr_rgb * 255.) / 12.75
        #print(preprocess_sr.shape, preprocess_hr.shape)
        sr_features = self.feature_extractor(preprocess_sr)
        hr_features = self.feature_extractor(preprocess_hr)

        return self.loss_func(hr_features, sr_features)
    
class EsrganTotalGeneratorLoss(tf.keras.losses.Loss):
    def __init__(self, G_loss_pixel_f, G_loss_pixel_sc, 
                 G_loss_percep_f, G_loss_percep_sc, 
                 G_loss_generator_f, G_loss_generator_sc,
                 name='G_total_loss'):
        super().__init__(name=name)
        self.G_loss_pixel_f = G_loss_pixel_f
        self.G_loss_pixel_sc = G_loss_pixel_sc
        self.G_loss_percep_f = G_loss_percep_f
        self.G_loss_percep_sc = G_loss_percep_sc
        self.G_loss_generator_f = G_loss_generator_f
        self.G_loss_generator_sc = G_loss_generator_sc
    
    def call(self, hr, sr): #hr == y_true, sr == y_pred
        hr_img, hr_D_output = hr
        sr_img, sr_D_output = sr
        
        pixel = self.G_loss_pixel_sc * self.G_loss_pixel_f(hr_img, sr_img)
        perceptual = self.G_loss_percep_sc * self.G_loss_percep_f(hr_img, sr_img)
        generator = self.G_loss_generator_sc * self.G_loss_generator_f(hr_D_output, sr_D_output)
        return tf.math.add_n([pixel, perceptual, generator])