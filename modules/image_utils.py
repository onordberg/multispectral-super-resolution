import matplotlib.pyplot as plt
import pathlib
import numpy as np
import math
import rasterio
import rasterio.plot
import tensorflow as tf

def stretch(image, individual_bands = True):
    dims = len(image.shape)
    assert dims == 3
    n_bands = image.shape[2]
    image_out = np.empty(image.shape)
    if individual_bands and n_bands > 1:
        for i in range(n_bands):
            image_out[:,:,i] = (image[:,:,i] - np.min(image[:,:,i])) / (np.max(image[:,:,i]) - np.min(image[:,:,i]))
    else:
        image_out = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image_out

def stretch_batch(image_batch, individual_bands = True):
    #TODO: Vectorize to increase efficiency, however it might not be worth it
    dims = len(image_batch.shape)
    assert dims == 4
    batch_size = image_batch.shape[0]
    batch_out = np.empty(image_batch.shape)
    for i in range(batch_size):
        batch_out[i,:,:,:] = stretch(image_batch[i,:,:,:], individual_bands=individual_bands)
    return batch_out

def pansharpen(ms, pan, sensor = 'WV02', method = 'brovey', fourth_band = 'nir', 
               w = [0.2]*5, stretch_output = True):
    ms = ms[:,:,:]
    pan = pan[:,:]
    
    scale = int(pan.shape[0]/ms.shape[0])

    ms_ups = tf.image.resize(ms, [ms.shape[0]*scale, ms.shape[1]*scale], 
                         method=tf.image.ResizeMethod.BICUBIC, preserve_aspect_ratio=False,
                         antialias=False, name=None)

    # Convert to ndarray if tensor
    if isinstance(pan, tf.python.framework.ops.EagerTensor):
        pan = pan.numpy()
    if isinstance(ms_ups, tf.python.framework.ops.EagerTensor):
        ms_ups = ms_ups.numpy()
    
    if sensor == 'WV02':
        r = ms_ups[:,:,4]
        g = ms_ups[:,:,2]
        b = ms_ups[:,:,1]
        if fourth_band == 'red_edge':
            i = ms_ups[:,:,5]
        elif fourth_band == 'nir':
            i = ms_ups[:,:,6]
        elif fourth_band == 'nir2':
            i = ms_ups[:,:,7]
    elif sensor == 'GE01':
        r = ms_ups[:,:,2]
        g = ms_ups[:,:,1]
        b = ms_ups[:,:,0]
        if fourth_band == 'red_edge':
            raise ValueError('GE01 does not contain a red_edge band')
        elif fourth_band == 'nir':
            i = ms_ups[:,:,3]
        elif fourth_band == 'nir2':
            raise ValueError('GE01 does not contain a nir2 band')
        
    if method == 'brovey':
        dnf = (pan - w[3]*i)/(w[0]*r + w[1]*g + w[2]*b)
        r = np.expand_dims(r * dnf, -1)
        g = np.expand_dims(g * dnf, -1)
        b = np.expand_dims(b * dnf, -1)
        i = np.expand_dims(i * dnf, -1)
        img_out = np.concatenate([r, g, b], axis = 2)
        #print(img_out.shape)
    
    if stretch_output:
        img_out = stretch(img_out)
        
    return img_out

def ms_to_rgb(ms, sensor = 'WV02'):
    if sensor == 'WV02':
        rgb = [np.expand_dims(ms[:,:,4], -1), 
               np.expand_dims(ms[:,:,2], -1), 
               np.expand_dims(ms[:,:,1], -1)]
    elif sensor == 'GE01':
        rgb = [np.expand_dims(ms[:,:,2], -1), 
               np.expand_dims(ms[:,:,1], -1), 
               np.expand_dims(ms[:,:,0], -1)]
    elif sensor == 'WV03_VNIR':
        rgb = [np.expand_dims(ms[:,:,1], -1), 
               np.expand_dims(ms[:,:,2], -1), 
               np.expand_dims(ms[:,:,3], -1)]
    else:
        raise ValueError('Only WV02, GE01 and WV03_VNIR band configurations implemented') 
    rgb = np.concatenate(rgb, axis = 2)
    return rgb

def ms_to_rgb_batch(ms_batch, sensor = 'WV02'):
    #TODO: Vectorize to increase efficiency, however it might not be worth it
    dims = len(ms_batch.shape)
    assert dims == 4
    batch_size = ms_batch.shape[0]
    batch_out = np.empty((batch_size, ms_batch.shape[1], ms_batch.shape[2], 3))
    for i in range(batch_size):
        batch_out[i,:,:,:] = ms_to_rgb(ms_batch[i,:,:,:], sensor=sensor)
    return batch_out

def geotiff_to_ndarray(tif_path):
    if isinstance(tif_path, str):
        tif_path = pathlib.Path(tif_path)

    with rasterio.open(tif_path) as src:
        img = src.read()
        img = rasterio.plot.reshape_as_image(img) # from channels first to channels last
        return img

def ndarray_to_png(arr, png_path, ms_or_pan='pan', scale=True):
    if isinstance(png_path, str):
        png_path = pathlib.Path(png_path)
    
    tf.keras.preprocessing.image.save_img(png_path, arr, data_format='channels_last', 
                                          file_format='png', scale=scale)
    
def geotiff_to_png(tif_path, ms_or_pan='pan', scale=True, sensor='WV02'):
    if isinstance(tif_path, str):
        tif_path = pathlib.Path(tif_path)
    png_path = tif_path.with_suffix('.png')
    img = geotiff_to_ndarray(tif_path)
    
    if ms_or_pan == 'ms':
        img = ms_to_rgb(img, sensor=sensor)
        img = stretch(img)
        
    ndarray_to_png(img, png_path, ms_or_pan=ms_or_pan, scale=scale)

    
def plot_subplot(ax, img, title, gray = False, metrics = False):
    ax.set_title(title)
    if gray:
        ax.imshow(img, cmap = 'gray')
    else:
        ax.imshow(img)
        
    # TODO: Implement metrics in title
    #axs[0,1].set_title('MS Bicubic Upsampling ' + 
    #                   '\n PSNR: ' + 
    #                   str(psnr_np(bicubic[i], pan[i]).numpy()) + 
    #                   '\n SSIM: ' +
    #                   str(ssim_np(bicubic[i], pan[i]).numpy()))
    #axs[0,1].imshow(bicubic[i], cmap = cmap)
    return ax

def save_image_pairs(ms_img, pan_img, save_dir, filename, 
                     sr_pretrain_img = False, sr_gan_img = False, file_ext = 'png'):
    full_filename = str(filename + '.' + file_ext)
    sr_pretrain, sr_gan = False, False
    if not isinstance(sr_pretrain_img, bool):
        sr_pretrain = True
    if not isinstance(sr_gan_img, bool):
        sr_gan = True
    print(sr_pretrain, sr_gan)
    hr_dir = pathlib.Path(save_dir, 'hr')
    hr_dir.mkdir(exist_ok = True)
    lr_dir = pathlib.Path(save_dir, 'lr')
    lr_dir.mkdir(exist_ok = True)
    
    sr_pretrain_dir, sr_gan_dir = None, None
    if sr_pretrain:
        sr_pretrain_dir = pathlib.Path(save_dir, 'sr-pretrain')
        sr_pretrain_dir.mkdir(exist_ok = True)
    if sr_gan:
        sr_gan_dir = pathlib.Path(save_dir, 'sr-gan')
        sr_gan_dir.mkdir(exist_ok = True)
    
    if file_ext == 'png':
        tf.keras.preprocessing.image.save_img(lr_dir.joinpath(full_filename), ms_img, 
                                              data_format='channels_last', file_format='png', scale=True)
        tf.keras.preprocessing.image.save_img(hr_dir.joinpath(full_filename), 
                                              np.expand_dims(pan_img, -1), 
                                              data_format='channels_last', file_format='png', scale=True)
        if sr_pretrain:
            tf.keras.preprocessing.image.save_img(sr_pretrain_dir.joinpath(full_filename), 
                                                  np.expand_dims(sr_pretrain_img, -1),
                                                  data_format='channels_last', file_format='png', scale=True)
        if sr_gan:
            tf.keras.preprocessing.image.save_img(sr_gan_dir.joinpath(full_filename), sr_gan_img,
                                                  data_format='channels_last', file_format='png', scale=True)
    #TODO: file_ext = tif

    
def plot_comparison(ds, pretrain_model = False, gan_model = False, bicubic = True, 
                    rgb = True, pansharp = False, sensor = 'WV02', 
                    save_dir = False, save_raw = False, filename = False, raw_file_ext = 'png'):
    imgs, plot, gray, metrics = {}, {}, {}, {}
    batch = next(iter(ds))
    
    pretrain, gan = False, False
    if not isinstance(pretrain_model, bool):
        pretrain = True
    if not isinstance(gan_model, bool):
        gan = True
    
    imgs['ms'] = batch[0][0].numpy()
    imgs['ms_mean'] = tf.math.reduce_mean(imgs['ms'], axis = -1).numpy()
    plot['ms_mean'], gray['ms_mean'] = True, True
    
    imgs['pan'] = batch[1].numpy()[0,:,:,0]
    plot['pan'], gray['pan'] = True, True
    
    if rgb:
        imgs['ms_rgb'] = stretch(ms_to_rgb(imgs['ms'], sensor = sensor))
        plot['ms_rgb'], gray['ms_rgb'] = True, False
    
    if bicubic:
        imgs['bicubic'] = tf.image.resize(imgs['ms'], 
                                          [imgs['pan'].shape[0], imgs['pan'].shape[1]], 
                                          method=tf.image.ResizeMethod.BICUBIC).numpy()

        imgs['bicubic_mean'] = tf.math.reduce_mean(imgs['bicubic'], axis = -1).numpy()
        plot['bicubic_mean'], gray['bicubic_mean'] = True, True
        if rgb:
            imgs['bicubic_rgb'] = stretch(ms_to_rgb(imgs['bicubic'], sensor = sensor))
            plot['bicubic_rgb'], gray['bicubic_rgb'] = True, False
    
    if pretrain:
        imgs['sr_pretrain'] = pretrain_model.predict(batch)[0,:,:,0]
        plot['sr_pretrain'], gray['sr_pretrain'] = True, True
        
    if gan:
        imgs['sr_gan'] = gan_model.predict(batch)[0,:,:,0]
        plot['sr_gan'] = True
        gray['sr_gan'] = True
    
    if pansharp:
        if pretrain:
            imgs['sr_pretrain_pansharp'] = pansharpen(imgs['ms'], imgs['sr_pretrain'], sensor = sensor, 
                                                      method = 'brovey', fourth_band = 'nir', 
                                                      w = [0.2]*5, stretch_output = True)
            plot['sr_pretrain_pansharp'], gray['sr_pretrain_pansharp'] = True, False

        if gan:
            imgs['sr_gan_pansharp'] = pansharpen(imgs['ms'], imgs['sr_gan'], sensor = sensor, 
                                                      method = 'brovey', fourth_band = 'nir', 
                                                      w = [0.2]*5, stretch_output = True)
            plot['sr_gan_pansharp'], gray['sr_gan_pansharp'] = True, False

        imgs['real_pansharp'] = pansharpen(imgs['ms'], imgs['pan'], sensor = sensor, 
                                                  method = 'brovey', fourth_band = 'nir', 
                                                  w = [0.2]*5, stretch_output = True)
        plot['real_pansharp'], gray['real_pansharp'] = True, False
    
    n_subplots = sum(plot.values())
    print('Plotting', n_subplots, 'subplots')
    for img_name in plot.keys():
        print(img_name, imgs[img_name].shape)
    n_cols = 3
    n_rows = math.ceil(n_subplots/n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, constrained_layout = True, 
                            figsize = (n_cols*10, n_rows*10))
    fig.suptitle('Satellite image tiles - Comparisons between multispectral, panchromatic and super-resolution images')
    fig.patch.set_facecolor('white')
    subplot_keys = list(plot.keys())
    k = 0
    for i in range(n_rows):
        for j in range(n_cols):
            img_name = subplot_keys[k]
            plot_subplot(axs[i,j], imgs[img_name], img_name, gray[img_name])
            k += 1
            if k >= len(subplot_keys):
                break
                
    if not isinstance(save_dir, bool) and not isinstance(filename, bool):
        fig_path = pathlib.Path(save_dir, str(filename + '.png'))
        fig.savefig(fig_path)
        if save_raw and raw_file_ext == 'png':
            # png implies saving an rgb version of the ms image
            if pretrain and not gan:
                save_image_pairs(imgs['ms_rgb'], imgs['pan'], save_dir, filename,
                                 sr_pretrain_img = imgs['sr_pretrain'], sr_gan_img = False,
                                 file_ext = raw_file_ext)
            elif not pretrain and gan:
                save_image_pairs(imgs['ms_rgb'], imgs['pan'], save_dir, filename,
                                 sr_pretrain_img = False, sr_gan_img = imgs['sr_gan'],
                                 file_ext = raw_file_ext)
            elif pretrain and gan:
                save_image_pairs(imgs['ms_rgb'], imgs['pan'], save_dir, filename,
                                 sr_pretrain_img = imgs['sr_pretrain'], sr_gan_img = imgs['sr_gan'],
                                 file_ext = raw_file_ext)
            else:
                save_image_pairs(imgs['ms_rgb'], imgs['pan'], save_dir, filename,
                                 sr_pretrain_img = False, sr_gan_img = False,
                                 file_ext = raw_file_ext)
        #TODO: raw_file_ext = tif
        
           
def show_batch(image_batch):
    ms = image_batch[0].numpy()
    pan = image_batch[1].numpy()
    print('ms batch shape', ms.shape)
    print('pan batch shape', pan.shape)

    plt.figure(figsize=(30,30))
    for i in range(BATCH_SIZE):
        i = i * 2
        ax_ms = plt.subplot(4,4,i+1, label = 'ms')
        
        ms_image = ms[i,:,:,:3] # Just showing channel 2 as grayscale
        pan_image = pan[i,:,:]

        #plt.imshow(ms_image)
        ms_image = ms_to_rgb(ms_image, sensor = 'GE01')
        print(ms_image.shape)
        ax_ms.imshow(ms_image)
        
        ax_pan = plt.subplot(4,4,i+2, label = 'pan')

        ax_pan.imshow(pan_image, cmap = 'gray')