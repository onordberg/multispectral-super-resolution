import numpy as np
import pathlib
import tensorflow as tf
import matlab.engine

def start_matlab(wd_path):
    if isinstance(wd_path, str):
        wd_path = pathlib.Path(wd_path)
    wd_path = str(wd_path.resolve()) # Taking special care around matlab and paths

    print('Starting matlab.engine ...')
    eng = matlab.engine.start_matlab()
    eng.cd(wd_path)
    eng.addpath('external/matlabPyrTools','external/randomforest-matlab/RF_Reg_C', nargout = 0)
    if isinstance(eng, matlab.engine.matlabengine.MatlabEngine):
        print('matlab.engine started')
    return eng

def shave_borders(img, shave_width):
    return img[shave_width:-shave_width, shave_width:-shave_width,:]

def matlab_ma_sr_metric(numpy_img):
    global matlab_engine
    #if not isinstance(matlab_engine, matlab.engine.matlabengine.MatlabEngine):
    #    matlab_engine = start_matlab()
    img = matlab.uint8(numpy_img.tolist())
    ma = matlab_engine.quality_predict(img)
    #print('ma:', type(ma), ma)
    return ma

def tf_ma_sr_metric(tensor_img):
    imgs = tensor_img.numpy()
    batch_size = imgs.shape[0]
    mas = np.empty(batch_size, dtype=np.float32)
    for i in range(batch_size):
        img = imgs[i,:,:,:]
        img = stretch(img)
        img = tf.image.convert_image_dtype(img, tf.uint8, saturate=True).numpy()
        img = shave_borders(img, 4)
        #print(img.shape, type(img), type(img[0,0,0]), np.min(img), np.max(img))
        mas[i] = matlab_ma_sr_metric(img, matlab_engine)
    #print(mas)
    return tf.constant(mas)