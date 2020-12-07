import numpy as np
import pathlib
import tensorflow as tf
import scipy.io
import matlab.engine
from modules.image_utils import *


class MatLabEngine:
    def __init__(self, wd_path, ma=True, niqe=True):
        self.wd_path = wd_path
        if niqe:
            self.mat_cache_path_niqe = None
            self.blocksizerow = 96.0
            self.blocksizecol = 96.0
            self.blockrowoverlap = 0.0
            self.blockcoloverlap = 0.0
        if ma:
            self.mat_cache_path_ma = None

        self.matlab_engine = self.start_matlab(ma=ma, niqe=niqe)

        if niqe:
            self.blocksizerow = 96.0
            self.blocksizecol = 96.0
            self.blockrowoverlap = 0.0
            self.blockcoloverlap = 0.0
        if ma:
            self.mat_cache_path_ma = None

    def start_matlab(self, ma=True, niqe=True):
        if isinstance(self.wd_path, str):
            self.wd_path = pathlib.Path(self.wd_path)

        print('Starting matlab.engine ...')
        matlab_engine = matlab.engine.start_matlab(background=False)
        self.wd_path = self.wd_path.resolve()
        matlab_engine.cd(str(self.wd_path))  # Taking special care around matlab and paths

        # Ma et al.'s SR metric
        if ma:
            matlab_engine.addpath('sr-metric/external/matlabPyrTools',
                                  'sr-metric/external/randomforest-matlab/RF_Reg_C', nargout=0)
            mat_cache_path_ma = self.wd_path.joinpath('cache')
            mat_cache_path_ma.mkdir(parents=True, exist_ok=True)
            self.mat_cache_path_ma = mat_cache_path_ma.joinpath('ma_cache.mat')

        # NIQE
        if niqe:
            matlab_engine.addpath('niqe_release', nargout=0)
            matlab_engine.eval("load('niqe_release/modelparameters.mat')", nargout=0)
            mat_cache_path_niqe = self.wd_path.joinpath('cache')
            mat_cache_path_niqe.mkdir(parents=True, exist_ok=True)
            self.mat_cache_path_niqe = mat_cache_path_niqe.joinpath('niqe_cache.mat')

        if isinstance(matlab_engine, matlab.engine.matlabengine.MatlabEngine):
            print('matlab.engine started')
        return matlab_engine

    def shave_borders(self, img, shave_width):
        # TODO: Rewrite as method
        return img[shave_width:-shave_width, shave_width:-shave_width,:]

    def matlab_niqe_metric(self, numpy_imgs):
        scipy.io.savemat(self.mat_cache_path_niqe, {'imgs': numpy_imgs})
        niqes = self.matlab_engine.computequality_batch(str(self.mat_cache_path_niqe),
                                                        self.blocksizerow, self.blocksizecol,
                                                        self.blockrowoverlap, self.blockcoloverlap,
                                                        self.matlab_engine.workspace['mu_prisparam'],
                                                        self.matlab_engine.workspace['cov_prisparam'])
        return list(niqes._data)

    def matlab_ma_sr_metric(self, numpy_img, matlab_engine):
        # TODO: Rewrite as method
        img = matlab.uint8(numpy_img.tolist())
        ma = matlab_engine.quality_predict(img)
        return ma

    def tf_ma_sr_metric(self, tensor_img, matlab_engine):
        # TODO: Rewrite as method
        imgs = tensor_img.numpy()
        batch_size = imgs.shape[0]
        mas = np.empty(batch_size, dtype=np.float32)
        for i in range(batch_size):
            img = imgs[i,:,:,:]
            img = stretch(img)
            img = tf.image.convert_image_dtype(img, tf.uint8, saturate=True).numpy()
            img = self.shave_borders(img, 4)
            # print(img.shape, type(img), type(img[0,0,0]), np.min(img), np.max(img))
            mas[i] = self.matlab_ma_sr_metric(img, matlab_engine)
        return tf.constant(mas)
