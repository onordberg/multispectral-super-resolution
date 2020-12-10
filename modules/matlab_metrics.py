import numpy as np
import pathlib
import tensorflow as tf
import scipy.io
import skvideo.measure
import matlab.engine
from modules.image_utils import *


class MatLabEngine:
    last_imgs = None

    def __init__(self, wd_path, ma=True, niqe=True, input_range=(-1,1), output_dtype='uint16', output_bit_depth=11,
                 scale_mean=0, stretch=False, shave_width=4):
        self.wd_path = wd_path
        self.input_range = input_range
        self.output_dtype = output_dtype
        self.output_bit_depth = output_bit_depth
        self.scale_mean = scale_mean
        self.stretch = stretch
        self.shave_width = shave_width

        if niqe:
            self.mat_cache_path_niqe = None
            self.blocksizerow = 96.0
            self.blocksizecol = 96.0
            self.blockrowoverlap = 0.0
            self.blockcoloverlap = 0.0
        if ma:
            self.mat_cache_path_ma = None

        self.matlab_engine = self.start_matlab(ma=ma, niqe=niqe)

    def start_matlab(self, ma=True, niqe=True):
        if isinstance(self.wd_path, str):
            self.wd_path = pathlib.Path(self.wd_path)

        print('Starting matlab.engine ...')
        matlab_engine = matlab.engine.start_matlab(background=False)
        self.wd_path = self.wd_path.resolve()
        matlab_engine.cd(str(self.wd_path))  # Taking special care around matlab and paths

        # Ma et al.'s SR metric
        if ma:
            matlab_engine.addpath('sr-metric', nargout=0)
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

    def preprocess_imgs(self, numpy_imgs):
        if numpy_imgs.dtype == self.output_dtype:  # uint8 or uint16
            pass
        elif numpy_imgs.dtype == 'float32' and self.output_dtype == 'uint16':
            assert abs(self.input_range[0]) == abs(self.input_range[1])
            radius = abs(self.input_range[0])
            numpy_imgs = output_scaler(numpy_imgs, radius=radius, output_dtype=self.output_dtype,
                                       uint_bit_depth=self.output_bit_depth,
                                       mean_correction=True, mean=self.scale_mean)
        else:
            raise NotImplementedError('dtype of imgs can only be equal to output_dtype or float32, dtype:',
                                      numpy_imgs.dtype)
        if self.shave_width > 0:
            numpy_imgs = shave_borders(numpy_imgs, self.shave_width)
        return numpy_imgs

    def matlab_niqe_metric(self, numpy_imgs):
        if tf.is_tensor(numpy_imgs):
            imgs = numpy_imgs.numpy()
        else:
            imgs = numpy_imgs.copy()
        imgs = self.preprocess_imgs(imgs)
        scipy.io.savemat(self.mat_cache_path_niqe, {'imgs': imgs})
        niqes = self.matlab_engine.computequality_batch(str(self.mat_cache_path_niqe),
                                                        self.blocksizerow, self.blocksizecol,
                                                        self.blockrowoverlap, self.blockcoloverlap,
                                                        self.matlab_engine.workspace['mu_prisparam'],
                                                        self.matlab_engine.workspace['cov_prisparam'])
        self.last_imgs = imgs
        if isinstance(niqes, float):
            niqes = [niqes]
        else:
            niqes = list(niqes[0])
        return niqes

    def matlab_ma_metric(self, numpy_imgs):
        if tf.is_tensor(numpy_imgs):
            imgs = numpy_imgs.numpy()
        else:
            imgs = numpy_imgs.copy()
        imgs = self.preprocess_imgs(imgs)
        scipy.io.savemat(self.mat_cache_path_ma, {'imgs': imgs})
        ma_scores = self.matlab_engine.quality_predict_batch(str(self.mat_cache_path_ma))
        self.last_imgs = imgs
        if isinstance(ma_scores, float):
            ma_scores = [ma_scores]
        else:
            ma_scores = list(ma_scores[0])
        return ma_scores

    def skvideo_niqe_metric(self, numpy_imgs):
        if tf.is_tensor(numpy_imgs):
            imgs = numpy_imgs.numpy()
        else:
            imgs = numpy_imgs.copy()
        imgs = self.preprocess_imgs(imgs)
        niqes = skvideo.measure.niqe(imgs)

        self.last_imgs = imgs
        return niqes
