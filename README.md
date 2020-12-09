# multispectral-super-resolution
This is a private working repository for my master's project: *Super-resolution of multispectral satellite images using artificial neural networks* (working title)

The actual thesis is located in a separate private repository https://github.com/onordberg/thesis

**Note:** Jupyter Notebook files named `*-display-version.ipynb` have cell outputs and it is recommended to view these if you want to see output of code.

## Task list

### Done
- [x] Get satellite imagery from Maxar with the help of FFI
- [x] Write functions to extract metadata from sat images
- [x] Write and send project description for approval by institute
- [x] Set up compute environment on *Durin* and *Dvalin* (desktop computers with RTX 2080 Ti card)
- [x] Set up repositories for thesis and code
- [x] Randomwly draw two WV02 images of Toulon for early trials
  - `WV02_Toulon_2013_03_16_011651062010_0` and `WV02_Toulon_2019_08_04_011650878010_0`
  - (see [`data-exploration-display-version.ipynb`](https://github.com/onordberg/multispectral-super-resolution/blob/master/data-exploration-display-version.ipynb) for random draw)
- [X] Implement a *simple* SR model, [SRCNN](https://arxiv.org/pdf/1501.00092v3.pdf), on *simple* datasets like MNIST and CIFAR-10
- [X] Manually assess cloud conditions on each image by roughly annotating vector polygons around clouds
- [X] Renamed image directories because they had wrong dates:
  - `WV02_Toulon_2016_04_06_011651052010_0` renamed to `WV02_Toulon_2014_04_06_011651052010_0`
  - `WV02_Toulon_2015_11_15_011651049010_0` renamed to `WV02_Toulon_2015_11_16_011651049010_0`
- [X] Dropped completely an image that are 100% opaque clouds: `WV02_La_Spezia_2018_06_25_011650581010_0`
- [X] Clarify satellite publication restrictions (copyright etc.)
  - Been in contact with the Defence Mapping Agency about this. She has confirmed that "Satellite image © 2020 Maxar Technologies" in the figure text is sufficient.
- [X] Decide on training, validation and test split - `data-exploration-display-version.ipynb`
- [X] Write data generation functions for extracting tiles from satellite imagery. Preserving georeferencing of tiles.
- [X] Write data generator to feed neural net from tiles saved on disk using the `tf.data` API
- [X] Implement a *simple* SR model, [SRCNN](https://arxiv.org/pdf/1501.00092v3.pdf), on satellite imagery
  - This is partly done, but not tuned and probably lots of room to improve on the performance.
- [X] Approval of project description by institute
  - I consider this approved, but have not received formal approval as of October 2020.
- [ ] ~~Implement a *state of the art* SR model on a simple dataset like MINST, CIFAR-10 ~~
  - Skipped this and don't currently see the need to do it.
- [X] Implement a *state of the art* SR model [ESRGAN](https://arxiv.org/pdf/1809.00219v2.pdf) on satellite imagery
 - First iteration of this is done and it gave reasonable results
- [X] Implement pan-sharpening
 - Implemented the Brovey method. Should also implement the Gram-Schmidt method.
- [X] Setup Zotero and import all identified litterature
- [X] Do a rough feasability study of whether training on WV02 and validating on GE01 sensor is doable
  - Results are promising enough to be of interest
  - Data pipeline and model modified to train on WV02 and validate on GE01 by:
   1. resizing GE01 images to match resolution of WV02 (2m MS, 0.5m PAN)
   2. only using RGB+NIR bands of WV02
- [X] Refactor the code for the GAN part. I want the benefits of `tf.keras` model fitting, evaluation and callbacks since this would make running experiments much smoother. Need to override the `train_step` and `test_step` of the Keras `Model` class https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
- [X] Implement stepwise learning rate scheduler for Adam optimizer (as in paper)
- [X] Document implementation of matlab code as a ´tf.keras´ metric as a [notebook](https://github.com/onordberg/deep-learning/blob/main/matlab-function-as-metric-in-tf.keras-model.ipynb)/blog post with simple demo data and model
- [X] Fix wrong scaling of LR and HR images! After noticing GAN training instability I went through a checklist and realized that i have scaled my input images to `[0,1]` `float32`, but my generator output also include negative real numbers. My hypothesis for why I experience GAN training issues is that my discriminator quickly identifies this as a discriminative attribute and my discriminator score inevatibly goes to zero. I am a little unsure of why I did not experience this in previous iterations approximately a month ago, but will for the time being not spend time digging into this and instead scale the images to `[-1,1]` taking into account that the actual bit depth of the satellite images are 11, not 16.
- [X] Decreased time use of matlab function calls by approximately 200x. Instead of sending matlab double arrays through the API (which for some reason is really slow) matlab arrays are rather saved to disk and loaded from the matlab function. Odd workaround, but seems to be best practice.
- [X] Implement metrics in TensorFlow (or in Python as a custom metric)
  - Perceptual index (PI) which is often used as an alternative to PSNR and SSIM and also used in the PIRM-SR challenge which ESRGAN won in 2018. PI is a function of two other metrics:
    - [NIQE](https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=6353522&ref=aHR0cHM6Ly9pZWVleHBsb3JlLmllZWUub3JnL2Fic3RyYWN0L2RvY3VtZW50LzYzNTM1MjI=&tag=1) which is available in Python as a measure in the [scikit-video](http://www.scikit-video.org/stable/modules/generated/skvideo.measure.niqe.html) package.
    - and [Ma et. al.](https://www.sciencedirect.com/science/article/pii/S107731421630203X) which to my knowledge is only available as a [matlab repository](https://github.com/chaoma99/sr-metric)
  - Both are available in the official [matlab repository](https://github.com/roimehrez/PIRM2018) used for evaluation in the PIRM-SR 2018 competition.
  - Status:
    - [X] `Ma et al.`
    - [X] NIQE: There are some indications in discussion threads that the scikit-video implementation is buggy so to be able to trust this implementation a comparison with the matlab implementation is needed. If there are deviations the Matlab implementation seem more trustworthy.
    - [X] PI (a simple average function of Ma and NIQE)

### Work in progress
- [ ] Create a simple cloud/sea tile classificator. After some preliminary training runs I have identified that it is troublesome that almost 50% of all tiles are sea or clouds only. I think the simplest, cleanest way to counter this is to create a neural net cloud/sea detector and undersample those tiles significantly when generating tiles from the satellite images. I hypothesize that not much labeled data is needed for the classificator to generalize well. Alternatively I could have manually drawn a polygon that include the sea area in the images, but this would also eliminate ships and would not help with clouds.
  - Status: 2500 tiles in various sizes generated and labeled. EfficientNet model trained with data augmentation achieving validation accuracy of approx (0.90, 0.95). Remaining: Integrate in tile generator.
- [ ] Do a study of how tile sizes affect NIQE and Ma metrics. Hypothesis: Too small tile sizes are not good. This could probably be an appendix in the thesis if it shows something interesting.
- [ ] Implement all abstract methods of my `Esrgan` model class
- [ ] Calculate `Ma`, `NIQE` and `PI` on a smaller proportion of the validation set (due to performance reasons)


### Next
- [ ] In the loss function: Integrate feature extraction from VGG-19 model trained on satellite images as alternative to VGG-19 model trained on ImageNet 
  - [BigEarthNet](https://gitlab.tubit.tu-berlin.de/rsim/bigearthnet-19-models) looked promising, but will likely not work very well due to input being the 13 bands of the Sentinel-2 sensor. Mine should either be input of panchromatic or RGB since I am comparing with my SR generated panchromatic.
- [ ] Implement evaluation metric LPIPS (NN-based image quality metric) in TensorFlow
- [ ] Implement network interpolation between the PSNR-pretrained model and the GAN trained model as done in the ESRGAN paper
