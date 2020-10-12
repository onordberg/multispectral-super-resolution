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

### Work in progress
- [ ] In the loss function: Integrate feature extraction from VGG-19 model trained on [BigEarthNet](https://gitlab.tubit.tu-berlin.de/rsim/bigearthnet-19-models) as alternative to VGG-19 model trained on ImageNet 
- [ ] Do a rough feasability study of whether training on WV02 and validating on GE01 sensor is doable

### Next
- [ ] Implement evaluation metrics NIQE and BRISQUE in TensorFlow
- [ ] Implement evaluation metric LPIPS (NN-based image quality metric) in TensorFlow
- [ ] Implement network interpolation between the PSNR-pretrained model and the GAN trained model as done in the ESRGAN paper
