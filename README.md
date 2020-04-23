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
- [x] Draw two WV02 images of Toulon for early trials
  - `WV02_Toulon_2013_03_16_011651062010_0` and `WV02_Toulon_2019_08_04_011650878010_0`
  - (see [`data-exploration-display-version.ipynb`](https://github.com/onordberg/multispectral-super-resolution/blob/master/data-exploration-display-version.ipynb) for random draw)

### Work in progress
- [ ] Identify significant litterature on SR and especially on SR of sat images
  - Papers with code on SR: https://paperswithcode.com/task/image-super-resolution
  - SRCNN - 2014: [paperswithcode](https://paperswithcode.com/paper/image-super-resolution-using-deep), [paper](https://arxiv.org/pdf/1501.00092v3.pdf)
  - SRGAN - 2017: [paperswithcode](https://paperswithcode.com/paper/photo-realistic-single-image-super-resolution), [paper](https://arxiv.org/pdf/1609.04802v5.pdf)
  - ESRGAN - 2018: [paperswithcode](https://paperswithcode.com/paper/esrgan-enhanced-super-resolution-generative), [paper](https://arxiv.org/pdf/1809.00219v2.pdf)
  - SAN - 2019: [paperswithcode](https://paperswithcode.com/paper/second-order-attention-network-for-single), [paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Dai_Second-Order_Attention_Network_for_Single_Image_Super-Resolution_CVPR_2019_paper.pdf)
  - EESRGAN - 2020 (satellite SR): [paperswithcode](https://paperswithcode.com/paper/small-object-detection-in-remote-sensing), [paper](https://arxiv.org/pdf/2003.09085v4.pdf)
- [ ] Approval of project description by institute
- [ ] Implement a *simple* SR model, [SRCNN](https://arxiv.org/pdf/1501.00092v3.pdf), on *simple* datasets like MNIST and CIFAR-10
- [ ] Do relevant tutorials in the book *Generative Deep Learning* by David Foster http://shop.oreilly.com/product/0636920189817.do

### Next
- [ ] Manually assess cloud conditions on a course scale [0-5%, 5-25%, 25-75%, 75-95%, 95-100%]
- [ ] If appropriate: Use automatic cloud classification from FFI
- [ ] Decide on training, validation and test split
- [ ] Clarify satellite publication restrictions (copyright etc.)
- [ ] Write data generation functions for extracting tiles from satellite imagery
- [ ] Write *Introduction* chapter of thesis
- [ ] Implement a *simple* SR model on the satellite imagery
- [ ] Implement a *state of the art* SR model on a simple dataset like MINST, CIFAR-10
