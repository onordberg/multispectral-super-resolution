# multispectral-super-resolution
This is a private working repository for my master's project: *Super-resolution of multispectral satellite images using artificial neural networks* (working title)

The actual thesis is located in a separate private repository https://github.com/onordberg/thesis

## Task list

### Done
- [x] Get satellite imagery from Maxar with the help of FFI
- [x] Write functions to extract metadata from sat images
- [x] Write and send project description for approval by institute
- [x] Set up compute environment on *Durin* and *Dvalin* (desktop computers with RTX 2080 Ti card)
- [x] Set up repositories for thesis and code

### In progress
- [ ] Identify significant litterature on SR and especially on SR of sat images
- [ ] Manually assess cloud conditions on a course scale [0-5%, 5-25%, 25-75%, 75-95%, 95-100%]
- [ ] If appropriate: Use automatic cloud classification from FFI
- [ ] Decide on training, validation and test split
- [ ] Approval of project description by institute
- [ ] Do extra assignment work in *INF368 Deep Learning* focusing on CNNs, data generation and efficient data pipelines in tensorflow 2
- [ ] Do relevant tutorials in the book *Generative Deep Learning* by David Foster http://shop.oreilly.com/product/0636920189817.do

### Next
- [ ] Clarify satellite publication restrictions (copyright etc.)
- [ ] Implement a *simple* SR model on *simple* datasets like MNIST, CIFAR-10 etc.
- [ ] Write data generation functions for extracting tiles from satellite imagery
- [ ] Write *Introduction* chapter of thesis
- [ ] Implement a *simple* SR model on the satellite imagery
- [ ] Implement a *state of the art* SR model on a simple dataset like MINST, CIFAR-10
