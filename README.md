# Least Squares Generative Adversarial Networks
Implementation of [LSGANs](https://arxiv.org/pdf/1611.04076v2.pdf) in Tensorflow. Official repo for
the paper can be found [here](https://github.com/martinarjovsky/WassersteinGAN).

___

Requirements
* Python 2.7
* [Tensorflow v1.0](https://www.tensorflow.org/)

Datasets
* LSUN

___


### Results
The results aren't as good as the paper shows, and I'm still investigating why.

![img](http://i.imgur.com/ilBIXhI.png)

### Training

### Data
I used the LSUN church dataset. The images are resized to 112x112 (same size as the generator produces).

### How to

#### Train
`python train.py --DATA_DIR=[/path/to/images/] --DATASET=[dataset] --BATCH_SIZE=[batch_size]`

For example, if you have the [LSUN dataset](http://lsun.cs.princeton.edu/2016/)

`pytohn train.py --DATA_DIR=/mnt/lsun/church/images/ --DATASET=church`


#### View Results

To see a fancy picture such as the one on this page, simply run

`python createPhotos.py checkpoints/celeba/`

or wherever your model is saved.

If you see the following as your "results", then you did not provide the complete path
to your checkpoint, and this is from the model's initialized weights.

![bad](http://i.imgur.com/MJfmze1.jpg)

