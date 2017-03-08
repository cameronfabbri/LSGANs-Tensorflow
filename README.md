# Least Squares Generative Adversarial Networks
Implementation of [LSGANs](https://arxiv.org/pdf/1611.04076v2.pdf) in Tensorflow. Official repo for
the paper can be found [here](https://github.com/martinarjovsky/WassersteinGAN).

___

Requirements
* Python 2.7
* [Tensorflow v1.0](https://www.tensorflow.org/)

Datasets
* LSUN
* Image-net (coming soon)

___


### Results

Critic loss

Generator loss

### Training

### Data
Standard practice is to resize the CelebA images to 96x96 and the crop a center 64x64 image. `loadceleba.py`
takes as input the directory to your images, and will resize them upon loading. To load the entire dataset
at the start instead of reading from disk each step, you will need about 200000\*64\*64\*3\*3 bytes = ~7.5
GB of RAM.

### Tensorboard
Tensorboard logs are stored in `checkpoints/celeba/logs`. I am updating Tensorboard every step as training
isn't completely stable yet. *These can get very big*, around 50GB. See around line 115 in `train.py` to
change how often logs are committed.

### How to

#### Train
```python
python train.py --DATA_DIR=/path/to/images/
`

#### View Results

To see a fancy picture such as the one on this page, simply run

`python createPhotos.py checkpoints/celeba/`

or wherever your model is saved.

If you see the following as your "results", then you did not provide the complete path
to your checkpoint, and this is from the model's initialized weights.

![bad](http://i.imgur.com/MJfmze1.jpg)

