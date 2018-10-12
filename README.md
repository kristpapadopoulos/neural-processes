## Categorical Variational Autoencoder

### Keras, Tensorflow Probability and Eager Execution Implementation 
Neural Processes for 1D regression implemented as per paper: [Neural Processes](https://arxiv.org/abs/1807.01622)

Code developed from:
    
1) Kasper Martens: 
https://kasparmartens.rbind.io/post/np/

2) Chris Orm
https://chrisorm.github.io/NGP.html

#### File: np_v0.1.py - Oct 11, 2018

- Tensorflow 1.10.0
- Numpy 1.14.5
- Epochs = 10001
- Learning Rate = 0.001

#### Items for further development:

- Deeper and wider layers
- Different activation functions
- Different priors and latent space transformations
- Learning over mutiple related functions

<p align='center'>
  <b> Example of generated MNIST images from 100 test samples</b>

![cat_vae MNIST samples](MNIST_cat_vae_v0.1_sample.png)
</p>
