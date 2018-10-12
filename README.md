## Neural Processes

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

<b> Figures: 30 Neural Process Function Draws for Different Training Epochs</b> 
![posterior_0](/figures/posterior_0.png)
![posterior_1000](/figures/posterior_1000.png)
![posterior_2000](/figures/posterior_2000.png)
![posterior_3000](/figures/posterior_3000.png)
![posterior_4000](/figures/posterior_4000.png)
![posterior_5000](/figures/posterior_5000.png)
