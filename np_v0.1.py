#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Processes with Tensorflow Eager Execution, Tensorflow Probability and
Keras

Based on implementions by:
    
1) https://kasparmartens.rbind.io/post/np/
2) https://chrisorm.github.io/NGP.html

By: Krist Papadopoulos

Date: October 11, 2018 - v0.1
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp 
tfe = tf.contrib.eager
tf.enable_eager_execution()

# define Tensorflow probability distributions
Normal = tfp.distributions.Normal
KL = tfp.distributions.kl_divergence

# define neural nets for inference and generating over data input and latent 
# samples
class NP(tf.keras.Model):
    def __init__(self, r_in_dim, l1_dim, r_out_dim, z_dim, de_out):
        super(NP, self).__init__()
        
        # input dimensions to represenation encoder
        self.r_in_dim = r_in_dim
        
        # representation encoder and generative net hidden dimension 
        self.l1_dim = l1_dim
        
        # representation encoder output dimension
        self.r_out_dim = r_out_dim
        
        # latent dimension
        self.z_dim = z_dim
        
        # decoder output dimension
        self.de_out = de_out
        
        # representation encoder
        self.r_encoder = tf.keras.Sequential(
          [
          tf.keras.layers.InputLayer(input_shape=(self.r_in_dim,)),
          tf.keras.layers.Dense(self.l1_dim, activation=tf.nn.elu),
          tf.keras.layers.Dense(self.r_out_dim)])
        
        # inference model to estimate the posterior
        self.inference_net = tf.keras.Sequential(
          [
          tf.keras.layers.InputLayer(input_shape=(self.r_out_dim,)),
          tf.keras.layers.Dense(self.z_dim + self.z_dim)])
        
        # generative model to estimate the likelihood p(x|z) for latent samples
        self.generative_net = tf.keras.Sequential(
          [
          tf.keras.layers.InputLayer(input_shape=(self.z_dim, self.z_dim + self.de_out,)),
          tf.keras.layers.Dense(self.l1_dim, activation=tf.nn.sigmoid),
          tf.keras.layers.Dense(self.de_out)])
    
    # function encoding and inference to estimate the posterior in latent space z     
    def zencode(self, x,y):
        xy = tf.concat([x,y], axis=1)
        r = self.r_encoder(xy)
        r_agg = tf.reshape(tf.reduce_mean(r, axis=0),[1,-1])
        mean, logvar = tf.split(self.inference_net(r_agg), num_or_size_splits=2, axis=1)
        return mean, tf.nn.softplus(logvar)

    # generation of latent samples from the posterior estimate (30 samples)
    def reparameterize(self, mean, sigma, n=10):
        # generate latent sample using Gaussian reparamaterization
        z = mean + sigma * tf.random_normal(shape=(n, self.z_dim))
        return z
    
    # generate function samples of y_test using x_test and latent samples
    def decode(self, x_star, z, noise_sd = 0.05):
        N_star = tf.shape(x_star)[0]
        n_draws = z.get_shape().as_list()[0]
        x_star_sample = tf.tile(tf.expand_dims(x_star, [0]),(n_draws,1,1))
        z_sample = tf.tile(tf.expand_dims(z, [1]),(1,N_star,1))
        xz = [x_star_sample, z_sample]
        xz_concat = tf.concat(xz, axis=2)
        mean_x_star = tf.transpose(tf.squeeze(self.generative_net(xz_concat), axis=2))
        return mean_x_star, tf.constant(noise_sd)

# define neural process loss function 
def np_loss(model, x_all, y_all, x_c, y_c, x_t, y_t):
    z_mean_all, z_std_all = model.zencode(x_all, y_all)
    z_mean_context, z_std_context = model.zencode(x_c, y_c)
    z = model.reparameterize(z_mean_all, z_std_all)
    mu, std = model.decode(x_t, z)
    reconstruction_error = tf.reduce_sum(Normal(loc=mu, scale=std).log_prob(y_t),axis=0)
    q_z = Normal(loc=z_mean_all, scale=z_std_all)
    p_z = Normal(loc=z_mean_context, scale=z_std_context)
    KL_qp = KL(q_z, p_z)
    KL_qp_sum = tf.reduce_sum(KL_qp)
    ELBO = tf.reduce_mean(reconstruction_error - KL_qp_sum)
    loss = -ELBO
    return loss

# compute gradients and loss using Tensorflow Eager execution
def compute_gradients(model, x_all, y_all, x_c, y_c, x_t, y_t):
    with tf.GradientTape() as tape:
        loss = np_loss(model, x_all, y_all, x_c, y_c, x_t, y_t)
    return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables, global_step=None):
    return optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

# randomly split input series into x,y context and x,y test sets
def random_split_context_target(x,y, n_context):
    ind = np.arange(x.shape[0])
    mask = np.random.choice(ind, size=n_context, replace=False)
    return x[mask], y[mask], np.delete(x, mask, axis=0), np.delete(y, mask, axis=0)

# plot function samples for unseen x inputs
def visualise(model, x, y, x_star, epoch):
    plt.figure(figsize=(8,8))
    z_mu, z_std = model.zencode(x,y)
    zsamples = model.reparameterize(z_mu, z_std, 30)
    mu, _ = model.decode(x_star, zsamples)
    for i in range(mu.shape[1]):
        plt.plot(x_star.numpy(), mu.numpy()[:,i], linewidth=1)
    
    plt.scatter(x.numpy(), y.numpy())
    plt.title('Function Samples for New Input at Epoch {}'.format(epoch))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('/Users/KP/Desktop/Project/Neural_Processes/Posterior_Plots/posterior_{}.png'.format(epoch), bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()

if __name__ == "__main__":
    
    print("TensorFlow version: {}".format(tf.VERSION))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    
    r_in_dim = 2
    l1_dim = 8
    r_out_dim = 2
    z_dim = 2
    de_out = 1
    
    epochs = 10001
    lr_init = 0.001
    
    lr = tfe.Variable(lr_init, name = "learning_rate", trainable=False)
    
    optimizer = tf.train.AdamOptimizer(lr)
    
    model = NP(r_in_dim, l1_dim, r_out_dim, z_dim, de_out)
    
    # define input range
    all_x_np = np.arange(-2,3,1).reshape(-1,1).astype(np.float32)
    
    # define output range (tests peformed with noise)
    all_y_np = np.sin(all_x_np) #+ np.random.normal(size=1).astype(np.float32)
    
    loss_list = []
    
    for epoch in range(epochs):
        x_context, y_context, x_target, y_target = random_split_context_target(
                                all_x_np, all_y_np, np.random.randint(1,4))
        x_c = tfe.Variable(x_context)
        x_t = tfe.Variable(x_target)
        y_c = tfe.Variable(y_context)
        y_t = tfe.Variable(y_target)

        x_all = tf.concat([x_c, x_t], axis=0)
        y_all = tf.concat([y_c, y_t], axis=0)
        
        gradients, loss = compute_gradients(model, x_all, y_all, x_c, y_c, x_t, y_t)
        apply_gradients(optimizer, gradients, model.trainable_variables)
        
        loss_list.append(loss)
        
        if epoch % 200 == 0:
            x_g = tfe.Variable(np.arange(-4,4, 0.1).reshape(-1,1).astype(np.float32))
            visualise(model, x_all, y_all, x_g, epoch)
        
        
    