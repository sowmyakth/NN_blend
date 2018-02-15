"""Script to create CNN for retrieving isolated galaxy image from a two
galaxy blend"""
import os
import sys
# import skimage.io as io
import numpy as np
import math
import tensorflow as tf


def get_loss(y, y_out):
    total_loss = tf.nn.l2_loss(y - y_out)
    mean_loss = tf.reduce_mean(total_loss)
    return mean_loss


class CNN_deblender(object):
    def __init__(self, ):
        self.optimizer = tf.train.AdamOptimizer(5e-4)

    def simple_model(self, X, y):
        """makes a simple 2 layer CNN
        layer 1 Conv 5*5*2s2, 256/ReLU
        layer 2 FC 32*32,1, 49
         """
        Wconv1 = tf.get_variable("Wconv1", shape=[5, 5, 2, 256])
        bconv1 = tf.get_variable("bconv1", shape=[256])
        W1 = tf.get_variable("W1", shape=[32, 32, 1, 49])
        b1 = tf.get_variable("b1", shape=[32, 32])
        # define our graph (e.g. two_layer_convnet)
        a1 = tf.nn.conv2d(X, Wconv1, strides=[1, 2, 2, 1],
                          padding='VALID') + bconv1
        h1 = tf.nn.relu(a1)
        h1_flat = tf.reshape(h1, [32, 32, 49, -1])
        y_out = tf.matmul(W1, h1_flat)
        y_out = tf.transpose(y_out)
        y_out = tf.reshape(y_out, [-1, 32, 32]) + b1
        return y_out

    def train(self, ):
        train_step = optimizer.minimize(mean_loss)

    def run_model(self, session, predict, mean_loss, Xd, yd,
                  epochs=1, batch_size=64, print_every=100,
                  training=None, plot_losses=False):
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)
    training_now = training is not None
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss, pred_loss,]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:, :, :],
                         y: yd[idx, :, :],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[idx].shape[0]
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, im = session.run(variables,feed_dict=feed_dict)
            pred = y_out.eval(session=session, feed_dict=feed_dict)
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1}"\
                      .format(iter_cnt,loss))
            iter_cnt += 1
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {1}, Overall loss = {0}"\
              .format(total_loss, e+1))
    return total_loss


def get_predicted_image()