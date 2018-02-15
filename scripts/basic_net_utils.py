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

class CNN_deblender(object):
    def __init__(self, ):

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




def get_predicted_image()