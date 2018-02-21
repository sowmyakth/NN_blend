"""Script to create CNN for retrieving isolated galaxy image from a two
galaxy blend"""
import os
import sys
# import skimage.io as io
import numpy as np
import math
import tensorflow as tf


class CNN_deblender(object):
    def __init__(self, ):
        self.optimizer = tf.train.AdamOptimizer(5e-4)
        self.sess = tf.Session()
        self.build_simple_model()
        self.sess.run(tf.global_variables_initializer())

    def get_mean_loss(self, y, y_out):
        total_loss = tf.nn.l2_loss(self.y - self.y_out)
        self.mean_loss = tf.reduce_mean(total_loss)

    def build_simple_model(self):
        """makes a simple 2 layer CNN
        layer 1 Conv 5*5*2s2, 256/ReLU
        layer 2 FC 32*32,1, 49
         """
        self.X = tf.placeholder(tf.float32, [None, 32, 32, 2])
        self.y = tf.placeholder(tf.float32, [None, 32, 32])
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
        self.y_out = tf.reshape(y_out, [-1, 32, 32]) + b1
        self.get_mean_loss()
        self.train_step = self.optimizer.minimize(self.mean_loss)

    def train(self, X_train, Y_train):
        variables = [self.train_step]
        feed_dict = {self.X: X_train[idx, :, :, :],
                     self.y: Y_train[idx, :, :]}
        self.sess.run(variables, feed_dict=feed_dict)

    def test(self, X_test):
        """Evaluates net for input X"""
        self.y_out.eval(session=self.sess,
                        feed_dict={X: X_test})

    def run_model(self, model, X_test, X_train, Y_train,
                  epochs=1, batch_size=64, print_every=100,
                  training=None, plot_losses=False):
        self.tot_loss = []
        # shuffle indicies
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)
        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        iter_cnt = 0
        for e in range(epochs):
            # keep track of losses and accuracy
            losses = []
            # make sure we iterate over the dataset once
            for i in range(int(math.ceil(Xd.shape[0] / batch_size))):
                # generate indicies for the batch
                start_idx = (i * batch_size)%Xd.shape[0]
                idx = train_indicies[start_idx:start_idx + batch_size]
                # create a feed dictionary for this batch
                if training is True:
                    self.train(X_train[idx, :, :, :],
                               Y_train[idx, :, :])
                else:
                    self.test(X_test)
                    self.get_mean_loss()
                # get batch size
                actual_batch_size = Y_test[idx].shape[0]  # fix this
                # aggregate performance stats
                losses.append(self.mean_loss * actual_batch_size)
                # print every now and then
                if training_now and (iter_cnt % print_every) == 0:
                    print("Iteration {0}: with minibatch training loss = {1}" \
                          .format(iter_cnt, loss))
                iter_cnt += 1
            self.tot_loss.append(np.sum(losses) / X_test.shape[0])  # fix
