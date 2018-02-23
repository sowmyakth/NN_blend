"""Script to create CNN for retrieving isolated galaxy image from a two
galaxy blend"""
# import skimage.io as io
import numpy as np
import math
import tensorflow as tf

def get_deconv_layer():

class CNN_deblender(object):
    """Class to initialize and run CNN"""
    def __init__(self, ):
        self.optimizer = tf.train.AdamOptimizer(5e-4)
        self.sess = tf.Session()
        self.build_net()
        self.sess.run(tf.global_variables_initializer())

    def get_mean_loss(self, y, y_out):
        total_loss = tf.nn.l2_loss(self.y - self.y_out)
        self.mean_loss = tf.reduce_mean(total_loss)

    def simple_model(self):
        """makes a simple 2 layer CNN
        layer 1 Conv 5*5*2s2, 256/ReLU
        layer 2 FC 32*32,1, 49
         """
        Wconv1 = tf.get_variable("Wconv1", shape=[5, 5, 2, 256])
        bconv1 = tf.get_variable("bconv1", shape=[256])
        W1 = tf.get_variable("W1", shape=[32, 32, 1, 49])
        b1 = tf.get_variable("b1", shape=[32, 32])
        # define our graph (e.g. two_layer_convnet)
        a1 = tf.nn.conv2d(self.X, Wconv1, strides=[1, 2, 2, 1],
                          padding='VALID') + bconv1
        h1 = tf.nn.relu(a1)
        h1_flat = tf.reshape(h1, [32, 32, 49, -1])
        y_out = tf.matmul(W1, h1_flat)
        y_out = tf.transpose(y_out)
        self.y_out = tf.reshape(y_out, [-1, 32, 32]) + b1

    def basic_unit(input_layer, i):
        with tf.variable_scope("basic_unit" + str(i)):
            part1 = tf.layers.conv2d(input_layer, 32, [3, 3],
                                     padding='VALID',
                                     activation=tf.nn.relu)
            part2 = tf.layers.batch_norm(part1,
                                         activation=tf.nn.relu)
            part3 = tf.layers.conv2d(part2, 32, [3, 3],
                                     padding='VALID',
                                     activation=tf.nn.relu)
            return part3

    def multi_layer_model(self, num_layers):
        """A 3 layer CNN
        Conv 3*3*2 s1, 32/ReLU
        [Conv 3*3 s1, 32/Relu] * 3
        deconv
          """
        layer_in = tf.layers.conv2d(self.X, 32, [3, 3, 2],
                                    padding='VALID',
                                    activation=tf.nn.relu,
                                    scope="conv_0")
        for i in range(num_layers):
            layer_in = basic_unit(layer_in, i)
        top = tf.nn.conv2d_transpose(layer_in, deconv_weights)


    def build_net(self):
        """makes a simple 2 layer CNN
        layer 1 Conv 5*5*2s2, 256/ReLU
        layer 2 FC 32*32,1, 49
         """
        self.X = tf.placeholder(tf.float32, [None, 32, 32, 2])
        self.y = tf.placeholder(tf.float32, [None, 32, 32])
        # Run the preferred CNN model here
        self.simple_model()
        self.get_mean_loss()
        self.train_step = self.optimizer.minimize(self.mean_loss)

    def train(self, X_train, Y_train):
        variables = [self.train_step]
        feed_dict = {self.X: X_train,
                     self.y: Y_train}
        self.sess.run(variables, feed_dict=feed_dict)

    def test(self, X_test):
        """Evaluates net for input X"""
        self.y_out.eval(session=self.sess,
                        feed_dict={self.X: X_test})

    def run_model(self, X_test, X_train, Y_train,
                  epochs=1, batch_size=64, print_every=100,
                  training=None, plot_losses=False):
        # shuffle indicies
        train_indicies = np.arange(X_train.shape[0])
        np.random.shuffle(train_indicies)
        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        iter_cnt = 0
        train_loss, test_loss = [], []
        for e in range(epochs):
            # keep track of losses and accuracy
            # make sure we iterate over the dataset once
            for i in range(int(math.ceil(X_test.shape[0] / batch_size))):  ## fix 
                # generate indicies for the batch
                start_idx = (i * batch_size)%X_train.shape[0]  #fix
                idx = train_indicies[start_idx:start_idx + batch_size]
                # create a feed dictionary for this batch
                self.train(X_train[idx, :, :, :],
                           Y_train[idx, :, :])
                loss = self.get_mean_loss()
                # print every now and then
                if (iter_cnt % print_every) == 0:
                    print("Iteration {0}: with minibatch training loss = {1}" \
                          .format(iter_cnt, loss))
                iter_cnt += 1
            # save training and test loss every epoch
            train_loss.append(loss)
            self.test(X_test)
            test_loss.append(self.get_mean_loss())
        return train_loss, test_loss