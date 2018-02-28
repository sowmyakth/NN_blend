"""Script to create CNN for retrieving isolated galaxy image from a two
galaxy blend"""
# import skimage.io as io
import os
import numpy as np
import math
import tensorflow as tf


class Meas_args(object):
    """Arguments to pass to model on how it should run.

    Keyword arguments:
    epochs      -- number of epochs to run (default:200)
    batch_size  -- number of input objects in each batch (default:16)
    print_every -- frequency to print loss  (default:100)
    """
    def __init__(self, epochs=200, batch_size=16,
                 print_every=100):
        self.epochs = epochs
        self.batch_size = batch_size
        self.print_every = print_every


def get_bi_weights(kernel_shape, name='weights'):
    """compute intialization weights here"""
    # Add computation here
    weights = tf.truncated_normal(kernel_shape, stddev=0.1)
    # init = tf.constant_initializer(value=weights,
    #                               dtype=tf.float32)
    # bi_weights = tf.get_variable(name="deconv_bi_kernel",
    #                             initializer=init,
    #                             shape=kernel_shape)
    bi_weights = tf.Variable(weights, name=name)
    return bi_weights


def get_conv_layer(input_layer, kernel_shape, name, stride=2):
    """Returns conv2d layer.
    Defines weight functions W, a tensor of shape kernel size
    Defines bias b, a tensor vector of shape number of kernels
    Creates conv layer
    """
    Wconv1 = tf.get_variable(name="W" + name, shape=kernel_shape)
    tf.summary.histogram("W" + name, Wconv1)
    bconv1 = tf.get_variable(name="b" + name, shape=kernel_shape[-1])
    tf.summary.histogram("b" + name, bconv1)
    layer = tf.nn.conv2d(input_layer, Wconv1,
                         strides=[1, stride, stride, 1],
                         padding='VALID', name=name) + bconv1
    return layer


class CNN_deblender(object):
    """Class to initialize and run CNN"""
    def __init__(self, num_cnn_layers=None, summary_hpram=0):
        self.num_cnn_layers = num_cnn_layers
        self.kernels = []
        self.biases = []
        self.activations = []
        self.sess = tf.Session()
        self.build_net()
        self.sess.run(tf.global_variables_initializer())
        self.initiate_writer(summary_hpram)

    def initiate_writer(self, summary_hpram):
        self.merged = tf.summary.merge_all()
        logdir = os.path.join(os.path.dirname(os.getcwd()),
                              "logfiles", str(summary_hpram))
        self.writer = tf.summary.FileWriter(logdir,
                                            self.sess.graph)
        self.writer.add_graph(self.sess.graph)

    def get_mean_loss(self):
        total_loss = tf.nn.l2_loss((self.y - self.y_out) * 100)
        self.mean_loss = tf.reduce_mean(total_loss)

    def simple_model(self):
        """makes a simple 2 layer CNN
        layer 1 Conv 5*5*2s2, 256/ReLU
        layer 2 FC 32*32,1, 49s
        """
        # weights for fully conected layer
        W1 = tf.get_variable("W1", shape=[32, 32, 1, 49])
        b1 = tf.get_variable("b1", shape=[32, 32])
        # define our graph (e.g. two_layer_convnet)
        a1 = get_conv_layer(self.X, [5, 5, 2, 256], "conv1")
        h1 = tf.nn.relu(a1)
        h1_flat = tf.reshape(h1, [32, 32, 49, -1])
        y_out = tf.matmul(W1, h1_flat)
        y_out = tf.transpose(y_out)
        self.y_out = tf.reshape(y_out, [-1, 32, 32]) + b1

    def simple_model2(self):
        """makes a simple 2 layer CNN
        layer 1 Conv 5*5*2s2, 256/ReLU
        layer 2 FC 32*32,1, 49s
        """
        # weights for fully conected layer
        # define our graph (e.g. two_layer_convnet)
        a1 = get_conv_layer(self.X, [2, 2, 2, 256], "conv1")
        layer1 = tf.nn.relu(a1)
        # Check this!!
        deconv_weights = get_bi_weights([2, 2, 1, 256])
        # shape = tf.Variable([-1, 32, 32, 1], dtype=tf.int32)
        in_shape = tf.shape(layer1)
        out_shape = tf.stack([in_shape[0], 32, 32, 1])
        # out_shape = tf.placeholder(tf.int32, [None, 32, 32, 1])
        print (deconv_weights, layer1)
        self.y_out = tf.nn.conv2d_transpose(layer1, deconv_weights,
                                            out_shape, strides=[1, 2, 2, 1],
                                            name="deconv", padding='VALID')

    def simple_model3(self):
        """makes a simple 2 layer CNN
        layer 1 Conv 5*5*2s2, 256/ReLU
        layer 2 FC 32*32,1, 49s
        """
        # weights for fully conected layer
        # define our graph (e.g. two_layer_convnet)
        a1 = get_conv_layer(self.X, [5, 5, 2, 32], "conv1", stride=1)
        layer1 = tf.nn.relu(a1)
        a2 = get_conv_layer(layer1, [3, 3, 32, 1], "conv2", stride=1)
        layer2 = tf.nn.relu(a2)
        # Check this!!
        # shape = tf.Variable([-1, 32, 32, 1], dtype=tf.int32)
        in_shape = tf.shape(layer2)
        out_shape = tf.stack([in_shape[0], 32, 32, 1])
        # out_shape = tf.placeholder(tf.int32, [None, 32, 32, 1])
        print (layer1, layer2)
        with tf.name_scope("deconv_layer"):
            deconv_weights = get_bi_weights([7, 7, 1, 1])
            tf.summary.histogram("deconv_weights", deconv_weights)
            self.y_out = tf.nn.conv2d_transpose(layer2, deconv_weights,
                                                out_shape,
                                                strides=[1, 1, 1, 1],
                                                name="transpose",
                                                padding='VALID')

    def basic_unit(self, input_layer, i):
        with tf.variable_scope("basic_unit" + str(i)):
            part1 = tf.layers.conv2d(input_layer, 32, [3, 3],
                                     padding='VALID',
                                     activation=tf.nn.relu,
                                     name="conv1")
            part2 = part1  # tf.layers.batch_norm(part1)
            part3 = tf.layers.conv2d(part2, 32, [3, 3],
                                     padding='VALID',
                                     activation=tf.nn.relu,
                                     name="conv2")
            return part3

    def multi_layer_model(self):
        """A 3 layer CNN
        Conv 3*3*2 s1, 32/ReLU
        [Conv 3*3 s1, 32/Relu] * 3
        deconv
        """
        with tf.variable_scope("first_layer"):
            first_cnn = get_conv_layer(self.X, [3, 3, 2, 32],
                                       "conv1", stride=1)
            layer_in = tf.nn.relu(first_cnn)
        for i in range(self.num_cnn_layers):
            layer_in = self.basic_unit(layer_in, i)
        with tf.variable_scope("last_layer"):
            last_cnn = get_conv_layer(layer_in, [3, 3, 32, 1],
                                      "conv_last", stride=1)
            layer_out = tf.nn.relu(last_cnn)
        # Check this!!
        deconv_weights = get_bi_weights([2, 2, 1, 1])
        # shape = tf.Variable([-1, 32, 32, 1], dtype=tf.int32)
        in_shape = tf.shape(layer_out)
        out_shape = tf.stack([in_shape[0], 32, 32, 1])
        # out_shape = tf.placeholder(tf.int32, [None, 32, 32, 1])
        with tf.name_scope("deconv_layer"):
            self.y_out = tf.nn.conv2d_transpose(layer_out, deconv_weights,
                                                out_shape,
                                                strides=[1, 2, 2, 1])

    def build_net(self):
        """makes a simple 2 layer CNN
        layer 1 Conv 5*5*2s2, 256/ReLU
        layer 2 FC 32*32,1, 49
         """
        with tf.name_scope("input"):
            self.X = tf.placeholder(tf.float32, [None, 32, 32, 2])
            self.y = tf.placeholder(tf.float32, [None, 32, 32, 1])
        # Run the preferred CNN model here
        if self.num_cnn_layers is not None:
            self.multi_layer_model()
        else:
            self.simple_model3()
        self.get_mean_loss()
        tf.summary.scalar("loss", self.mean_loss)
        with tf.name_scope("train"):
            self.optimizer = tf.train.AdamOptimizer(5e-4)
            self.train_step = self.optimizer.minimize(self.mean_loss)

    def train(self, feed_dict):
        variables = [self.mean_loss, self.train_step]
        loss, _ = self.sess.run(variables, feed_dict=feed_dict)
        return loss

    def get_kernel_bias_layer(self, graph,
                              layer_name):
        """Returns kernel values and bias of layers
        for a given scope name"""
        kernel = self.sess.run(graph.get_tensor_by_name(
            layer_name + '/kernel:0'))
        bias = self.sess.run(graph.get_tensor_by_name(
            layer_name + '/bias:0'))
        self.kernels.append(kernel)
        self.biases.append(bias)

    def get_relu_activations(self, graph,
                             layer_name):
        """Returns output from activation layer for a given input
        scope name """
        bar = graph.get_tensor_by_name(
            layer_name + '/Relu:0')
        activation = bar.eval(session=self.sess)
        self.activations.append(activation)

    def get_batch_norm_images(self, graph,
                              layer_name):
        """Returns output from batch normalization for a given input
        scope name """
        bar = graph.get_tensor_by_name(
            layer_name + '/Relu:0')
        activation = self.sess.run(bar)
        self.activations.append(activation)

    def get_interim_images(self):
        """Returns values of weights, biases and output images
        from interim activation layers.
        """
        gr = tf.get_default_graph()
        layer_name = 'first_layer/conv1'
        self.get_kernel_bias(gr, layer_name)
        self.get_relu_activations(gr, layer_name)
        for i in range(self.num_cnn_layers):
            layer_name = 'basic_unit{0}/conv1'.format(i)
            self.get_kernel_bias(gr, layer_name)
            self.get_relu_activations(gr, layer_name)
            # self.get_batch_norm_images(gr, layer_name)
            layer_name = 'basic_unit{0}/conv2'.format(i)
            self.get_kernel_bias(gr, layer_name)
            self.get_relu_activations(gr, layer_name)

    def get_summary(self, feed_dict, num):
        """evealuates summary items"""
        s = self.sess.run([self.merged],
                          feed_dict=feed_dict)
        self.writer.add_summary(s, num)

    def test(self, X_test, Y_test,
             get_interim_images=False):
        """Evaluates net for input X"""
        variables = [self.mean_loss]
        feed_dict = {self.X: X_test,
                     self.y: Y_test}
        loss = self.sess.run(variables, feed_dict=feed_dict)
        if get_interim_images:
            self.get_interim_images()
        return loss

    def run_model(self, X_train, Y_train,
                  Args, X_test=None, Y_test=None):
        # shuffle indicies
        train_indicies = np.arange(X_train.shape[0])
        np.random.shuffle(train_indicies)
        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        iter_cnt = 0
        train_loss, test_loss = [], []
        for e in range(Args.epochs):
            # print("running epoch ", e)
            # keep track of losses
            # make sure we iterate over the dataset once
            # fix this
            for i in range(int(math.ceil(X_test.shape[0] / Args.batch_size))):
                # generate indicies for the batch
                # Fix this
                start_idx = (i * Args.batch_size) % X_train.shape[0]
                idx = train_indicies[start_idx:start_idx + Args.batch_size]
                # create a feed dictionary for this batch
                feed_dict = {self.X: X_train[idx, :, :, :],
                             self.y: Y_train[idx, :, :, :]}
                loss = self.train(feed_dict)
                # print every now and then
                if (iter_cnt % Args.print_every) == 0:
                    print("Iteration {0}: with minibatch training loss = {1}"
                          .format(iter_cnt, loss))
                    self.get_summary(feed_dict, i)
                iter_cnt += 1
            # save training and test loss every epoch
            train_loss.append(loss)
            if X_test is not None:
                loss = self.test(X_test, Y_test)
                test_loss.append(loss)
        feed_dict = {self.X: X_test,
                     self.y: Y_test}
        pred = self.y_out.eval(session=self.sess, feed_dict=feed_dict)
        return train_loss, test_loss, pred
