"""Script to create CNN for retrieving isolated galaxy image from a two
galaxy blend"""
# import skimage.io as io
import os
import numpy as np
import math
import tensorflow as tf
import subprocess


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


def make_layer(layer, num_filters):
    a2 = get_conv_layer(layer, [3, 3, num_filters, num_filters],
                        "conv", stride=1)
    act = tf.nn.crelu(a2, name='act')
    tf.summary.histogram("act_summ", act)
    return act


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


def get_individual_loss(y, y_out):
    """Returns loss of each object
    Keyword Arguments:
        y     -- True isolated galaxy image as numpy array
        y_out -- Output of net as numpy array
    Returns
        L2 loss for each galaxy
    """
    diff = np.subtract(y, y_out)**2
    loss = diff.sum(axis=3).sum(axis=1).sum(axis=1)
    return loss / 2.


def get_conv_layer(input_layer, kernel_shape, name, stride=2):
    """Returns conv2d layer.
    Defines weight functions W, a tensor of shape kernel size
    Defines bias b, a tensor vector of shape number of kernels
    Creates conv layer
    """
    # Wconv1 = tf.get_variable(name="W" + name, shape=kernel_shape)
    W = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1),
                    name="W" + name,)
    tf.summary.histogram("W" + name + "_summ", W)
    # bconv1 = tf.get_variable(name="b" + name, shape=kernel_shape[-1])
    b = tf.Variable(tf.truncated_normal([kernel_shape[-1]], stddev=0.1),
                    name="b" + name)
    tf.summary.histogram("b" + name + "_summ", b)
    conv = tf.nn.conv2d(input_layer, W,
                        strides=[1, stride, stride, 1],
                        padding='VALID', name=name) + b
    return conv


class CNN_deblender(object):
    """Class to initialize and run CNN"""
    def __init__(self, num_cnn_layers=None, run_ident=0,
                 bands=3, learning_rate=1e-3, config=True):
        self.num_cnn_layers = num_cnn_layers
        self.run_ident = str(run_ident)
        self.bands = bands
        self.learning_rate = learning_rate
        self.kernels = []
        self.biases = []
        self.activations = []
        tf.reset_default_graph()
        if config is True:
            inter = os.environ['NUM_INTER_THREADS']
            intra = os.environ['NUM_INTRA_THREADS']
            print("Custom NERSC/Intel config op_parallelism_threads:inters({}), intra ({})".format(inter, intra))
            config = tf.ConfigProto(inter_op_parallelism_threads=int(inter),
                                    intra_op_parallelism_threads=int(intra))
            self.sess = tf.Session(config=config)
        else:
            self.sess = tf.Session()
        self.build_net()
        self.merged = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())
        self.initiate_writer()

    def initiate_writer(self):
        logdir = os.path.join(os.path.dirname(os.getcwd()),
                              "logfiles", self.run_ident)
        self.writer = tf.summary.FileWriter(logdir)
        self.writer.add_graph(self.sess.graph)

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

    def simple_model3(self):
        """makes a simple 2 layer CNN
        layer 1 Conv 5*5*2s2, 256/ReLU
        layer 2 FC 32*32,1, 49s
        """
        # weights for fully conected layer
        # define our graph (e.g. two_layer_convnet)
        with tf.name_scope("conv_layer1"):
            a1 = get_conv_layer(self.X, [5, 5, self.bands, 32],
                                name="conv1", stride=1)
            layer1 = tf.nn.crelu(a1)
        with tf.name_scope("conv_layer2"):
            a2 = get_conv_layer(layer1, [3, 3, 64, 1], "conv2", stride=1)
            layer2 = tf.nn.crelu(a2)
        with tf.name_scope("deconv_layer"):
            deconv_weights = get_bi_weights([7, 7, 1, 2])
            out_shape = tf.stack([tf.shape(layer2)[0], 32, 32, 1])
            tf.summary.histogram("deconv_weights", deconv_weights)
            layer3 = tf.nn.conv2d_transpose(layer2, deconv_weights,
                                            out_shape,
                                            strides=[1, 1, 1, 1],
                                            name="transpose",
                                            padding='VALID')
            # y_out = tf.nn.crelu(layer3)
            y_out = layer3
            return y_out

    def simple_model4(self):
        print ("Initializing simple model4")
        with tf.name_scope("conv_layer_in"):
            num_filters = 4
            a1 = get_conv_layer(self.X, [5, 5, self.bands, num_filters],
                                name="conv", stride=1)
            layer = tf.nn.crelu(a1, name='act')
            tf.summary.histogram("act_summ", layer)
        for i in range(6):
            num_filters *= 2
            with tf.name_scope("conv_layer" + str(i)):
                layer = make_layer(layer, num_filters)
        with tf.name_scope("deconv_layer"):
            deconv_weights = get_bi_weights([2, 2, 1, num_filters * 2])
            tf.summary.histogram("deconv_weights", deconv_weights)
            out_shape = tf.stack([tf.shape(layer)[0], 32, 32, 1])
            layer_out = tf.nn.conv2d_transpose(layer, deconv_weights,
                                               out_shape,
                                               strides=[1, 2, 2, 1],
                                               name="transpose",
                                               padding='VALID')
            y_out = layer_out
            return y_out

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
        # shape = tf.Variable([-1, 32, 32, 1], dtype=tf.int32)
        # in_shape = tf.shape(layer_out)
        # out_shape = tf.placeholder(tf.int32, [None, 32, 32, 1])
        with tf.name_scope("deconv_layer"):
            deconv_weights = get_bi_weights([2, 2, 1, 1])
            out_shape = tf.stack([tf.shape(self.X)[0], 32, 32, 1])
            self.y_out = tf.nn.conv2d_transpose(layer_out, deconv_weights,
                                                out_shape,
                                                strides=[1, 2, 2, 1])

    def build_net(self):
        """makes a simple 2 layer CNN
        layer 1 Conv 5*5*2s2, 256/ReLU
        layer 2 FC 32*32,1, 49
         """
        with tf.name_scope("input"):
            self.X = tf.placeholder(tf.float32, [None, 32, 32, self.bands])
            self.y = tf.placeholder(tf.float32, [None, 32, 32, 1])
            tf.summary.image('X', self.X)
            tf.summary.image('y', self.y)
        # Run the preferred CNN model here
        if self.num_cnn_layers is not None:
            # self.multi_layer_model()
            self.y_out = self.simple_model4()
        else:
            self.y_out = self.simple_model3()
        with tf.name_scope("loss_function"):
            diff = tf.subtract(self.y, self.y_out, name='diff')
            loss = tf.nn.l2_loss(diff, name='l2_loss')
            num = tf.cast(tf.shape(self.y)[0], tf.float32, name="number")
            self.mean_loss = tf.divide(loss, num, name="mean_loss")
            tf.summary.scalar("train_loss_summ", self.mean_loss)
        with tf.name_scope("train"):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
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
        # gr = tf.get_default_graph()
        gr = self.sess.graph
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
        [s] = self.sess.run([self.merged],
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

    def run_basic(self, X_train, Y_train,
                  Args, X_test=None, Y_test=None):
        # shuffle indicies
        train_indicies = np.arange(X_train.shape[0])
        np.random.shuffle(train_indicies)
        iter_cnt = 0
        train_loss, test_loss = [], []
        num_mini_batches = int(math.ceil(X_train.shape[0] / Args.batch_size))
        for e in range(Args.epochs):
            # print("running epoch ", e)
            # keep track of losses
            # make sure we iterate over the dataset once
            epoch_loss = []
            assert len(epoch_loss) == 0, "Epoch loss must be empty"
            for i in range(num_mini_batches):
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
                epoch_loss.append(loss)
            # save training and test loss every epoch
            assert len(epoch_loss) == num_mini_batches, "Wrong size"
            train_loss.append(np.mean(epoch_loss))
            if X_test is not None:
                loss = self.test(X_test, Y_test)
                test_loss.append(loss)
        feed_dict = {self.X: X_test,
                     self.y: Y_test}
        pred = self.y_out.eval(session=self.sess, feed_dict=feed_dict)
        ind_loss = get_individual_loss(Y_test, pred)
        return train_loss, test_loss, pred, ind_loss

    def restore(self, filename):
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.sess, filename)

    def save(self):
        path = os.path.join(os.path.dirname(os.getcwd()), "outputs",
                            "models", self.run_ident)
        subprocess.call(['mkdir', path])
        fname = os.path.join(path, "model")
        saver = tf.train.Saver(tf.global_variables())
        saver.save(self.sess, fname)
        return
