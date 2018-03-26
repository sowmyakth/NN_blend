"""Script to create CNN for retrieving isolated galaxy image from a two
galaxy blend"""
# import skimage.io as io
import os
import numpy as np
import math
import tensorflow as tf
import subprocess
import lilac.utils as utils
import lilac.loss_fns as loss_fns


class CNN_deblender(object):
    """Class to initialize and run CNN"""
    def __init__(self, num_cnn_layers=None, run_ident=0,
                 bands=3, learning_rate=1e-3, config=False,
                 loss_fn="l2"):
        self.num_cnn_layers = num_cnn_layers
        self.run_ident = str(run_ident)
        self.bands = bands
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
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

    def make_conv_unit(self, in_layer, num_filters):
        """ Makes a layer comprising of convolution, batch normalization and crelu
        activation.
        Keyword Arguments:
            in_layer    -- input tensor to the layer
            num_filters -- Number of convolution filters.
        Returns
            tensor output of the layer
        """
        print(in_layer.get_shape().as_list())
        a1 = utils.get_conv_layer(in_layer, [3, 3, num_filters, num_filters],
                                  "conv", stride=1)
        a2 = utils.make_BatchNorm_layer(a1, self.is_training)
        act = tf.nn.crelu(a2, name='act')
        tf.summary.histogram("act_summ", act)
        return act

    def make_deconv_unit(self, in_layer, num_filters):
        """ Makes a layer comprising of convolution, batch normalization and crelu
        activation.
        Keyword Arguments:
            in_layer    -- input tensor to the layer
            num_filters -- Number of convolution filters.
        Returns
            tensor output of the layer
        """
        print(in_layer.get_shape().as_list())
        kernel_shape = [3, 3, int(num_filters / 2.), num_filters * 2]
        a1 = utils.get_deconv_layer(in_layer,
                                    kernel_shape,
                                    "deconv", stride=1)
        a2 = utils.make_BatchNorm_layer(a1, self.is_training)
        act = tf.nn.crelu(a2, name='act')
        tf.summary.histogram("act_summ", act)
        return act

    def multi_layer_model(self):
        print ("Initializing multi_layer_model")
        with tf.name_scope("conv_layer_in"):
            print ("Conv layer in ")
            print(self.X.get_shape().as_list())
            num_filters = 2
            a1 = utils.get_conv_layer(self.X, [5, 5, self.bands, num_filters],
                                      name="conv", stride=1)
            layer = tf.nn.crelu(a1, name='act')
            tf.summary.histogram("act_summ", layer)
        for i in range(5):
            num_filters *= 2
            with tf.variable_scope("conv_layer_" + str(i)):
                print("Conv layer {} ".format(i))
                layer = self.make_conv_unit(layer, num_filters)
        for i in range(5):
            with tf.variable_scope("deconv_layer_" + str(i)):
                print("Deconv layer {} ".format(i))
                layer = self.make_deconv_unit(layer, num_filters)
            num_filters /= 2
        with tf.name_scope("deconv_layer_out"):
            print ("Deconv layer out")
            print(layer.get_shape().as_list())
            # deconv_weights = get_bi_weights([5, 5, 1, 4])
            # tf.summary.histogram("deconv_weights", deconv_weights)
            # out_shape = tf.stack([tf.shape(layer)[0], 32, 32, 1])
            # layer_out = tf.nn.conv2d_transpose(layer, deconv_weights,
            #                                    out_shape,
            #                                   strides=[1, 1, 1, 1],
            #                                   name="transpose",
            #                                   padding='VALID')
            layer_out = utils.get_deconv_layer(layer, [5, 5, 1, 4],
                                               "deconv", stride=1)
            return layer_out

    def build_net(self):
        """makes a simple 2 layer CNN
        layer 1 Conv 5*5*2s2, 256/ReLU
        layer 2 FC 32*32,1, 49
         """
        with tf.name_scope("input"):
            self.is_training = tf.placeholder(tf.bool)
            self.X = tf.placeholder(tf.float32, [None, 32, 32, self.bands])
            self.y = tf.placeholder(tf.float32, [None, 32, 32, 1])
            tf.summary.image('X', self.X)
            tf.summary.image('y', self.y)
        # Run the preferred CNN model here
        if self.num_cnn_layers is not None:
            # self.multi_layer_model()
            self.y_out = self.multi_layer_model()
        else:
            self.y_out = self.simple_model3()
        with tf.name_scope("loss_function"):
            if self.loss_fn == 'l1':
                loss = loss_fns.get_l1_loss(self.y,
                                            self.y_out)
            elif self.loss_fn == 'l2':
                loss = loss_fns.get_l2_loss(self.y,
                                            self.y_out)
            elif self.loss_fn == 'chi_sq':
                loss = loss_fns.get_chi_sq_loss(self.y,
                                                self.y_out)
            else:
                raise ValueError('Unknown loss_fn')
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

    def get_summary(self, feed_dict, num):
        """evealuates summary items"""
        [s] = self.sess.run([self.merged],
                            feed_dict=feed_dict)
        self.writer.add_summary(s, num)

    def test(self, X_test, Y_test):
        """Evaluates net for input X"""
        variables = [self.mean_loss]
        feed_dict = {self.X: X_test,
                     self.y: Y_test}
        loss = self.sess.run(variables, feed_dict=feed_dict)
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
            epoch_loss = []
            assert len(epoch_loss) == 0, "Epoch loss must be empty"
            for i in range(num_mini_batches):
                # generate indicies for the batch
                # Fix this
                start_idx = (i * Args.batch_size) % X_train.shape[0]
                idx = train_indicies[start_idx:start_idx + Args.batch_size]
                # create a feed dictionary for this batch
                feed_dict = {self.X: X_train[idx, :, :, :],
                             self.y: Y_train[idx, :, :, :],
                             self.is_training: True}
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
                     self.y: Y_test,
                     self.is_training: False}
        pred = self.y_out.eval(session=self.sess, feed_dict=feed_dict)
        ind_loss = utils.get_individual_loss(Y_test, pred)
        return train_loss, test_loss, pred, ind_loss

    def restore(self, filename=None):
        if filename is None:
            path = os.path.join(os.path.dirname(os.getcwd()), "outputs",
                                "models", self.run_ident)
            filename = os.path.join(path, "model")
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.sess, filename)

    def save(self):
        path = os.path.join(os.path.dirname(os.getcwd()), "outputs",
                            "models", self.run_ident)
        if os.path.isdir(path) is False:
            subprocess.call(['mkdir', path])
        fname = os.path.join(path, "model")
        saver = tf.train.Saver(tf.global_variables())
        saver.save(self.sess, fname)
        return
