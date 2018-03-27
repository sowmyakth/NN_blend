import numpy as np
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
    bi_weights = tf.Variable(name=name, initializer=weights)
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
    W = tf.get_variable(initializer=tf.truncated_normal(kernel_shape,
                                                        stddev=0.1),
                        name="W" + name)
    tf.summary.histogram("W" + name + "_summ", W)
    b = tf.get_variable(intialization=tf.truncated_normal([kernel_shape[-1]],
                                                          stddev=0.1),
                        name="b" + name)
    tf.summary.histogram("b" + name + "_summ", b)
    conv = tf.nn.conv2d(input_layer, W,
                        strides=[1, stride, stride, 1],
                        padding='VALID', name=name) + b
    return conv


def get_deconv_ouput_shape(input_layer, kernel_shape):
    """This is for strode=1 and padding=valid only"""
    in_shape = input_layer.get_shape().as_list()
    channels = kernel_shape[-2]
    height = in_shape[1] + kernel_shape[0] - 1
    width = in_shape[2] + kernel_shape[1] - 1
    batch = tf.shape(input_layer)[0]
    out_shape = tf.stack([batch, height, width, channels])
    return out_shape


def get_deconv_layer(input_layer, kernel_shape, name, stride=2):
    """Returns conv2d layer.
    Defines weight functions W, a tensor of shape kernel size
    Defines bias b, a tensor vector of shape number of kernels
    Creates conv layer
    """
    # Wconv1 = tf.get_variable(name="W" + name, shape=kernel_shape)

    W = tf.get_variable(initializer=tf.truncated_normal(kernel_shape,
                                                        stddev=0.1),
                        name="W" + name)
    tf.summary.histogram("W" + name + "_summ", W)
    b = tf.get_variable(initializer=tf.truncated_normal([kernel_shape[-2]],
                                                        stddev=0.1),
                        name="b" + name)
    tf.summary.histogram("b" + name + "_summ", b)
    out_shape = get_deconv_ouput_shape(input_layer, kernel_shape)
    print (out_shape)
    conv = tf.nn.conv2d_transpose(input_layer, W, out_shape,
                                  strides=[1, stride, stride, 1],
                                  padding='VALID', name=name) + b
    print (conv.get_shape().as_list())
    return conv


def make_BatchNorm_layer(layer_in, train):
    # Perform a batch normalization after a conv layer or a fc layer
    # gamma: a scale factor
    # beta: an offset
    # epsilon: the variance epsilon - a small float number to avoid dividing by 0
    with tf.variable_scope('BatchNorm'):
        depth = layer_in.get_shape().as_list()[-1]
        gamma = tf.get_variable("gamma", depth,
                                initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta", [depth],
                               initializer=tf.constant_initializer(0.0))
        moving_avg = tf.get_variable("moving_avg", [depth],
                                     initializer=tf.constant_initializer(0.0),
                                     trainable=False)
        moving_var = tf.get_variable("moving_var", [depth],
                                     initializer=tf.constant_initializer(1.0),
                                     trainable=False)
        # if train:
        avg, var = tf.nn.moments(layer_in, axes=[0, 1, 2])
        moving_avg.assign(avg)
        moving_var.assign(var)
        # update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
        # update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
        # control_inputs = [update_moving_avg, update_moving_var]
        #else:
        # avg = moving_avg
        #var = moving_var
        #with tf.control_dependencies(control_inputs):
        output = tf.nn.batch_normalization(layer_in, moving_avg, moving_var,
                                           offset=beta, scale=gamma,
                                           variance_epsilon=1e-4,
                                           name='batchnorm')
        tf.summary.histogram("avg_summ", moving_avg)
        tf.summary.histogram("var_summ", moving_var)
        tf.summary.histogram("gamma_summ", gamma)
        tf.summary.histogram("beta_summ", beta)
    return output
