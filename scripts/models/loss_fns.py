import tensorflow as tf


def get_l2_loss(y, y_out):
    """Return l2 loss
    Keyword arguments:
        y     -- true vale
        y_out -- value output from net
    Returns L2 loss
        loss = sum((y-yout)**2)/2

    """
    diff = tf.subtract(y, y_out, name='diff')
    loss = tf.nn.l2_loss(diff, name='l2_loss')
    return loss


def get_l1_loss(y, y_out):
    """Return l2 loss
    Keyword arguments:
        y     -- true vale
        y_out -- value output from net
    Returns L2 loss
        loss = sum((y-yout)**2)/2

    """
    diff = tf.subtract(y, y_out, name='diff')
    loss = tf.reduce_sum(tf.abs(diff), name='l1_loss')
    return loss


def get_chi_sq_loss(y, y_out):
    """Return l2 loss
    Keyword arguments:
        y     -- true vale
        y_out -- value output from net
    Returns L2 loss
        loss = sum((y-yout)**2)/2

    """
    diff = tf.subtract(y, y_out, name='diff')
    loss = tf.reduce_sum(diff**2 / y, name='chi_sq_loss')
    return loss
