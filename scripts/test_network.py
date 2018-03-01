"""Set of unit tests for the CNNN deblender network"""
from data_utils import get_data
import numpy as np
import tensorflow as tf
import basic_net_utils as utils


def load_net():
    """Tests if CNN deblender model can be loaded"""
    tf.reset_default_graph()
    model = utils.CNN_deblender()
    assert len(model.biases) == 0, "biases list is not empty"
    assert len(model.kernels) == 0, "biases list is not empty"
    assert len(model.activations) == 0, "biases list is not empty"


def load_data():
    """Check if input data has correct dimensions"""
    inputs = get_data()
    for i in range(len(inputs)):
        np.testing.assert_array_equal(inputs[i].shape[1:3],
                                      [32, 32],
                                      err_msg="Arrays are not 32*32")
    np.testing.assert_equal(inputs[0].shape[-1],
                            inputs[2].shape[-1],
                            err_msg="train and test data dont match")


def main():
    """All unit test functions are called here"""
    load_net()
    # load_data()
    print ("All tests passed")


if __name__ == "__main__":
    main()
