"""Set of unit tests for the CNNN deblender network"""
import os
import basic_run
import numpy as np
import tensorflow as tf
import basic_net_utils as utils
import shutil


def load_net(name):
    """Tests if CNN deblender model can be loaded"""
    # tf.reset_default_graph()
    # model1 = utils.CNN_deblender()
    model = utils.CNN_deblender(config=True, num_cnn_layers=6,
                                run_ident=name)
    print ("Model successfully loaded")
    # run_params = utils.Meas_args(epochs=Args.epochs,
    #                             batch_size=Args.batch_size,
    #                             print_every=500)


def load_data():
    """Check if input data has correct dimensions
    Input has dimensions[batch_size, image_x, image_y, num_filters]
    """
    path = '/global/cscratch1/sd/sowmyak/training_data'
    filename = os.path.join(path, 'stamps.npz')
    inputs = basic_run.load_data(filename)
    for i in range(len(inputs)):
        np.testing.assert_array_equal(inputs[i].shape[1:3],
                                      [32, 32],
                                      err_msg="Arrays are not 32*32")
    np.testing.assert_equal(inputs[0].shape[-1],
                            inputs[2].shape[-1],
                            err_msg="train and test input dont match")
    np.testing.assert_equal(inputs[1].shape[-1],
                            inputs[3].shape[-1],
                            err_msg="train and test output dont match")
    np.testing.assert_array_equal(inputs[0].shape,
                                  inputs[1].shape,
                                  err_msg="Train input and outputs dont match")
    np.testing.assert_array_equal(inputs[2].shape,
                                  inputs[3].shape,
                                  err_msg="Test input and outputs dont match")
    print("Data correctly loaded")


def test_summary_save(model):
    """Test if summary is correctly saved"""
    fname = os.path.join(os.path.dirname(os.getcwd()),
                         "logfiles", model.run_ident)
    assert (os.path.isdir(fname)), "Summary save FAILED!"
    shutil.rmtree(fname)



def test_model_save(model):
    """Test that model save to file is successful"""
    name = model.run_ident
    model.save()
    model.sess.close()
    path = os.path.join(os.path.dirname(os.getcwd()), "outputs",
                            "models", model.run_ident)
    fname = os.path.join(path, "model")
    assert (os.path.isdir(fname)), "Model save FAILED!"
    shutil.rmtree(fname)


def main():
    """All unit test functions are called here"""
    name = 'unit_test'
    model = load_net(name)
    inputs = load_data()
    test_model_save(model)
    print ("All tests passed")


if __name__ == "__main__":
    main()
