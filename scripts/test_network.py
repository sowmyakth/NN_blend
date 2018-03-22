"""Set of unit tests for the CNNN deblender network"""
import os
import basic_run
import numpy as np
import basic_net_utils as utils
import shutil
import galsecom.loss_fns
import tensorflow as tf


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_tf_loss(y_np, y_out_np):
    tf.reset_default_graph()
    y = tf.Variable(y_np)
    y_out = tf.Variable(y_out_np)
    l1_tf = galsecom.loss_fns.get_l1_loss(y, y_out)
    l2_tf = galsecom.loss_fns.get_l2_loss(y, y_out)
    cs_tf = galsecom.loss_fns.get_chi_sq_loss(y, y_out)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        return session.run([l1_tf, l2_tf, cs_tf])


def test_loss_fn():
    """Tests loss functions"""
    y_np = np.array([1., 2.])
    y_out_np = np.array([1.1, 2.2])
    l1 = np.sum(np.abs(y_np - y_out_np))
    l2 = np.sum((y_np - y_out_np)**2) / 2.
    cs = np.sum((y_np - y_out_np)**2 / y_np)
    l1_tf, l2_tf, cs_tf = get_tf_loss(y_np, y_out_np)
    assert (l1 == l1_tf), "Error in L1 loss"
    assert (l2 == l2_tf), "Error in L2 loss"
    assert (cs == cs_tf), "Error in Chi squared loss"
    print ("Loss functions test passed")


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
    return model


def load_data():
    """Laods training and test input and outputs"""
    path = '/global/cscratch1/sd/sowmyak/training_data'
    filename = os.path.join(path, 'stamps.npz')
    inputs = basic_run.load_data(filename)
    return inputs


def test_data(inputs):
    """Check if input data has correct dimensions
    Input has dimensions[batch_size, image_x, image_y, num_filters]
    """
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
    np.testing.assert_array_equal(inputs[0].shape[0:3],
                                  inputs[1].shape[0:3],
                                  err_msg="Train input and outputs dont match")
    np.testing.assert_array_equal(inputs[2].shape[0:3],
                                  inputs[3].shape[0:3],
                                  err_msg="Test input and outputs dont match")
    print("Data correctly loaded")


def test_summary_save(model):
    """Test if summary is correctly saved"""
    fname = os.path.join(os.path.dirname(os.getcwd()),
                         "logfiles", model.run_ident)
    assert (os.path.isdir(fname)), "Summary save FAILED!"
    shutil.rmtree(fname)
    print ("Summary save passed")


def test_model_save(model):
    """Test that model save to file is successful"""
    path = os.path.join(os.path.dirname(os.getcwd()), "outputs",
                        "models", model.run_ident)
    assert (os.path.isdir(path)), "Model save FAILED!"
    shutil.rmtree(path)
    print ("Model save passed")


def test_model():
    name = 'unit_test'
    model = load_net(name)
    model.save()
    model.sess.close()
    test_model_save(model)
    test_summary_save(model)


def main():
    """All unit test functions are called here"""
    test_loss_fn()
    inputs = load_data()
    test_data(inputs)
    test_model()
    print (bcolors.OKGREEN + "All tests passed" + bcolors.ENDC)


if __name__ == "__main__":
    main()
