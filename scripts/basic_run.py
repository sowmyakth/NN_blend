""" Basic Script to run CNN deblender"""


import os
from __future__ import division
import basic_net_utils as utils
import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    train_data = np.load(filename)
    X_train = train_data['X_train']
    Y_train = train_data['Y_train']
    X_val = train_data['X_val']
    Y_val = train_data['Y_val']
    return X_train, Y_train, X_val, Y_val


def save_diff_blend(pred, Y_val):
    path = os.path.join(os.path.dirname(os.getcwd()), "data")
    filename = os.path.join(path, 'diff')
    diff = (Y_val - pred)
    diff_val =np.sum(np.sum(diff[:,:,:,0], axis=1), axis=1)
    np.save(filename, diff_val)
    return diff_val


def plot_preds(pred, X_val, Y_val):
    for num in range(0, 10):
        plt.figure(figsize=[10, 6])
        plt.subplot(5, 5, 1)
        plt.imshow(pred[num, :, :, 0])
        plt.colorbar()
        plt.title('Network output')
        plt.subplot(5, 5, 2)
        plt.imshow(Y_val[num, :, :, 0])
        plt.colorbar()
        plt.title('Training central galaxy (I band)')
        plt.subplot(5, 5, 3)
        plt.imshow(pred[num, :, :, 0] - Y_val[num, :, :, 0])
        plt.colorbar()
        plt.title('Output - truth')
        plt.subplot(5, 5, 4)
        plt.imshow(X_val[num, :, :, 0])
        plt.colorbar()
        plt.title('Training galaxy blend (i band)')
        plt.subplot(5, 5, 5)
        plt.imshow(X_val[num, :, :, 0] - Y_val[num, :, :, 0])
        plt.colorbar()
        plt.title('truths diff')
        plt.show()


def main():
    run_ident = 'test_3filter2'
    path = os.path.join(os.path.dirname(os.getcwd()), "data")
    filename = os.path.join(path, 'training_data.npz')
    X_train, Y_train, X_val, Y_val = load_data(filename)
    model = utils.CNN_deblender(run_ident=run_ident)
    run_params = utils.Meas_args(epochs=100, batch_size=32)
    train_loss, val_loss, pred = model.run_basic(X_train, Y_train,
                                                 run_params, X_val, Y_val)
    model.save()
    model.sess.close()
    plt.plot(train_loss, label='training')
    plt.plot(val_loss, '.', label='validation', alpha=0.5)
    plt.ylabel('mean loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    plot_preds(pred, X_val, Y_val)


if __name__ == "__main__":
    main()
