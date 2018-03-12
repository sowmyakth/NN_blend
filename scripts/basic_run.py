""" Basic Script to run CNN deblender"""
from __future__ import division
import os
import basic_net_utils as utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def load_data(filename):
    train_data = np.load(filename)
    X_train = train_data['X_train']
    Y_train = train_data['Y_train']
    X_val = train_data['X_val']
    Y_val = train_data['Y_val']
    return X_train, Y_train, X_val, Y_val


def save_diff_blend(pred, Y_val, ident):
    path = os.path.join(os.path.dirname(os.getcwd()), "outputs")
    filename = os.path.join(path, 'diff_' + ident)
    diff = (Y_val - pred)
    diff_val = np.sum(np.sum(diff[:, :, :, 0], axis=1), axis=1)
    np.save(filename, diff_val)
    return diff_val


def get_rgb(im):
    min_val = [np.min(im[:, :, i]) for i in range(3)]
    new_im = [np.sqrt(im[:, :, i] + min_val[i]) for i in range(3)]
    norm = np.max(new_im)
    new_im = [new_im[i].T / norm * 255 for i in range(3)]
    new_im = np.array(new_im, dtype='u1')
    return new_im.T


def plot_loss(train_loss, val_loss, ident):
    path = os.path.join(os.path.dirname(os.getcwd()), "outputs")
    filename = os.path.join(path, ident + '_loss.png')
    plt.plot(train_loss, label='training')
    plt.plot(val_loss, '.', label='validation', alpha=0.5)
    plt.ylabel('mean loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(filename)


def plot_preds(pred, X_val, Y_val):
    for num in range(0, 10):
        plt.figure(figsize=[10, 6])
        plt.subplot(4, 4, 1)
        color_im = get_rgb(X_val[num, :, :, :])
        plt.imshow(color_im,
                   norm=LogNorm(vmin=0,
                                vmax=np.max(color_im) - 10 * np.std(color_im)))
        plt.title('Input blend (g, r, i)')
        plt.subplot(4, 4, 3)
        plt.imshow(pred[num, :, :, 0])
        plt.colorbar()
        plt.title('Network output')
        plt.subplot(4, 4, 2)
        plt.imshow(Y_val[num, :, :, 0])
        plt.colorbar()
        plt.title('Input central galaxy (i)')
        plt.subplot(4, 4, 4)
        plt.imshow(Y_val[num, :, :, 0] - pred[num, :, :, 0])
        plt.colorbar()
        plt.title('input - output')
        plt.show()


def main(Args):
    run_ident = 'lr_ ' + str(Args.learn_rate)
    path = os.path.join(os.path.dirname(os.getcwd()), "data")
    filename = os.path.join(path, 'training_data.npz')
    X_train, Y_train, X_val, Y_val = load_data(filename)
    model = utils.CNN_deblender(run_ident=run_ident,
                                learning_rate=Args.learn_rate)
    run_params = utils.Meas_args(epochs=Args.epochs,
                                 batch_size=Args.batch_size)
    train_loss, val_loss, pred = model.run_basic(X_train, Y_train,
                                                 run_params, X_val, Y_val)
    model.save()
    model.sess.close()
    plot_loss(train_loss, val_loss, run_ident)
    plot_preds(pred, X_val, Y_val)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--learn_rate', default=5e-4, type=float,
                        help="Learning rate of net [Default:5e-4]")
    parser.add_argument('--epochs', default=100, type=int,
                        help="Number of times net trained on entire training\
                        set [Default:100]")
    parser.add_argument('--batch_size', default=32, type=int,
                        help="Size of each mini batch [Default:32]")
    args = parser.parse_args()
    main(args)
