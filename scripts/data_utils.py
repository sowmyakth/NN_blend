"""Script creates training and validation data for deblending two-galaxy
pairs"""
import os
from astropy.io import fits
import numpy as np


def get_stamps(full_image, rows, cols, in_size, out_size):
    num = rows * cols  # Total number of objects
    low = int(in_size / 2 - out_size / 2)
    high = int(in_size / 2 + out_size / 2)
    image_rows = full_image.reshape(rows, in_size, full_image.shape[1])
    stamps = [image_rows[j].T.reshape(cols, in_size, in_size) for j in range(len(image_rows))]
    stamps = np.array(stamps).reshape(num, in_size, in_size)
    stamps = stamps[:, low:high, low:high]
    return stamps.T


def load_images(filename, bands):
    """Returns individual postage stamp images of each blend in each band"""
    in_size, out_size = 150, 32
    num = 2048
    image = np.zeros([num, out_size, out_size, len(bands)])
    for i, band in enumerate(bands):
        print ("Getting pstamps for band", band)
        full_image = fits.open(filename.replace("band", band))[0].data
        print (full_image.shape)
        image.T[i] = get_stamps(full_image, rows=16, cols=128,
                                in_size=in_size, out_size=out_size)
    return image


def get_train_val_sets(X, Y, subtract_mean, split=0.1):
    """Separates the dataset into training and validation set with splitting
    ratio split.ratio
    Also subtracts the mean of the training image if input"""
    np.random.seed(0)
    num = X.shape[0]
    np.random.seed(0)
    validation = np.random.choice(num, int(num * split), replace=False)
    train = np.delete(range(num), validation)
    Y_val = Y[validation]  # Y[:, :, :, 0][validation]
    X_val = X[validation]
    Y_train = Y[train]  # Y[:, :, :, 0][train]
    X_train = X[train]
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        mean_image = np.mean(Y_train, axis=0)
        Y_train -= mean_image
        Y_val -= mean_image
    return X_train, Y_train, X_val, Y_val


def get_data(subtract_mean=False,
             normalize_stamps=True, save_files=False):
    bands = ['i', 'r', 'g']
    #path = os.path.join(os.path.dirname(os.getcwd()), "data")
    path = '/global/projecta/projectdirs/lsst/groups/WL/projects/wl-btf/two_gal_blend_data/training_data'
    filename = os.path.join(path, 'gal_pair_band_wldeb.fits')
    X = load_images(filename, bands)
    filename = os.path.join(path, 'central_gal_band_wldeb.fits')
    Y = load_images(filename, ['i'])
    if normalize_stamps:
        sum_image = X.sum(axis=3).sum(axis=1).sum(axis=1)
        X = (X.T / sum_image.T).T * 100
        # sum_image = Y.sum(axis=3).sum(axis=1).sum(axis=1)
        Y = (Y.T / sum_image.T).T * 100
    X_train, Y_train, X_val, Y_val = get_train_val_sets(X, Y, subtract_mean)
    if save_files is True:
        path = os.path.join(os.path.dirname(os.getcwd()), "data")
        filename = os.path.join(path, 'training_data')
        np.savez(filename, X_train=X_train,
                 Y_train=Y_train, X_val=X_val,
                 Y_val=Y_val)
        return
    return X_train, Y_train, X_val, Y_val
