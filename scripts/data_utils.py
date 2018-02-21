"""Script creates training and validation data for deblending two-galaxy pairs"""
from astropy.io import fits
import numpy as np


def load_images(filename, bands):
    """Returns individual postage stamp images of each blend in each band"""
    size = 240
    out_size = 32
    num, rows, cols = 200, 25, 8
    low, high = int(size / 2 - out_size / 2), int(size / 2 + out_size / 2)
    image = np.zeros([num, out_size, out_size, len(bands)])
    for i, band in enumerate(bands):
        full_image = fits.open(filename.replace("band", band))[0].data
        image_rows = full_image.reshape(rows, size, full_image.shape[1])
        stamps = [image_rows[j].T.reshape(cols, size, size) for j in range(len(image_rows))]
        stamps = np.array(stamps).reshape(num, size, size)
        stamps = stamps[:, low:high, low:high]
        image.T[i] = stamps.T
    return image


def get_train_val_sets(X, Y, split=0.1,
                       subtract_mean):
    """Separates the dataset into training and validation set with splitting
    ratio split.ratio
    Also subtracts the mean of the training image if input"""
    np.random.seed(0)
    num = X.shape[0]
    np.random.seed(0)
    validation = np.random.choice(num, int(num * split), replace=False)
    train = range(num)
    [train.remove(i) for i in validation]
    Y_val = Y[:, :, :, 0][validation]
    X_val = X[validation]
    Y_train = Y[:, :, :, 0][train]
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
             normalize_stamps=True):
    bands = ['i', 'r']
    path = '/global/homes/s/sowmyak/NN_blend/data/'
    filename = path + 'gal_pair_band_wldeb.fits'
    X = load_images(filename, bands)
    filename = path + 'central_gal_band_wldeb.fits'
    Y = load_images(filename, ['i'])
    if normalize_stamps:
        sum_image = X.sum(axis=3).sum(axis=1).sum(axis=1)
        X = (X.T / sum_image.T).T
        sum_image = Y.sum(axis=3).sum(axis=1).sum(axis=1)
        Y = (Y.T / sum_image.T).T
    X_train, Y_train, X_val, Y_val = get_train_val_sets(X, Y, subtract_mean)
    return X_train, Y_train, X_val, Y_val
