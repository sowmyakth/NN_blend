#!/usr/bin/env python
"""Script creates training and validation data for deblending two-galaxy
pairs"""
import os
import pickle
from astropy.io import fits
from astropy.table import Table, Column
import numpy as np
import copy

# path to image fits files
out_dir = '/global/cscratch1/sd/sowmyak/training_data'
# in_path = "/global/projecta/projectdirs/lsst/groups/WL/projects/wl-btf/two_\
# gal_blend_data/training_data"


def get_stamps(full_image, Args):
    """Gets individual stamps of size out_size.

    Keyword Arguments
        full_image              -- Full field image
        Args                    -- Class describing input image.
        @Args.num               -- Number of galaxy blends in catalog.
        @Args.num_columns       -- Number of columns in total field.
        @Args.in_size           -- Size of each stamp in pixels.
        @Args.out_size          -- Desired size of postage stamps in pixels.
    Returns
        array of individual postage stamps
    """
    print ("getting individual stamps")
    nrows = int(np.ceil(Args.num / Args.num_columns))  # Total number of rows
    out_size = Args.out_size
    low = int(Args.in_size / 2 - out_size / 2)
    high = int(Args.in_size / 2 + out_size / 2)
    nStamp = (nrows, Args.num_columns)
    stampSize = Args.in_size
    s2 = np.hstack(np.split(full_image,nStamp[0])).T.reshape(nStamp[0]*nStamp[1],
                   stampSize, stampSize)
    stamps = s2[:, low:high, low:high]
    return stamps


def load_images(filename, bands, Args):
    """Returns individual postage stamp images of each blend in each band

    Keyword Arguments
        filename  -- Name of file to load image
        bands     -- differnt image filters
        Args      -- Class describing input image.
    Returns
        array of individual postage stamps in all bands
    """
    image = np.zeros([Args.num, Args.out_size,
                      Args.out_size, len(bands)])
    for i, band in enumerate(bands):
        print ("Getting pstamps for band", band)
        full_image = fits.open(filename.replace("band", band))[0].data
        image[:, :, :, i] = get_stamps(full_image, Args)
    return image


def load_input_images(bands, Args):
    name = 'gal_pair_band_wldeb_noise.fits'
    filename = os.path.join(out_dir, Args.model, name)
    blend_image = load_images(filename, bands, Args)
    X = {'blend_image': blend_image}
    if Args.model == "lilac":
        name = 'gal_pair2_band_wldeb_noise.fits'
        filename = os.path.join(out_dir, Args.model, name)
        blend_image2 = load_images(filename, bands, Args)
        X['blend_image2'] = blend_image2
    else:
        name = 'loc_map1_band_wldeb.fits'
        filename = os.path.join(out_dir, Args.model, name)
        loc1 = load_images(filename, ['i'], Args)
        # load second galaxy images
        name = 'loc_map2_band_wldeb.fits'
        filename = os.path.join(out_dir, Args.model, name)
        loc2 = load_images(filename, ['i'], Args)
        X['loc1'] = loc1
        X['loc2'] = loc2
    return X


def load_isolated_images(Args):
    """Returns dict of individual isolated galaxy image for each blend
    Keyword Arguments
        Args         -- Class describing input image.
        @Args.model  -- CNN model for which the data is designed for.
    """
    # load first galaxy images
    name = 'first_gal_band_wldeb_noise.fits'
    filename = os.path.join(out_dir, Args.model, name)
    Y1 = load_images(filename, ['i'], Args)
    # load second galaxy images
    name = 'second_gal_band_wldeb_noise.fits'
    filename = os.path.join(out_dir, Args.model, name)
    Y2 = load_images(filename, ['i'], Args)
    Y = {'Y1': Y1,
         'Y2': Y2}
    return Y


def add_blend_param(cat, cent, other, blend_cat):
    """Computes distance between pair, magnitude, color, flux amd size of neighbor.
    Also saves column indicating if galaxy pair will be used in validation  and
    column to save id number.

    Keyword Arguments

        cat       --  Input galaxy pair catalog.
        cent      --  Indices of central galaxy.
        other     --  Indices of other galaxy.
        blend_cat --  Catalog to save blend parametrs to.
    """
    dist = np.hypot(cat[cent]['dx'] - cat[other]['dx'],
                    cat[cent]['dy'] - cat[other]['dy'])
    col = Column(dist, "distance_neighbor")
    blend_cat.add_column(col)
    col = Column(cat['ab_mag'][other], "mag_neighbor")
    blend_cat.add_column(col)
    col = Column(cat['ri_color'][other], "color_neighbor")
    blend_cat.add_column(col)
    col = Column(cat['flux'][other], "flux_neighbor")
    blend_cat.add_column(col)
    col = Column(cat['sigma_m'][other], "sigma_neighbor")
    blend_cat.add_column(col)


def get_blend_catalog(Args):
    """Creates catalog that saves central galaxy true parametrs + selected
    parametrs of other galaxy

    Keyword Arguments
        Args       -- Class describing input image.
        @Args.num  -- Number of galaxy blends in catalog.
    Returns
        blend_cat --  Catalog to save blend parametrs to.
    """
    filename = os.path.join(out_dir,
                            Args.model, 'gal_pair_i_wldeb.fits')
    cat = Table.read(filename, hdu=1)
    assert len(cat) % 2 == 0, "Catalog must contain only 2 galaxy blends"
    cent = np.linspace(0, Args.num, Args.num, dtype=int, endpoint=False)
    other = np.linspace(Args.num, Args.num * 2, Args.num,
                        dtype=int, endpoint=False)
    assert len(cent) == len(other), 'Each central galaxy must have a blend'
    blend_cat = cat[cent]
    add_blend_param(cat, cent, other, blend_cat)
    return blend_cat


def normalize_other_inputs(X, Args):
    """Normalizes the loc map or second blend image for the input model
    Keyword Arguments
        X            --  dict containg images input to net
        Args         -- Class describing input image.
        @Args.model  -- CNN model for which the data is designed for.
    Returns
        normalized location map image()s)
    """
    other_keys = list(X.keys())
    other_keys.remove("blend_image")
    for key in other_keys:
        X[key] = (X[key] - np.mean(X[key])) / np.std(X[key])
    if Args.model == "orchid":
        loc_im = np.zeros_like(X[other_keys[0]])
        for i, key in enumerate(other_keys):
            im = X.pop(key)
            maximum = np.min((im.max(axis=2).max(axis=1)))
            im[im < maximum / 1.5] = 0
            im[im >= maximum / 1.5] = i + 1
            loc_im += im
        X['loc_im'] = loc_im
    return X


def normalize_images(data, blend_cat, Args):
    """Galaxy images are normalized such that the sum of flux of blended
    images in all bands is 100.

    Keyword Arguments
        X -- array of blended galaxy postage stamps images.
        Y -- array of isolated central galaxy postage stamp images.
        blend_cat  -- Catalog to save blend parametrs to.
    Returns
        X_norm -- normalized blended galaxy postage stamps images.
        Y_norm -- normalized isolated galaxy postage stamps images.
        sum_image -- sum of images in 3 bands; normalization value.
    """
    im = data['X_train']['blend_image']
    std = np.std(im)
    mean = np.mean(im)
    data['X_train']['blend_image'] = (im - mean) / std
    data['X_val']['blend_image'] = (data['X_val']['blend_image'] - mean) / std
    data['X_train'] = normalize_other_inputs(data['X_train'], Args)
    data['X_val'] = normalize_other_inputs(data['X_val'], Args)
    for key in data['Y_train'].keys():
        data['Y_train'][key] = (data['Y_train'][key] - mean) / std
        data['Y_val'][key] = (data['Y_val'][key] - mean) / std
    blend_cat['std'] = std
    blend_cat['mean'] = mean
    return data


def add_nn_id_blend_cat(blend_cat, validation, train):
    """Saves index of galaxy that will be used in the CNN deblender,
    separated into training and validation sets. If pair entry is to be used
    for validation then catalog parameter 'is_validation' is set to 1. The
    order in which galaxies will appear in the CNN training and validation set
    is also saved.

    Keyword Arguments
        blend_cat  -- Catalog to save blend parametrs to.
        sum_images -- sum of images in 3 bands; normalization value.
        validation -- index of galaxies to be used in validation set.
        train      -- index of galaxies to be used in training set.
    """
    col = Column(np.zeros(len(blend_cat)), "is_validation", dtype=int)
    blend_cat.add_column(col)
    col = Column(np.zeros(len(blend_cat)), "nn_id", dtype=int)
    blend_cat.add_column(col)
    col = Column(np.zeros(len(blend_cat)), "norm", dtype=float)
    blend_cat.add_column(col)
    col = Column(np.zeros(len(blend_cat)), "mean", dtype=float)
    blend_cat.add_column(col)
    blend_cat['is_validation'][validation] = 1
    blend_cat['nn_id'][validation] = range(len(validation))
    blend_cat['nn_id'][train] = range(len(train))


def concat_rotated_images(X):
    """Increase training set by performing data augmentation"""
    X_aug = copy.deepcopy(X)
    for key in X_aug.keys():
        X_aug[key] = np.concatenate([X[key][:, :, :, :],
                                     X[key][:, :, ::-1, :],
                                     X[key][:, ::-1, :, :],
                                     X[key][:, ::-1, ::-1, :]])
    return X_aug


def get_train_val_sets(X, Y, blend_cat,
                       Args, split=0.1):
    """Separates the dataset into training and validation set with splitting
    ratio. Also subtracts the mean of the training image if input

    Keyword Arguments
        X          -- array of blended galaxy postage stamps images.
        Y          -- array of isolated central galaxy postage stamp images.
        blend_cat  -- Catalog to save blend parametrs to.
        sum_images -- sum of images in 3 bands; normalization value.
    Returns
        Training and validation input and output
    """
    num = X['blend_image'].shape[0]
    validation = np.random.choice(num, int(num * split), replace=False)
    train = np.delete(range(num), validation)
    X_val, X_train = {}, {}
    for key in X.keys():
        X_train[key] = X[key][train]
        X_val[key] = X[key][validation]
    Y_val, Y_train = {}, {}
    for key in Y.keys():
        Y_train[key] = Y[key][train]
        Y_val[key] = Y[key][validation]
    data = {'X_train': X_train,
            'Y_train': Y_train,
            'X_val': X_val,
            'Y_val': Y_val}
    add_nn_id_blend_cat(blend_cat, validation, train)
    return data


def augment_images(data):
    data['X_train'] = concat_rotated_images(data['X_train'])
    data['Y_train'] = concat_rotated_images(data['Y_train'])
    return data


def main(Args):
    np.random.seed(0)
    bands = ['i', 'r', 'g']
    # load CNN input data
    X = load_input_images(bands, Args)
    blend_cat = get_blend_catalog(Args)
    # load CNN output data
    Y = load_isolated_images(Args)
    # Divide data into training and validation sets.
    data = get_train_val_sets(X, Y,
                              blend_cat, Args)
    norm_data = normalize_images(data, blend_cat, Args)
    aug_data = augment_images(norm_data)
    filename = os.path.join(out_dir,
                            Args.model, 'stamps.pickle')
    with open(filename, 'wb') as handle:
        pickle.dump(aug_data, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    filename = os.path.join(out_dir,
                            Args.model + '_blend_param.tab')
    blend_cat.write(filename, format='ascii', overwrite=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', default=49000, type=int,
                        help="# of distinct galaxy pairs [Default:49000")
    parser.add_argument('--num_columns', default=700, type=int,
                        help="Number of columns in total field [Default:700]")
    parser.add_argument('--in_size', default=80, type=int,
                        help="Size of input stamps in pixels [Default:80]")
    parser.add_argument('--out_size', default=32, type=int,
                        help="Size of stamps desired, in pixels [Default:32]")
    parser.add_argument('--model', default='lilac',
                        help="Model for which catalog is made \
                        [Default:lavender]")
    args = parser.parse_args()
    main(args)
