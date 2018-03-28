#!/usr/bin/env python
"""Script creates training and validation data for deblending two-galaxy
pairs"""
import os
from astropy.io import fits
from astropy.table import Table, Column
import numpy as np

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
    # image_rows = full_image.reshape(nrows, Args.in_size, full_image.shape[1])
    # image_rows = np.vstack(np.hsplit(full_image, Args.num_columns))
    # print ("Number of rows", len(image_rows))
    # stamps = [image_rows[j].T.reshape(Args.num_columns, Args.in_size, Args.in_size) for j in range(len(image_rows))]
    # stamps = np.array(stamps).reshape(Args.num, Args.in_size, Args.in_size)
    # stamps = image_rows.reshape(Args.num, Args.in_size, Args.in_size)
    #nStamp = (70, 700)
    #stampSize = 80
    nStamp = (nrows, Args.num_columns)
    stampSize = Args.in_size
    print(nStamp)
    s2 = np.hstack(np.split(full_image,nStamp[0])).T.reshape(nStamp[0]*nStamp[1], stampSize, stampSize)
    # stamps = stamps[:, low:high, low:high]
    stamps = s2[:, low:high, low:high]
    print ("Number of rows", stamps.shape)
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
        Args              -- Class describing input image.
        @Args.num         -- Number of galaxy blends in catalog.
    Returns
        blend_cat --  Catalog to save blend parametrs to.
    """
    filename = os.path.join(out_dir, 'gal_pair_i_wldeb.fits')
    cat = Table.read(filename, hdu=1)
    assert len(cat) % 2 == 0, "Catalog must contain only 2 galaxy blends"
    cent = np.linspace(0, Args.num, Args.num, dtype=int, endpoint=False)
    other = np.linspace(Args.num, Args.num * 2, Args.num,
                        dtype=int, endpoint=False)
    assert len(cent) == len(other), 'Each central galaxy must have a blend'
    blend_cat = cat[cent]
    add_blend_param(cat, cent, other, blend_cat)
    return blend_cat


def normalize_images(X, Y):
    """Galaxy images are normalized such that the sum of flux of blended
    images in all bands is 100.

    Keyword Arguments
        X -- array of blended galaxy postage stamps images.
        Y -- array of isolated central galaxy postage stamp images.
    Returns
        X_norm -- normalized blended galaxy postage stamps images.
        Y_norm -- normalized isolated galaxy postage stamps images.
        sum_image -- sum of images in 3 bands; normalization value.
    """
    sum_image = X.sum(axis=3).sum(axis=1).sum(axis=1)
    X_norm = (X.T / sum_image).T * 100
    # sum_image = Y.sum(axis=3).sum(axis=1).sum(axis=1)
    Y_norm = (Y.T / sum_image).T * 100
    np.testing.assert_almost_equal(X_norm.sum(axis=3).sum(axis=1).sum(axis=1),
                                   100, err_msg="Incorrectly normalized")
    assert np.all(Y_norm.sum(axis=3).sum(axis=1).sum(axis=1) <= 100),\
        "Incorrectly normalized"
    return X_norm, Y_norm, sum_image


def add_nn_id_blend_cat(blend_cat, sum_images,
                        validation, train):
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
    blend_cat['is_validation'][validation] = 1
    blend_cat['nn_id'][validation] = range(len(validation))
    blend_cat['nn_id'][train] = range(len(train))
    blend_cat['norm'] = sum_images


def subtract_mean(X_train, Y_train, X_val, Y_val):
    """Subtracts mean image"""
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    mean_image = np.mean(Y_train, axis=0)
    Y_train -= mean_image
    Y_val -= mean_image
    return X_train, Y_train, X_val, Y_val


def get_train_val_sets(X, Y, blend_cat,
                       subtract_mean, split=0.1):
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
    X_norm, Y_norm, sum_images = normalize_images(X, Y)
    num = X.shape[0]
    validation = np.random.choice(num, int(num * split), replace=False)
    train = np.delete(range(num), validation)
    Y_val = Y_norm[validation]
    X_val = X_norm[validation]
    Y_train = Y_norm[train]
    X_train = X_norm[train]
    add_nn_id_blend_cat(blend_cat, sum_images, validation, train)
    if subtract_mean:
        X_train, Y_train, X_val, Y_val = subtract_mean(X_train, Y_train,
                                                       X_val, Y_val)
    return X_train, Y_train, X_val, Y_val


def main(Args):
    np.random.seed(0)
    bands = ['i', 'r', 'g']
    # load blended galaxy images
    name = args.model + '_gal_pair_band_wldeb_noise.fits'
    filename = os.path.join(out_dir,
                            name)
    X = load_images(filename, bands, Args)
    blend_cat = get_blend_catalog(Args)
    # load first galaxy images
    name = args.model + '_first_gal_band_wldeb_noise.fits'
    filename = os.path.join(out_dir,
                            name)
    Y = load_images(filename, ['i'], Args)
    X_train, Y_train, X_val, Y_val = get_train_val_sets(X, Y,
                                                        blend_cat,
                                                        subtract_mean=False)
    filename = os.path.join(out_dir,
                            'stamps')
    np.savez(filename, X_train=X_train,
             Y_train=Y_train, X_val=X_val,
             Y_val=Y_val)
    filename = os.path.join(out_dir,
                            'blend_param.tab')
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
