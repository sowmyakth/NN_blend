"""Script creates pair of galaxies in the format of catsim galaxy catalog.
The central galaxy and second galaxy is picked at random form the OneDegSq.fits
catsim catalog.

Default settings
Each pstamp is assumed to be centered at the central galaxy. Each pstamo is
is 48 arcseconds or 240 pixels.
Total field is made of --num of pstamps: 4 in a row

Ring test not included.
"""
import os
import numpy as np
from astropy.table import Table, vstack


def get_galaxies(Args, catdir):
    """Randomly picks num_primary_gal * 2 galaxies, with certain conditions
    """
    fname = os.path.join(catdir, 'data', 'wldeb_data',
                         'OneDegSq.fits')
    cat = Table.read(fname, format='fits')
    a = np.hypot(cat['a_d'], cat['a_b'])
    cond = (a <= 1.2) & (a > 0.2)
    q1, = np.where(cond & (cat['i_ab'] < 24))
    q2, = np.where(cond)
    select1 = q1[np.random.randint(0, len(q1), size=Args.num)]
    select2 = q2[np.random.randint(0, len(q2), size=Args.num)]
    return vstack([cat[select1], cat[select2]])


def get_second_centers(Args, cat):
    """Randomly select centers between 0.6 to  2.4 arcseconds"""
    x0 = np.random.uniform(0.6, 2.4, size=Args.num)
    y0 = np.random.uniform(0.6, 2.4, size=Args.num)
    mult_x = np.array([[1] * int(Args.num / 2) + [-1] * int(Args.num / 2)])[0]
    mult_y = np.array([[1] * int(Args.num / 2) + [-1] * int(Args.num / 2)])[0]
    np.random.shuffle(mult_x)
    np.random.shuffle(mult_y)
    c = 1 / 3600.
    cat['ra'][Args.num:] += x0 * mult_x * c
    cat['dec'][Args.num:] += y0 * mult_y * c


def get_center_of_field(Args):
    """Returns cenetr pixel value"""
    nrows = int(np.ceil(Args.num / Args.num_columns))
    x_cent = (Args.num_columns * Args.stamp_size - 1) / 2.
    y_cent = (nrows * Args.stamp_size - 1) / 2.
    return x_cent, y_cent


def get_central_centers(Args, cat):
    """Gets x and y coordinates of central galaxy.
    The centers of second galaxy are also assigne dto the same value as
    their neighboring central galaxy.
    """
    num = Args.num
    ncols = Args.num_columns
    nrows = int(np.ceil(num / ncols))
    c = 0.2 / 3600.  # conversion from pixels to arcseconds
    x_cent, y_cent = get_center_of_field(Args)
    xs = list(range(ncols)) * nrows
    xs = (np.array(xs)[list(range(num))]) * Args.stamp_size
    ys = np.array(list(range(num)), dtype=int)[list(range(num))] / ncols
    ys = ys.astype(int)
    ys *= Args.stamp_size
    cat['dec'] = (np.append(ys, ys) + Args.stamp_size / 2. - y_cent) * c
    cat['ra'] = (np.append(xs, xs) + Args.stamp_size / 2. - x_cent) * c


def main(Args):
    print ("Creating input catalog")
    catdir = '/global/homes/s/sowmyak/blending'
    np.random.seed(Args.seed)
    catalog = get_galaxies(Args, catdir)
    get_central_centers(Args, catalog)
    get_second_centers(Args, catalog)
    parentdir = os.path.abspath("..")
    fname = os.path.join(parentdir, 'data',
                         'gal_pair_catalog.fits')
    catalog.write(fname, format='fits', overwrite=True)
    fname = os.path.join(parentdir, 'data',
                         'central_gal_catalog.fits')
    catalog[:Args.num].write(fname, format='fits', overwrite=True)


def add_args(parser):
    # from argparse import ArgumentParser
    # parser = ArgumentParser()
    parser.add_argument('--num', default=2048, type=int,
                        help="# of distinct galaxy pairs [Default:16]")
    parser.add_argument('--seed', default=0, type=int,
                        help="Seed to randomly pick galaxies [Default:0]")
    parser.add_argument('--num_columns', default=128, type=int,
                        help="Number of columns in total field [Default:8]")
    parser.add_argument('--stamp_size', default=150, type=int,
                        help="Size of each stamp in pixels [Default:240]")
#    main(args)


# if __name__ == "__main__":
#    add_args()
