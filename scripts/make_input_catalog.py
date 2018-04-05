"""Script creates pair of galaxies in the format of catsim galaxy catalog.
The central galaxy and second galaxy is picked at random form the OneDegSq.fits
catsim catalog. Each pstamp is assumed to be centered at the central galaxy,
with a random shift in x and y direction. The secon dalaxy is loacated at a
random distance between 0.6 - 2.4 arcseconds from the central galaxy.
"""
from __future__ import division
import os
import numpy as np
from astropy.table import Table, vstack, Column
import copy
import subprocess
out_dir = '/global/cscratch1/sd/sowmyak/training_data'


def get_weights_for_uniform_sample(arr, nbins=40):
    """Returns weights to get uniform sample of array arr
    Keyword arguments
        arr -- array to compute weights
        nbins -- Number of bins
    Returns
        weights
    """
    num = np.digitize(arr, np.linspace(np.min(arr), np.max(arr), nbins))
    bin_frequency = np.bincount(num)
    bin_prob = bin_frequency / len(arr)
    individual_prob = bin_prob[num]
    req_prob = 1. / individual_prob
    weights = req_prob / sum(req_prob)
    return weights


def get_galaxies(Args, catdir):
    """Randomly picks num_primary_gal * 2 galaxies, that satisfy certain
    selection conditions.

    Keyword arguments:
        Args                 -- Class describing catalog.
        @Args.num            -- Number of galaxy blends in catalog.
        Args.no_undetectable -- if true, no undetectable galaxies will be\
                                added to the dataset
        catdir    -- path to directory with catsim catalog.

    Returns
        Combined catalog of central and secondary galaxies.
    """
    fname = os.path.join(catdir, 'data', 'wldeb_data',
                         'OneDegSq.fits')
    cat = Table.read(fname, format='fits')
    a = np.hypot(cat['a_d'], cat['a_b'])
    cond = (a <= 1.2) & (a > 0.2)
    q, = np.where(cond & (cat['i_ab'] <= 25.2))
    if Args.no_undetectable:
        cat1 = cat[np.random.choice(q, size=Args.num)]  # , p=weights)
        cat2 = cat[np.random.choice(q, size=Args.num)]  # , p=weights)
    else:
        # Make data set contain 1% undetectable galaxies
        undet_num = int(0.01 * Args.num)
        q2, = np.where(cond & (cat['i_ab'] < 27) & (cat['i_ab'] > 25.2))
        cat1 = vstack([cat[np.random.choice(q, size=Args.num - undet_num)],
                       cat[np.random.choice(q2, size=undet_num)]])
        cat2 = vstack([cat[np.random.choice(q, size=Args.num - undet_num)],
                       cat[np.random.choice(q2, size=undet_num)]])
    # Get uniform distribution in mag
    # weights = get_weights_for_uniform_sample(cat['i_ab'][q])
    assert (len(cat1) == Args.num), "Incorrect # of galaxies selected"
    assert (len(cat2) == Args.num), "Incorrect # of galaxies selected"
    return vstack([cat1, cat2])


def make_first_center(Args, cat):
    """Checks that the first galaxy is at the center.
    If second galaxy is closer to center, then swaps first
    and second galaxy catlog entries.
    Keyword Arguments
        Args      -- Class describing catalog.
        @Args.num -- Number of galaxy blends in catalog.
        cat       -- Combined catalog of central and secondary galaxies.
    """
    col = Column(np.zeros(Args.num * 2), "is_swapped", dtype=int)
    cat.add_column(col)
    ds = np.hypot(cat['dx'], cat['dy'])
    q, = np.where(ds[:Args.num] >= ds[Args.num:])
    cat['is_swapped'][Args.num:][q] = 1
    cat['is_swapped'][:Args.num][q] = 1
    cat1_copy = copy.deepcopy(cat[:Args.num][q])
    cat2_copy = copy.deepcopy(cat[Args.num:][q])
    cat[:Args.num][q] = cat2_copy
    cat[Args.num:][q] = cat1_copy


def get_second_centers(Args, cat, check_center=True):
    """ Assigns a random x and y cordinate distance of the second galaxy from
    the central galaxy.

    Keyword arguments:
        Args      -- Class describing catalog.
        @Args.num -- Number of galaxy blends in catalog.
        cat       -- Combined catalog of central and secondary galaxies.
    """
    dr = np.random.uniform(3, 10, size=Args.num)
    theta = np.random.uniform(0, 360, size=Args.num) * np.pi / 180.
    dx2 = dr * np.cos(theta)
    dy2 = dr * np.sin(theta)
    cat['dx'][Args.num:] += dx2  # x0 * mult_x
    cat['dy'][Args.num:] += dy2  # y0 * mult_y
    if check_center:
        make_first_center(Args, cat)
    cat['ra'] += cat['dx'] * 0.2 / 3600.   # ra in degrees
    cat['dec'] += cat['dy'] * 0.2 / 3600.  # dec in degrees


def add_center_shift(Args, cat, maxshift=3):
    """Shifts center of galaxies by a random value upto 5 pixels in
    both coordinates. The shift is same for central and secondary galaxies.

    Keyword arguments:
        Args      -- Class describing catalog.
        @Args.num -- Number of galaxy blends in catalog.
        cat       -- Combined catalog of central and secondary galaxies.
    """
    dx1 = np.random.uniform(-maxshift, maxshift, size=Args.num)
    dy1 = np.random.uniform(-maxshift, maxshift, size=Args.num)
    dx = np.append(dx1, dx1)
    dy = np.append(dy1, dy1)
    col = Column(dx, "dx")
    cat.add_column(col)
    col = Column(dy, "dy")
    cat.add_column(col)


def get_center_of_field(Args):
    """Computes x and y coordinates of the center of the field
    Keyword arguments:
        Args              -- Class describing catalog.
        @Args.num         -- Number of galaxy blends in catalog.
        @Args.num_columns -- Number of columns in total field.
        @args.stamp_size  -- Size of each stamp in pixels.
    Returns
        x and y coordu=inates of field center
    """
    nrows = int(np.ceil(Args.num / Args.num_columns))
    x_cent = (Args.num_columns * Args.stamp_size - 1) / 2.
    y_cent = (nrows * Args.stamp_size - 1) / 2.
    return x_cent, y_cent


def get_central_centers(Args, cat):
    """Gets x and y coordinates of central galaxy.
    The centers of second galaxy are also assigned to the same value as
    their neighboring central galaxy.

    Keyword arguments:
        Args              -- Class describing catalog.
        @Args.num         -- Number of galaxy blends in catalog.
        @Args.num_columns -- Number of columns in total field.
        @args.stamp_size  -- Size of each stamp in pixels.
    """
    num = Args.num
    ncols = Args.num_columns
    nrows = int(np.ceil(num / ncols))
    c = 0.2 / 3600.  # conversion from pixels to degrees
    x_cent, y_cent = get_center_of_field(Args)
    xs = list(range(ncols)) * nrows
    xs = (np.array(xs)[list(range(num))]) * Args.stamp_size
    ys = np.array(list(range(num)), dtype=int)[list(range(num))] / ncols
    ys = ys.astype(int)
    ys *= Args.stamp_size
    cat['dec'] = (np.append(ys, ys) + Args.stamp_size / 2. - y_cent) * c
    cat['ra'] = (np.append(xs, xs) + Args.stamp_size / 2. - x_cent) * c


def switch_centers(catalog, catalog2, num):
    """Switch centers to get second galaxy at center of stamp"""
    catalog2['dx'][num:] = -1 * catalog['dx'][:num]
    catalog2['dy'][num:] = -1 * catalog['dy'][:num]
    catalog2['dx'][:num] = -1 * catalog['dx'][num:]
    catalog2['dy'][:num] = -1 * catalog['dy'][num:]
    # get ra dec of stamp center
    catalog2['ra'] = catalog['ra'] - catalog['dx'] * 0.2 / 3600.
    catalog2['dec'] = catalog['dec'] - catalog['dy'] * 0.2 / 3600.
    # add shift in ra dec of each galaxy
    catalog2['ra'] += catalog2['dx'] * 0.2 / 3600.
    catalog2['dec'] += catalog2['dy'] * 0.2 / 3600.
    # catalog2['ra'][num:] = -1 *
    # catalog2['dec'][num:] = -1 * catalog['dec'][:num]
    # catalog2['ra'][:num] = -1 * catalog['ra'][num:]
    # catalog2['dec'][:num] = -1 * catalog['dec'][num:]


def make_cats_for_lilac(Args, catalog):
    """Assigns center for central and second galaxy required for
    lilac training and testing
     Keyword arguments:
        Args        -- Class describing catalog.
        @Args.model -- CNN model for which the data is designed for.
        catalog     -- Combined catalog of central and secondary galaxies.
    """
    add_center_shift(Args, catalog)  # adds random shift to central galaxy
    get_second_centers(Args, catalog)  # assign center of second galaxy
    catalog2 = copy.deepcopy(catalog)
    switch_centers(catalog, catalog2, Args.num)
    fname = os.path.join(out_dir, Args.model, 'gal_pair_catalog.fits')
    catalog.write(fname, format='fits', overwrite=True)  # blend catalog1
    fname = os.path.join(out_dir, Args.model, 'gal_pair2_catalog.fits')
    catalog2.write(fname, format='fits', overwrite=True)  # blend catalog2
    # first galaxy catalog
    fname = os.path.join(out_dir, Args.model, 'first_gal_catalog.fits')
    catalog[:Args.num].write(fname, format='fits', overwrite=True)
    # second galaxy catalog
    fname = os.path.join(out_dir, Args.model, 'second_gal_catalog.fits')
    catalog2[Args.num:].write(fname, format='fits', overwrite=True)
    return


def make_loc_map(catalog, Args):
    """Create a catalog with a star of mag 22 at the location of the galaxies
    in the input blend catalog.
    Keyword arguments:
        catalog     -- Combined catalog of central and secondary galaxies.
        Args        -- Class describing catalog.
        @Args.model -- CNN model for which the data is designed for.
     """
    dtype = [('startileid', np.int32),
             ('ra', np.float32),
             ('dec', np.float32),
             ('i_ab', np.float32),
             ('r_ab', np.float32),
             ('redshift', np.float32),
             ('fluxnorm_star', np.float32)]
    data = np.zeros(len(catalog), dtype=dtype)
    cat2 = Table(data, copy=False)
    cat2['startileid'] = catalog['galtileid']
    cat2['ra'] = catalog['ra']
    cat2['dec'] = catalog['dec']
    cat2['i_ab'] = 22
    cat2['r_ab'] = 23
    cat2['redshift'] = 0.3
    cat2['fluxnorm_star'] = 6e-9
    fname = os.path.join(out_dir, Args.model, 'loc_map1_catalog.fits')
    cat2[:Args.num].write(fname, format='fits', overwrite=True)
    fname = os.path.join(out_dir, Args.model, 'loc_map2_catalog.fits')
    cat2[Args.num:].write(fname, format='fits', overwrite=True)


def main(Args):
    if not os.path.isdir(os.path.join(out_dir, Args.model)):
        subprocess.call(["mkdir", os.path.join(out_dir, Args.model)])
    print ("Creating input catalog")
    catdir = '/global/homes/s/sowmyak/blending'  # path to catsim catalog
    np.random.seed(Args.seed)
    catalog = get_galaxies(Args, catdir)  # make basic catalog of 2 gal blend
    get_central_centers(Args, catalog)  # Assign center of blend in the grid
    if Args.model == 'lilac':
        make_cats_for_lilac(Args, catalog)
        return
    add_center_shift(Args, catalog, maxshift=10)
    get_second_centers(Args, catalog, check_center=False)
    make_loc_map(catalog, Args)
    fname = os.path.join(out_dir, Args.model, 'gal_pair_catalog.fits')
    catalog.write(fname, format='fits', overwrite=True)  # blend catalog
    # first galaxy catalog
    fname = os.path.join(out_dir, Args.model, 'first_gal_catalog.fits')
    catalog[:Args.num].write(fname, format='fits', overwrite=True)
    # second galaxy catalog
    fname = os.path.join(out_dir, Args.model, 'second_gal_catalog.fits')
    catalog[Args.num:].write(fname, format='fits', overwrite=True)


def add_args(parser):
    parser.add_argument('--num', default=49000, type=int,
                        help="# of distinct galaxy pairs [Default:49000]")
    parser.add_argument('--no_undetectable', action='store_true',
                        help="Undetectable galaxies not added[Default:False]")
    parser.add_argument('--seed', default=0, type=int,
                        help="Seed to randomly pick galaxies [Default:0]")
    parser.add_argument('--num_columns', default=700, type=int,
                        help="Number of columns in total field [Default:700]")
    parser.add_argument('--stamp_size', default=80, type=int,
                        help="Size of each stamp in pixels [Default:80]")
    parser.add_argument('--model', default='lilac',
                        help="Model for which catalog is made \
                        [Default:lilac]")
