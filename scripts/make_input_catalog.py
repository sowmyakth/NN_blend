"""Script creates pair of galaxies in the format of catsim galaxy catalog.
The central galaxy and second galaxy is picked at random form the OneDegSq.fits
catsim catalog. Each pstamp is assumed to be centered at the central galaxy,
with a random shift in x and y direction. The secon dalaxy is loacated at a
random distance between 0.6 - 2.4 arcseconds from the central galaxy.
"""
import os
import numpy as np
from astropy.table import Table, vstack, Column
import copy
out_dir = '/global/cscratch1/sd/sowmyak/'


def get_galaxies(Args, catdir):
    """Randomly picks num_primary_gal * 2 galaxies, that satisfy certain
    selection conditions.

    Keyword arguments:
        Args      -- Class describing catalog.
        @Args.num -- Number of galaxy blends in catalog.
        catdir    -- path to directory with catsim catalog.

    Returns
        Combined catalog of central and secondary galaxies.
    """
    fname = os.path.join(catdir, 'data', 'wldeb_data',
                         'OneDegSq.fits')
    cat = Table.read(fname, format='fits')
    a = np.hypot(cat['a_d'], cat['a_b'])
    cond = (a <= 1.2) & (a > 0.2)
    q1, = np.where(cond & (cat['i_ab'] < 25.2))
    q2, = np.where(cond & (cat['i_ab'] < 25.2))
    select1 = q1[np.random.randint(0, len(q1), size=Args.num)]
    select2 = q2[np.random.randint(0, len(q2), size=Args.num)]
    return vstack([cat[select1], cat[select2]])


def check_center(Args, cat):
    """Checks that the central galaxy is at the center.
    If second galaxy is closer to center, swaps central
    and second galaxy catlog entries.
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


def get_second_centers(Args, cat):
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
    # mult_x = np.array([[1] * int(Args.num / 2) + [-1] * int(Args.num / 2)])[0]
    # mult_y = np.array([[1] * int(Args.num / 2) + [-1] * int(Args.num / 2)])[0]
    # np.random.shuffle(mult_x)
    # np.random.shuffle(mult_y)
    cat['dx'][Args.num:] += dx2  # x0 * mult_x
    cat['dy'][Args.num:] += dy2  # y0 * mult_y
    check_center(Args, cat)
    cat['ra'] += cat['dx'] * 0.2 / 3600.   # ra in degrees
    cat['dec'] += cat['dy'] * 0.2 / 3600.  # dec in degrees


def add_center_shift(Args, cat):
    """Shifts center of galaxies by a random value upto 5 pixels in
    both coordinates. The shift is same for central and secondary galaxies.

    Keyword arguments:
        Args      -- Class describing catalog.
        @Args.num -- Number of galaxy blends in catalog.
        cat       -- Combined catalog of central and secondary galaxies.
    """
    dx1 = np.random.uniform(-3, 3, size=Args.num)
    dy1 = np.random.uniform(-3, 3, size=Args.num)
    dx = np.append(dx1, dx1)
    dy = np.append(dy1, dy1)
    col = Column(dx, "dx")
    cat.add_column(col)
    col = Column(dy, "dy")
    cat.add_column(col)
    # cat['ra'] += dx * 0.2 / 3600.  # ra in degrees
    # cat['dec'] += dy * 0.2 / 3600.  # dec in degrees


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


def main(Args):
    print ("Creating input catalog")
    catdir = '/global/homes/s/sowmyak/blending'  # path to catsim catalog
    np.random.seed(Args.seed)
    catalog = get_galaxies(Args, catdir)  # make basic catalog of 2 gal blend
    get_central_centers(Args, catalog)  # Assign center of blend in the grid
    add_center_shift(Args, catalog)  # adds random shift to central galaxy
    get_second_centers(Args, catalog)  # assign center of second galaxy
    fname = os.path.join(out_dir, 'training_data',
                         'gal_pair_catalog.fits')  # blend catalog
    catalog.write(fname, format='fits', overwrite=True)
    fname = os.path.join(out_dir, 'training_data',
                         'central_gal_catalog.fits')  # central galaxy catalog
    catalog[:Args.num].write(fname, format='fits', overwrite=True)


def add_args(parser):
    parser.add_argument('--num', default=49000, type=int,
                        help="# of distinct galaxy pairs [Default:49000]")
    parser.add_argument('--seed', default=0, type=int,
                        help="Seed to randomly pick galaxies [Default:0]")
    parser.add_argument('--num_columns', default=700, type=int,
                        help="Number of columns in total field [Default:700]")
    parser.add_argument('--stamp_size', default=150, type=int,
                        help="Size of each stamp in pixels [Default:150]")
