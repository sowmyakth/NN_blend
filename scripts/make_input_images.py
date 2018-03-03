"""Creates input catalog which is then run via WLDeblending package
to produce galaxy images.
Setting input option make_input_catalog to False implies the base catlog will
not be created and the debelending package will run for the input options
and input catalogs as the already exosting gal_pair_catalog.fits and
central_gal_catalog.fits.
"""
import os
import subprocess
import argparse
import make_input_catalog
import numpy as np
import galsim
import sys
wldeb_path = "/global/homes/s/sowmyak/blending_tutorial/Blending_tutorial/WeakLensingDeblending/"
sys.path.insert(0, wldeb_path)
import descwl


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--make_input_catalog', default="True",
                        help="Run make_input_catalog.main() [Default:True]")
    catalog_group = parser.add_argument_group('Catlog options',
                                              'Input catalog options')
    make_input_catalog.add_args(catalog_group)
    wldeb_group = parser.add_argument_group('WLDEB package options',
                                            'Input options to WLdeb package')
    second_args(wldeb_group)
    args = parser.parse_args()
    if args.make_input_catalog is "True":
        make_input_catalog.main(args)
    # import ipdb;ipdb.set_trace()
    num = args.num
    ncols = args.num_columns
    nrows = int(np.ceil(num / ncols))
    args.image_height = nrows * args.stamp_size
    args.image_width = ncols * args.stamp_size
    run_wl_deb(args, 'gal_pair')
    run_wl_deb(args, 'central_gal')
    # add_noise(args, 'gal_pair')
    # add_noise(args, 'central_gal')


def second_args(parser):
    """Arguments to be passed to the WLdeblender"""
    parser.add_argument('--cosmic-shear-g1', default=0.01, type=float,
                        help="g1 component of shear [Default:0.01]")
    parser.add_argument('--cosmic-shear-g2', default=0.01, type=float,
                        help="g1 component of shear [Default:0.01]")
    parser.add_argument('--exposure-time', type=float, default=5520,
                        help='Simulated camera total exposure time seconds.')
    parser.add_argument('--filter-band', choices=['u', 'g', 'r', 'i', 'z',
                                                  'y'],
                        default='i', help='LSST imaging band to simulate')
    parser.add_argument('--image-width', type=int, default=100,
                        help='Simulated mage width in pixels.')
    parser.add_argument('--image-height', type=int, default=100,
                        help='Simulated image height in pixels.')


def run_wl_deb(Args, cat_string):
    """Runs wldeblender package on the input cataloag"""
    path = wldeb_path
    path += "/simulate.py"
    keys = ['exposure_time', 'cosmic_shear_g2', 'image_width', 'filter_band',
            'cosmic_shear_g1', 'image_height']
    kys2 = ['exposure-time', 'cosmic-shear-g2', 'image-width', 'filter-band',
            'cosmic-shear-g1', 'image-height']
    parentdir = os.path.abspath("..")
    name = cat_string + "_" + Args.filter_band
    in_cat = os.path.join(parentdir, 'data',
                          cat_string + '_catalog.fits')
    out_dir = '/global/projecta/projectdirs/lsst/groups/WL/projects/wl-btf/two_gal_blend_data/'
    out_cat = os.path.join(out_dir, 'training_data',
                           name + '_wldeb.fits')
    com = "python " + path + " --no-stamps"
    com += " --catalog-name " + in_cat
    com += " --output-name " + out_cat
    for i, key in enumerate(keys):
        com += " --" + str(kys2[i]) + " "
        com += str(vars(Args)[key])
    p = subprocess.call(com, shell=True, stdout=subprocess.PIPE)
    if p == 0:
        print ("Success!")
    else:
        print ("FAILURE!!!")


def add_noise(Args, cat_string):
    """Adds noise to the wldeb output image"""
    # parentdir = os.path.abspath("..")
    out_dir = '/global/projecta/projectdirs/lsst/groups/WL/projects/wl-btf/two_gal_blend_data/'
    name = cat_string + "_" + Args.filter_band
    in_cat = os.path.join(out_dir, 'training_data',
                          name + '_wldeb.fits')
    out_cat = os.path.join(out_dir, 'training_data',
                           name + '_wldeb_noise.fits')
    # Read the image using descwl's package
    wldeb = descwl.output.Reader(in_cat).results
    wldeb.add_noise(noise_seed=Args.seed)
    galsim.fits.write(wldeb.survey.image, out_cat)


if __name__ == "__main__":
    main()
