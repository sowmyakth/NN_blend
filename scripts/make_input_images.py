"""Creatres input catalog which is then run via WLDeblending package 
to produce galaxy images
"""
import os
import subprocess
import argparse
import make_input_catalog


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    catalog_group = parser.add_argument_group('Catlog options',
                                              'Input catalog options')
    make_input_catalog.add_args(catalog_group)
    wldeb_group = parser.add_argument_group('WLDEB package options',
                                            'Input options to WLdeb package')
    second_args(wldeb_group)
    args = parser.parse_args()
    make_input_catalog.main(args)
    # import ipdb;ipdb.set_trace()
    num = args.num
    ncols = args.num_columns
    nrows = int(num / ncols) + 1
    args.image_height = nrows * args.stamp_size
    args.image_width = ncols * args.stamp_size
    run_wl_deb(args, 'gal_pair')
    run_wl_deb(args, 'central_gal')


def second_args(parser):
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
    path = "/global/homes/s/sowmyak/blending_tutorial/Blending_tutorial"
    path += "/WeakLensingDeblending/simulate.py"
    keys = ['exposure_time', 'cosmic_shear_g2', 'image_width', 'filter_band',
            'cosmic_shear_g1', 'image_height']
    kys2 = ['exposure-time', 'cosmic-shear-g2', 'image-width', 'filter-band',
            'cosmic-shear-g1', 'image-height']
    parentdir = os.path.abspath("..")
    in_cat = os.path.join(parentdir, 'data',
                          cat_string + '_catalog.fits')
    out_cat = os.path.join(parentdir, 'data',
                           cat_string + '_wldeb.fits')
    com = "python " + path + " --no-stamps"
    com += " --catalog-name " + in_cat
    com += " --output-name " + out_cat
    arguments = [com, ]
    for i, key in enumerate(keys):
        #arguments.append("--" + str(key))
        #arguments.append(str(vars(Args)[key]))
        com += " --" + str(kys2[i]) + " "
        com += str(vars(Args)[key])
    print (com)
    p = subprocess.call(com, shell=True, stdout=subprocess.PIPE)
    print (p)


if __name__ == "__main__":
    main()
