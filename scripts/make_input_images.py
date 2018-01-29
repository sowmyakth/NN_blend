"""Creatres input catalog which is then run via WLDeblending package 
to produce galaxy images
"""
import os
import subprocess


def main(Args):
    num = Args.num
    ncols = Args.num_columns
    nrows = num / ncols + 1
    im_h = nrows * Args.stamp_size
    im_w = ncols * Args.stamp_size


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--num', default=10, type=int,
                        help="# of distinct galaxy pairs [Default:10]")
    parser.add_argument('--seed', default=0, type=int,
                        help="Seed to randomly pick galaxies [Default:0]")
    parser.add_argument('--num_ring', default=2, type=int,
                        help="# pairs the image is rotated [Default:2]")
    parser.add_argument('--num_columns', default=8, type=int,
                        help="Number of columns in total field [Default:8]")
    parser.add_argument('--stamp_size', default=240, type=int,
                        help="Size of each stamp in pixels [Default:240]")
    args = parser.parse_args()
    main(args)
