#!/bin/env python3

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import h5py
import numpy
import scipy.optimize

from arl.test_support import import_visibility_from_oskar, export_visibility_to_fits
from crocodile.synthesis import w_kernel

# Parse arguments
parser = argparse.ArgumentParser(description='Generate w kernel file for gridding a dataset')
parser.add_argument('output', type=str, help='output file')
parser.add_argument('--theta', dest='theta', type=float, required=True,
                    help='Field of view size')
parser.add_argument('--ff', dest='ff', type=int, required=True,
                    help='Far field size')
parser.add_argument('--wstep', dest='wstep', type=float, required=True,
                    help='Step length for w coordinates')
parser.add_argument('--wcount', dest='wcount', type=int, required=True,
                    help='Number of w-planes to generate kernels for')
parser.add_argument('--size', dest='size', type=int, required=True,
                    help='Size of kernels to generate')
parser.add_argument('--oversample', dest='oversample', type=int, default=8,
                    help='Amount of oversampling to use for kernel (default 8)')
parser.add_argument('--overwrite', dest='overwrite', const=True, default=False, action='store_const',
                    help='Overwrite existing kernels in output file?')
args = parser.parse_args()

# Open output file
output = h5py.File(args.output, "a")

# Calculate recommended w-kernel size
wmax = args.wstep * args.wcount
def recommended_size(eps=0.01):
    usupp = numpy.sqrt((wmax*args.theta/2)**2 + (wmax**1.5 * args.theta / 2 / numpy.pi / eps))
    return 2 * args.theta * usupp
def recommended_diff(eps):
    return abs(recommended_size(eps) - args.size)
print("w range:        %.1f - %.1f lambda" % (-wmax, wmax))
print("Kernel size:    %d px (> %.1f recommended)" % (args.size, recommended_size()))
print("Expected error: %f %%" % (100 * scipy.optimize.minimize(recommended_diff, 0.0001, bounds=[(1e-10,1)]).x))

# Generate kernels
for iw in range(-args.wcount, args.wcount+1):
    w = iw * args.wstep
    print("%f " % w, end="", flush=True)

    # Check whether it already exists
    kern_name = 'wkern/%s/%s/kern' % (args.theta, w)
    if kern_name in output:
        if args.overwrite:
            del output[kern_name]
        else:
            continue

    # Make kernel
    kern = w_kernel(args.theta, w, NpixFF=args.ff, NpixKern=args.size, Qpx=args.oversample)
    output[kern_name] = kern
    output.flush()

print("done")
output.close()
