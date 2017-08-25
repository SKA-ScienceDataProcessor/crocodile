#!/bin/env python3

import sys
import os
import argparse
import numpy
import math



# Parse arguments
parser = argparse.ArgumentParser(description='Visualise a grid')
parser.add_argument('input', type=str, help='input file')
parser.add_argument('--ref', type=str, help='reference file')
parser.add_argument('--dtype', default='float64', type=str, help='numpy element dtype (e.g. float, complex)')
parser.add_argument('--theta', dest='theta', type=float, required=True,
                    help='Field of view size')
parser.add_argument('--norm', type=float,
                    help='Use colour scale centered at zero')
parser.add_argument('--backend', type=str, default='TkAgg',
                    help='matplotlib backend to use')

parser.add_argument('--threshold', type=float, default=0,
                    help='Source detection threshold')
parser.add_argument('--fwhm', type=float, default=1.0,
                    help='Source detection kernel full width half maximum')
parser.add_argument('--npixels', type=int, default=5,
                    help='Source detection number of pixels required for detection')

args = parser.parse_args()

# Read file
data = numpy.fromfile(args.input, dtype=args.dtype, count=-1, sep='')
if args.ref is not None:
    data -= numpy.fromfile(args.ref, dtype=args.dtype, count=-1, sep='')
size = data.size
width = int(math.sqrt(size))
print("Size %dx%d, lambda %g" % (width, width, width / args.theta))
print("Min %g, max %g, mean %g, mean square %g" % (numpy.min(data), numpy.max(data),
                                                   numpy.mean(data), numpy.mean(data**2)))
image = data.reshape((width, width))

# Detect sources, if requested
if args.threshold > 0:

    from astropy.convolution import Gaussian2DKernel, Box2DKernel
    from astropy.stats import gaussian_fwhm_to_sigma
    import astropy.units as u
    from photutils import segmentation
    kernel = Gaussian2DKernel(args.fwhm * gaussian_fwhm_to_sigma,
                              x_size=int(1.5*args.fwhm), y_size=int(1.5*args.fwhm))
    kernel.normalize()
    segments = segmentation.detect_sources(image, args.threshold, npixels=args.npixels, filter_kernel=kernel)
    print("Have %d segments:" % (segments.nlabels))
    props = segmentation.source_properties(image, segments, filter_kernel=kernel)
    for segment in props:
        print("l=%+.6f m=%+.6f intensity=%.f" %
              (segment.xcentroid.value / width * args.theta - args.theta/2,
               segment.ycentroid.value / width * args.theta - args.theta/2,
               segment.max_value))

# Visualise
import matplotlib
matplotlib.use(args.backend)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from util.visualize import show_image
show_image(image, args.input, args.theta, norm=args.norm)
