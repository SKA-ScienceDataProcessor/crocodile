#!/bin/env python3

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import h5py
import numpy

from arl.test_support import import_visibility_from_oskar, export_visibility_to_fits
from crocodile.synthesis import kernel_oversample

# Parse arguments
parser = argparse.ArgumentParser(description='Generate random A kernel file for gridding a dataset')
parser.add_argument('output', type=str, help='output file')
parser.add_argument('--theta', dest='theta', type=float, required=True,
                    help='Field of view size')
parser.add_argument('--phase-random', dest='prand', type=float, default=0.01,
                    help='Amount of phase randomness (default 0.01)')
parser.add_argument('--amp-random', dest='arand', type=float, default=0.01,
                    help='Amount of amplitude randomness (default 0.01)')
parser.add_argument('--ff', dest='ff', type=int, required=True,
                    help='Far field size')
parser.add_argument('--antennas', dest='ants', type=int, required=True,
                    help='Number of antennas')
parser.add_argument('--size', dest='size', type=int, required=True,
                    help='Size of kernels to generate')
parser.add_argument('--time', dest='time', type=float, required=True,
                    help='Time to generate kernels around [MJD UTC]')
parser.add_argument('--time-step', dest='time_step', type=float, required=True,
                    help='Length of time steps [s]')
parser.add_argument('--time-slots', dest='time_slots', type=int, required=True,
                    help='Numer of time steps in either direction')
parser.add_argument('--freq', dest='freq', type=float, required=True,
                    help='Frequency to generate kernels around [MHz]')
parser.add_argument('--freq-step', dest='freq_step', type=float, required=True,
                    help='Length of time steps [MHz]')
parser.add_argument('--freq-slots', dest='freq_slots', type=int, required=True,
                    help='Number of frequency steps in either direction')
parser.add_argument('--overwrite', dest='overwrite', const=True, default=False, action='store_const',
                    help='Overwrite existing kernels in output file?')
args = parser.parse_args()

# Open output file
output = h5py.File(args.output, "a")

# Generate kernels
for a in range(args.ants):
  print("%d " % a, end="", flush=True)
  for i_t in range(-args.time_slots, args.time_slots+1):
    t = args.time + i_t * args.time_step / 24 / 3600
    for i_f in range(-args.freq_slots, args.freq_slots+1):
      f = (int(1000000 * args.freq) + i_f * int(1000000 * args.freq_step))

      # Check whether it already exists
      kern_name = 'akern/%s/%s/%s/%s/kern' % (args.theta, a, t, f)
      if kern_name in output:
          if args.overwrite:
              del output[kern_name]
          else:
              continue

      # Make slightly random image
      def rand_img():
          return numpy.random.rand(args.ff, args.ff) - .5
      img = numpy.exp(2j * numpy.pi * args.prand * rand_img())
      img *= 1.0 + args.arand * rand_img()

      # Transform into kernel
      kern = kernel_oversample(img, args.ff, 1, args.size)[0,0]
      output[kern_name] = kern
      output.flush()

print("done")
output.close()
