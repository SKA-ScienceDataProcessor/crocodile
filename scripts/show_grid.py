#!/bin/env python3

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import h5py
import numpy
import scipy.optimize

from utils.visualise import show_image

# Parse arguments
parser = argparse.ArgumentParser(description='Visualise a grid')
parser.add_argument('input', type=str, help='input file')
parser.add_argument('--theta', dest='theta', type=float, required=True,
                    help='Field of view size')
parser.add_argument('--norm', type=float,
                    help='Use colour scale centered at zero')
args = parser.parse_args()

# Read file
data = np.fromfile(args.input, dtype=np.complex128, count=-1, sep='')
size = data.size
width = int(math.sqrt(size))

show_grid(data.reshape((width, width)), args.input, norm=args.norm)
