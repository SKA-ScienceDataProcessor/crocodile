#!/bin/env python3

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import h5py

from arl.test_support import *

# Parse arguments
parser = argparse.ArgumentParser(description='Coalesce binned visibilities.')
parser.add_argument('input', metavar='input', nargs='*',
                    help='input files')
parser.add_argument('--out', dest='out', metavar='OUT',
                    type=argparse.FileType('w'), required=True,
                    help='HD5 file to write visibility data to')
args = parser.parse_args()

# Open output file
f = h5py.File(args.out.name, "w")

for inp in args.input:

    # Read oskar file
    print("Reading", inp, "...")
    vis = import_visibility_from_fits(inp)

    # Write to HDF5
    export_visibility_to_hdf5(vis, f, "/" + os.path.splitext(inp)[0])
    f.flush()

f.close()
