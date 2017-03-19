#!/bin/env python3

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import numpy
import h5py

import gc

from arl.test_support import *
from arl.visibility_operations import *

# Parse arguments
parser = argparse.ArgumentParser(description='Merge contents of one HDF5 file into another. Overwrites existing data.')
parser.add_argument('--out', metavar='out', type=str,
                    help='output file')
parser.add_argument('input', metavar='input', type=str,
                    nargs='*', help='input files')
args = parser.parse_args()

# Open input & output files
out_file = h5py.File(args.out, "a")

def copy(name, obj, to):
    print(obj.name)
    # Is a dataset? Copy!
    if not name in to or isinstance(obj, h5py.Dataset):
        p = obj.parent
        if name in to:
            print("Overwriting", to[name].name)
            del to[name]
        p.copy(obj, to, name=name)
    else:
        for n, d in obj.items():
            copy(n, d, to[name])

for input_name in args.input:
    in_file = h5py.File(input_name, "r+")
    copy("/", in_file, out_file)
    in_file.close()

out_file.close()

