#!/bin/env python3

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import astropy
import astropy.units as units
from astropy.coordinates import SkyCoord
import numpy
import numpy.linalg

from arl.test_support import import_visibility_from_oskar, export_visibility_to_fits
from arl.visibility_operations import *

import gc

# Parse arguments
parser = argparse.ArgumentParser(description='Collect and coalesce visibilities for a baseline from OSKAR files.')
parser.add_argument('input', metavar='files', type=argparse.FileType('r'),
                    nargs='*',
                    help='input files')
parser.add_argument('--out', dest='pre', metavar='pre', type=str, required=True,
                    help='output prefix')
args = parser.parse_args()

# Loop through files
for i, inp in enumerate(args.input):

    # Make sure we are going into this with a clean slate
    gc.collect()

    # Read. First one will already be loaded
    print("Reading", inp.name, "...")
    vis = import_visibility_from_oskar(inp.name)
    gc.collect()

    # Loop through visibilities
    print("Grouping...")
    data_by_antenna = vis.data.group_by(['antenna1', 'antenna2'])
    gc.collect()

    # Loop through baselines
    print("Collecting...", end="", flush=True)
    last_a1 = -1
    for j, key in enumerate(data_by_antenna.groups.keys):

        # Interested in this baseline?
        a1 = key['antenna1']
        a2 = key['antenna2']
        if a1 != last_a1:
            print(" %d" % a1, end="", flush=True)
            last_a1 = a1

        # Concatenate visibilities
        v = data_by_antenna.groups[j]['vis']
        with open(args.pre + "%d-%d.bin" % (a1, a2), "ab") as f:
            f.write(v.tobytes())

    print(" done")
