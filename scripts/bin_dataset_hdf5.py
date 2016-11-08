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
parser = argparse.ArgumentParser(description='Collect and coalesce visibilities for a baseline from OSKAR files.')
parser.add_argument('input', metavar='files', type=argparse.FileType('r'),
                    nargs='*',
                    help='input files')
parser.add_argument('--out', dest='out', metavar='OUT',
                    type=str, required=True,
                    help='HD5 file to write visibility data to')
parser.add_argument('--freqs', dest='freqs', metavar='FREQS',
                    type=int, required=True,
                    help='Maximum number of frequecies')
args = parser.parse_args()

# Open output file
f = h5py.File(args.out, "a")

# Loop through files
for i, inp in enumerate(args.input):

    # Make sure we are going into this with a clean slate
    gc.collect()

    # Read
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

        # Check whether we have the group, and it contains our frequency
        grp_name = '/vis/%d/%d' % (a1, a2)
        if grp_name in f.keys():
            if len(set(f[grp_name+"/frequency"]).intersection(vis.frequency)) != 0:
                print("s", end="", flush=True)
                continue

        gdata = data_by_antenna.groups[j]

        # Create group, if necessary
        if grp_name not in f.keys():

            # Create table
            vis = Visibility(vis, data = gdata)

            # Write, resizable
            max_shape = list(gdata['vis'].shape)
            max_shape[1] = args.freqs
            maxshape={ 'frequency': (args.freqs,),
                       'weight': max_shape,
                       'vis': max_shape }
            export_visibility_to_hdf5(vis, f, grp_name, maxshape=maxshape)

        else:

            # Resize visibilites and frequencies
            vis_ds = f[grp_name+'/vis']
            vis_shape = list(vis_ds.shape)
            vis_shape[1] += len(vis.frequency)
            vis_ds.resize(vis_shape)
            vis_ds[:,-len(vis.frequency):,:] = gdata['vis']

            # Do the same for frequencies
            freq_ds = f[grp_name+'/frequency']
            freq_shape = list(freq_ds.shape)
            freq_shape[0] += 1
            freq_ds.resize(freq_shape)
            freq_ds[-len(vis.frequency):] = vis.frequency

    f.flush()
    print(" done")
