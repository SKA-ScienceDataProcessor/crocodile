#!/bin/env python3

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import matplotlib as mpl
mpl.use("Agg")

import argparse
import h5py
import itertools
import numpy
import pylru
from multiprocessing import Process, Array, Queue
import ctypes

import arl.test_support
from crocodile.synthesis import *
import util.visualize

import matplotlib.pyplot as plt

# Parse arguments
parser = argparse.ArgumentParser(description='Grid a data set')
parser.add_argument('input', metavar='input', type=argparse.FileType('r'),
                    help='input visibilities')
parser.add_argument('--lambda', dest='lam', type=float,
                    help='Size of uvw-plane')
parser.add_argument('--out', dest='out', type=argparse.FileType('w'),
                    help='Output image')
args = parser.parse_args()

# Open input file
print("Reading %s..." % args.input.name)
input = h5py.File(args.input.name, "r")

# Get baselines
print("Reading baselines...")
viss = arl.test_support.import_visibility_baselines_from_hdf5(input)
print("Got %d visibility chunks" % len(viss))

# Utility to collect data from visibility blocks
def collect_blocks(prop):
    result = []
    for vis in viss:
        vres = []
        for chan in range(len(vis.frequency)):
            vres.append(prop(vis, chan))
            result.append(numpy.vstack(numpy.transpose(vres, (1,0,2))))
    return numpy.vstack(result)

uvw = collect_blocks(lambda vis, chan: vis.uvw_lambda(chan))

# Show statistics
print()
print("Have %d visibilities" % uvw.shape[0])
print("u range:     %.2f - %.2f lambda" % (numpy.min(uvw[:,0]), numpy.max(uvw[:,0])))
print("v range:     %.2f - %.2f lambda" % (numpy.min(uvw[:,1]), numpy.max(uvw[:,1])))
print("w range:     %.2f - %.2f lambda" % (numpy.min(uvw[:,2]), numpy.max(uvw[:,2])))
print()

plt.scatter(uvw[:,0], uvw[:,1], c=uvw[:,2], lw=0, s=.01)
plt.scatter(-uvw[:,0], -uvw[:,1], c=-uvw[:,2], lw=0, s=.01)
plt.xlabel('u [lambda]')
plt.ylabel('v [lambda]')
if args.lam is not None:
    plt.xlim(-args.lam/2, args.lam/2)
    plt.ylim(-args.lam/2, args.lam/2)
plt.colorbar()
if args.out is not None:
    plt.savefig(args.out.name, dpi=1200)
else:
    plt.show()

