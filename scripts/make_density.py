#!/bin/env python3

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import h5py
import numpy

import arl.test_support

# Parse arguments
parser = argparse.ArgumentParser(description='Make density bins from visibilities\' UVW distribution')
parser.add_argument('input', metavar='input', type=argparse.FileType('r'),
                    help='input visibilities (HDF5)')
parser.add_argument('density', metavar='density', type=argparse.FileType('w'),
                    help='density output file')
parser.add_argument('--theta', dest='theta', type=float, default=0.08,
                    help='Field of view size (l/m range)')
parser.add_argument('--lambda', dest='lam', type=float, default=300000,
                    help='Grid size (u/v range)')
parser.add_argument('--wmax', dest='wmax', type=float, default=205,
                    help='Maximum w value')
parser.add_argument('--wstep', dest='wstep', type=float, default=1.5,
                    help='Step length for w coordinates')
parser.add_argument('--du', dest='du', type=float, default=100,
                    help='u/v bin size (cells)')
parser.add_argument('--dw', dest='dw', type=float, default=1,
                    help='w bin size (steps)')
args = parser.parse_args()

# Determine bin dimensions
wmax = args.wmax
umax = args.lam/2
wstep = args.wstep
args.ustep = ustep = 1/args.theta
wsize = 2*numpy.round(wmax / wstep)
usize = 2*numpy.round(umax / ustep)
print("Grid size:   %d x %d x %d" % (usize, usize, wsize))
print("Bin size:    %d x %d x %d" % (args.du, args.du, args.dw))

ucount = 2*int(numpy.round(usize / 2 / args.du))+1
wcount = 2*int(numpy.round(wsize / 2 / args.dw))+1
print("Bin count:   %d x %d x %d (%d MB)" % (ucount, ucount, wcount,
                                             numpy.dtype('int').itemsize*ucount*ucount*wcount/1000000))
print()

# Coordinate translations
counts = numpy.array([ucount-1, ucount-1, wcount-1])
step_sizes = numpy.array([ustep*usize,ustep*usize,wstep*wsize])
mids = numpy.array([ucount/2,ucount/2,wcount/2],dtype=int)
def bin_to_uvw(iuvw, coords=slice(0,3)):
    return (iuvw - mids[coords]) / counts[coords] * step_sizes[coords]
def uvw_to_bin(uvw):
    return numpy.round(counts * uvw / step_sizes).astype(int) + mids

# Open input file
print("Reading %s..." % args.input.name)
input = h5py.File(args.input.name, "r", driver='core')

# Get baselines
print("Reading baselines...")
viss = arl.test_support.import_visibility_baselines_from_hdf5(input, None, ['uvw'])
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

# Normalise
v_negative = uvw[:,1] > 0
uvw[v_negative] = -uvw[v_negative]
iuvw = numpy.round(counts * uvw / step_sizes).astype(int) + mids

# Create densities
density = numpy.zeros((wcount, ucount, ucount), dtype=int)
for iu,iv,iw in iuvw:
    density[iw, iv, iu] += 1

print("Writing densities to %s..." % args.density.name)
numpy.save(args.density.name, density)
