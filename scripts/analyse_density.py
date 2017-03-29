#!/bin/env python3

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import argparse
import h5py
import itertools
import numpy
import random
import time
import cProfile, pstats, io

import arl.test_support
from crocodile.bins import *
from crocodile.synthesis import *
import util.visualize

# Parse arguments
parser = argparse.ArgumentParser(description='Grid a data set')
parser.add_argument('density', metavar='density',
                    type=argparse.FileType('r'),
                    help='input grid density bins')
parser.add_argument('--theta', dest='theta', type=float, default=0.08,
                    help='Field of view size (l/m range)')
parser.add_argument('--lambda', dest='lam', type=float, default=300000,
                    help='Grid size (u/v range)')
parser.add_argument('--wmax', dest='wmax', type=float, default=1300,
                    help='Maximum w value')
parser.add_argument('--wstep', dest='wstep', type=float, default=1.5,
                    help='Step length for w coordinates')
parser.add_argument('--du', dest='du', type=float, default=100,
                    help='u/v bin size (cells)')
parser.add_argument('--dw', dest='dw', type=float, default=10,
                    help='w bin size (steps)')
parser.add_argument('--epsw', dest='epsw', type=float, default=0.01,
                    help='W-kernel accuracy')
parser.add_argument('--asize', dest='asize', type=int, default=9,
                    help='A-kernel size (cells)')
parser.add_argument('--scalew', dest='scalew', type=float, default=1,
                    help='Scale problem up by given factor in w direction (larger snapshot, roughly)')
parser.add_argument('--save', dest='save', type=argparse.FileType('w'),
                    help='Save end state to file')
parser.add_argument('--load', dest='load', type=argparse.FileType('r'),
                    help='Load start state from file')
parser.add_argument('--dump', dest='dump', action='store_true',
                    help='Dump bins in end state')
parser.add_argument('--savefig', dest='savefig', type=argparse.FileType('w'),
                    help='Save image of end state')
parser.add_argument('--showfig', dest='showfig', action='store_true',
                    help='Show image of end state')
parser.add_argument('--profile', dest='profile', action='store_true',
                    help='Use cProfile to produce a profiling report')
parser.add_argument('--tmax', dest='tmax', type=float, default=0.5,
                    help='Start temperature for simulated annealing')
parser.add_argument('--tmin', dest='tmin', type=float, default=0.0001,
                    help='End temperature for simulated annealing')
parser.add_argument('--steps', dest='steps', type=int, default=200000,
                    help='Number of simulated annealing steps')
parser.add_argument('--updates', dest='updates', type=int, default=100,
                    help='Number of progress updates to show')

args = parser.parse_args()

# Determine bin dimensions
wmax = args.wmax*args.scalew
umax = args.lam/2
args.wstep = wstep = args.wstep*args.scalew
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
def uvw_to_bin(uvw, coords=slice(0,3)):
    return numpy.round(counts[coords] * uvw / step_sizes[coords]).astype(int) + mids[coords]

# Read densities
density = args.scalew * numpy.load(args.density.name)
assert density.shape == (wcount, ucount, ucount), \
    "Density bins have wrong shape: Got %s, but expected %s!" % (density.shape, (wcount, ucount, ucount))

# Calculate cost for FFT and Reprojection
c_FFT = 5 * numpy.ceil(numpy.log(usize*usize)/numpy.log(2))*usize*usize
c_Reproject = 50 * (usize*usize)

if args.load is None:

    # Create grid bins
    initial_bins = []
    bin_grid_size = 500 / args.theta
    bins_grid = int( (umax+bin_grid_size-1) // bin_grid_size )
    for iu in range(-bins_grid, bins_grid):
        for iv in range(-bins_grid, bins_grid):
            u0,v0 = uvw_to_bin(numpy.array([iu * bin_grid_size, iv * bin_grid_size]), coords=slice(0,2))
            u1,v1 = uvw_to_bin(numpy.array([(iu+1) * bin_grid_size, (iv+1) * bin_grid_size]), coords=slice(0,2))
            initial_bins += [(max(0, u0), min(ucount, u1), max(0, v0), min(v1, ucount), 0, wcount)]

    # Create initial bin
    bs1 = BinSet(bin_to_uvw, args, density, [(0, ucount, 0, ucount, 0, wcount)],
                 name=os.path.splitext(os.path.basename(args.density.name))[0],
                 add_cost = c_FFT+c_Reproject)
    bs = BinSet(bin_to_uvw, args, density, initial_bins,
                name=os.path.splitext(os.path.basename(args.density.name))[0],
                add_cost = c_FFT+c_Reproject)
    assert bs.nvis0 == bs1.nvis0, "%d %d" % (bs.nvis0, bs1.nvis0)
    b = bs1.bins[0]
    print("Start:        %s" % bs.state)
    print("Visibilities: %d" % bs.nvis0)
    print("Gridding:     %.2f Gflop" % (b.cost_direct / 1000000000))
    if b.wplanes > 0:
        print("  w-stacking: %.2f Gflop (%d u-chunks, %d v-chunks, %d w-planes)" % (
            b.cost / 1000000000, b.uchunks, b.vchunks, b.wplanes))
    print("  bin grid:   %.2f Gflop (%d bins, %d non-empty, max %.2f MB)" % (
        bs.cost0 / 1000000000, len(initial_bins), len(bs.state),
        16 * (bin_grid_size * args.theta) ** 2 / 1000000
    ))
    print("FFT:          %.2f Gflop" % (c_FFT  / 1000000000))
    print("Reproject:    %.2f Gflop" % (c_Reproject / 1000000000))
    print()
    print("Efficiency:   %.2f flop/vis" % ((c_FFT + c_Reproject + b.cost_direct) / b.nvis))
    if b.wplanes > 0:
        print("  w-stacking: %.2f flop/vis" % ((c_FFT + c_Reproject + bs1.cost0) / bs.nvis0))
    print("  bin grid:   %.2f flop/vis" % ((c_FFT + c_Reproject + bs.cost0) / bs.nvis0))
    print()

else:

    print("State read from %s", args.load.name)
    bs = BinSet(bin_to_uvw, args, density,
                name=os.path.splitext(os.path.basename(args.load.name))[0],
                load_state=args.load.name,
                add_cost = c_FFT+c_Reproject)

    print()
    print("Visibilities: %d" % bs.nvis0)
    print("Gridding:     %.2f Gflop (%d bins)" % (bs.cost0 / 1000000000, len(bs.state)))
    print("FFT:          %.2f Gflop" % (c_FFT  / 1000000000))
    print("Reproject:    %.2f Gflop" % (c_Reproject / 1000000000))
    print("Efficiency:   %.2f flop/vis" % ((c_FFT + c_Reproject + bs.cost0) / bs.nvis0))
    print()

# Set parameters
bs.Tmax = args.tmax
bs.Tmin = args.tmin
bs.steps = args.steps
bs.updates = args.updates
bs.save_state_on_exit = False

if args.profile:
    # Start profile
    pr = cProfile.Profile()
    pr.enable()

# Do simulated annealing
start_time = time.time()
result_bins, energy = bs.anneal()

# Save
if args.save is not None:
    print("Saving to %s" % args.save.name)
    bs.save_state(args.save)
    print()
elif time.time() - start_time > 5:
    # Automatically save if we invested more than 5 minutes
    print("Auto-saving state")
    bs.save_state()
    print()

if args.profile:
    # Show profile
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(5)
    print("Profile:")
    print(s.getvalue())

if args.dump:
    print("Resulting bins:")
    for b in sorted(result_bins, key=lambda b: b.cost):
        u0, v0, w0 = bin_to_uvw(numpy.array([b.iu0, b.iv0, b.iw0]))
        u1, v1, w1 = bin_to_uvw(numpy.array([b.iu1, b.iv1, b.iw1]))

        print("u %+d:%+d, v %+d:%+d, w %+d:%+d, %d vis, " % (
            u0, u1, v0, v1, w0, w1, b.nvis), end='')

        if b.wplanes == 0:
            print("direct ", end='')
        else:
            if b.uchunks > 1:
                print("%d u-chunks, " % b.uchunks, end='')
            if b.vchunks > 1:
                print("%d v-chunks, " % b.vchunks, end='')
            print("%d w-planes " % b.wplanes, end='')

            print(" -> %.1f Gflops, %.1f flops/vis" % (b.cost / 1000000000, b.cost / b.nvis))

if args.savefig is not None or args.showfig is not None:
    bs.visualize("Finished, energy = %.f flops/vis" % bs.energy(),
                 save=args.savefig.name if args.savefig is not None else None)
