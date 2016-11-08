#!/bin/env python3

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

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

# Parse arguments
parser = argparse.ArgumentParser(description='Grid a data set')
parser.add_argument('input', metavar='input', type=argparse.FileType('r'),
                    help='input visibilities')
parser.add_argument('-N', dest='N', type=int, default=1,
                    help='Process parallelism')
parser.add_argument('--theta', dest='theta', type=float, required=True, default=0.08,
                    help='Field of view size')
parser.add_argument('--lambda', dest='lam', type=float, required=True, default=300000,
                    help='Grid size')
parser.add_argument('--grid', dest='grid', type=argparse.FileType('w'),
                    help='grid output file')
parser.add_argument('--image', dest='image', type=argparse.FileType('w'),
                    help='image output file')
parser.add_argument('--wkern', dest='wkern', type=argparse.FileType('r'),
                    help='w-kernel file to use for w-projection')
parser.add_argument('--akern', dest='akern', type=argparse.FileType('r'),
                    help='A-kernel file to use for w-projection')
parser.add_argument('--kern-cache', dest='kern_cache', type=int,
                    help='Size of A-kernel cache')
parser.add_argument('--quick', dest='method', const='quick', action='store_const',
                    help='Only use one visibility from every baseline')
parser.add_argument('--psf', dest='psf', const=True, default=False, action='store_const',
                    help='generate point spread function')
parser.add_argument('--show-grid', dest='show_grid', const=True, default=False, action='store_const',
                    help='Open a matplotlib window to inspect the result grid')
parser.add_argument('--show-image', dest='show_image', const=True, default=False, action='store_const',
                    help='Open a matplotlib window to inspect the result image')
args = parser.parse_args()

# Open input file
print("Reading %s..." % args.input.name)
input = h5py.File(args.input.name, "r")

# Get baselines
print("Reading baselines...")
viss = arl.test_support.import_visibility_baselines_from_hdf5(input)
print("Got %d visibility chunks" % len(viss))

# Generate UVW and visibilities
if args.method == 'quick':

    # Select one visibility from every chunk
    uvw = numpy.array([vis.uvw_lambda(0)[0] for vis in viss])
    src = numpy.hstack([
        [vis.antenna1[0] for vis in viss],
        [vis.antenna2[0] for vis in viss],
        [vis.time[0] for vis in viss],
        [vis.frequency[0] for vis in viss]
    ])
    vis = numpy.array([vis.vis[0,0,0] for vis in viss])

else:

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
    src = collect_blocks(
        lambda vis, chan: numpy.transpose([
            vis.antenna1,
            vis.antenna2,
            vis.time,
            vis.frequency[chan] * numpy.ones(vis.time.shape)
        ]))
    vis = collect_blocks(lambda vis, chan: vis.vis[:,chan,:])[:,0]

# Show statistics
print()
print("Have %d visibilities" % vis.shape[0])
print("u range:     %.2f - %.2f lambda" % (numpy.min(uvw[:,0]), numpy.max(uvw[:,0])))
print("v range:     %.2f - %.2f lambda" % (numpy.min(uvw[:,1]), numpy.max(uvw[:,1])))
print("w range:     %.2f - %.2f lambda" % (numpy.min(uvw[:,2]), numpy.max(uvw[:,2])))
print("Antennas:    %d - %d"            % (numpy.min(src[:,0]), numpy.max(src[:,1])))
print("t range:     %.6f - %.6f MJD UTC" %(numpy.min(src[:,2]), numpy.max(src[:,2])))
print("f range:     %.2f - %.2f MHz"    % (numpy.min(src[:,3])/1e6, numpy.max(src[:,3])/1e6))
print()

# Initialise gridder
if args.wkern is None:

    # Simple imaging without convolution. No source dependency.
    print("Gridder: Simple imaging")
    grid_fn = simple_imaging
    grid_pars = {}
    src = numpy.zeros((src.shape[0],0))

else:

    # Determine w-cache steps
    wkern_file = h5py.File(args.wkern.name, "r", driver='core')
    wsteps = numpy.array(sorted(map(float, wkern_file['wkern/%s' % args.theta].keys())))
    wstep = wsteps[1] - wsteps[0]
    print("w kernels:   %.2f - %.2f lambda (step %.2f lambda)" % (min(wsteps), max(wsteps), wstep))

    # Make a custom kernel cache that reads from the hdf5 file
    def closest(xs, x):
        return xs[numpy.argmin(numpy.abs(numpy.array(xs) - x))]
    def w_kernel_fn(theta, w):
        kernw = closest(wsteps, w)
        #print("w=", kernw)
        return wkern_file['wkern/%s/%s/kern' % (theta, kernw)]
    w_cache = pylru.FunctionCacheManager(w_kernel_fn, len(wsteps))

    # A-kernels?
    if args.akern is None:

        # Just pure w-projection, also no source dependency.
        print("Gridder: W-projection")
        grid_fn = w_cache_imaging
        grid_pars = { 'wstep': wstep, 'kernel_cache': w_cache }
        src = numpy.zeros((src.shape[0],0))

    else:

        # Open A-kernel file
        akern_file = h5py.File(args.akern.name, "r", driver='core')
        times = list(map(float, akern_file['akern/%s/0' % args.theta]))
        freqs = list(map(float, akern_file['akern/%s/0/%s' % (args.theta, times[0])]))
        print("A kernels:   %d antennas" %
              max(map(int, akern_file['akern/%s' % args.theta])))
        print(" \" t range:  %.6f - %.6f MJD UTC (step %.2f s)" % (
            numpy.min(times), numpy.max(times), (times[1] - times[0]) * 24 * 3600))
        print(" \" f range:  %.2f - %.2f MHz (step %.2f MHz)" % (
            numpy.min(freqs)/1e6, numpy.max(freqs)/1e6, (freqs[1] - freqs[0]) /1e6))

        # Make a custom kernel cache that reads from the hdf5 file
        def a_kernel_fn(theta, a, t, f):
            # print("a=%d, t=%f, f=%f" % (a, t, f))
            return akern_file['akern/%s/%d/%s/%d/kern' % (theta, a, t, f)]
        a_cache = pylru.FunctionCacheManager(a_kernel_fn, args.kern_cache)

        # And yet another cache for AW-combinations
        aw_cache = pylru.FunctionCacheManager(aw_kernel_fn(a_cache, w_cache), args.kern_cache)

        # Round time and frequency to closest one that we actually have data for
        def tf_round_fn(theta, w, a1, a2, t, f):
            kernt = closest(times, t)
            kernf = closest(freqs, f)
            return aw_cache(theta, w, a1, a2, kernt, kernf)

        # Use w-imaging function, but with AW kernels
        print("Gridder: AW-projection")
        grid_fn = w_cache_imaging
        grid_pars = { 'wstep': wstep, 'kernel_cache': tf_round_fn }

# Generate PSF? Set all visibilities to 1
if args.psf:
    vis[:] = 1.0

# Weight, mirror visibilities with negative v
print("\nWeight...")
wt = doweight(args.theta, args.lam, uvw, numpy.ones(len(uvw)))
uvw, vis = mirror_uvw(uvw, vis)

# Make grid
N = max(1, args.N)
if N == 1:
    print("Gridding...")
    uvgrid = grid_fn(args.theta, args.lam, uvw, src, wt * vis,
                     **grid_pars)
else:

    # Crude attempt at parallelisation to make imaging big datasets
    # at least bearable...
    print("Make shared grid...")
    step = vis.shape[0] // N
    px = int(round(args.theta * args.lam))
    grid_arr = Array(ctypes.c_double, px * px * 2) # slow!
    uvgrid = numpy.frombuffer(grid_arr.get_obj(), dtype=complex).reshape((px, px))
    uvgrid[:] = 0

    print("Gridding using %d procs (%d visibilities each)..." % (N, step))
    def do_grid(start):
        uvg = grid_fn(args.theta, args.lam,
                      uvw[start:start+step,:],
                      src[start:start+step,:],
                      wt[start:start+step] * vis[start:start+step],
                      **grid_pars)
        with grid_arr.get_lock():
            uvgrid = numpy.frombuffer(grid_arr.get_obj(), dtype=complex).reshape((px, px))
            uvgrid += uvg
            print("... worker %d done" % (start / step))

    procs = []
    for start in range(0, vis.shape[0], step):
        p = Process(target=do_grid, args=(start,))
        p.start()
        procs.append(p)

    # Accumulate grids
    for p in procs:
        p.join()

# Make hermitian
uvgrid = make_grid_hermitian(uvgrid)

if args.grid is not None:
    uvgrid.tofile(args.grid)
    args.grid.close()
if args.show_grid:
    util.visualize.show_grid(uvgrid, "result", args.theta)

# FFT, if requested
if args.image is not None or args.show_image:
    print("FFT...")
    img = numpy.real(ifft(uvgrid))
    if args.image is not None:
        img.tofile(args.image)
        args.image.close()
    if args.show_image:
        util.visualize.show_image(img, "result", args.theta)
