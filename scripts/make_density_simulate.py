#!/bin/env python3

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import argparse
from astropy import constants as const
import astropy.coordinates
import astropy.time
from astropy import units as u
import h5py
import itertools
import numpy
import numpy.linalg
import numba

import arl.test_support
import arl.visibility_operations
from crocodile.simulate import *

# Parse arguments
parser = argparse.ArgumentParser(description='Make density bins from visibilities\' UVW distribution')
parser.add_argument('config', metavar='config', type=str,
                    help='antenna configuration (e.g. LOWBD2)')
parser.add_argument('density', metavar='density', type=argparse.FileType('w'),
                    help='density output file')
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
parser.add_argument('--tsnap', dest='tsnap', type=float, default=45,
                    help='Snapshot length (seconds)')
parser.add_argument('--freq-min', dest='freq_min', type=float, default=10*numpy.ceil(35 / 1.35),
                    help='Start frequency (MHz)')
parser.add_argument('--freq-max', dest='freq_max', type=float, default=350,
                    help='End frequency (MHz)')
parser.add_argument('--dt-min', dest='dt_min', type=float, default=0.9, # dump time
                    help='Minimum time step between visibilities (s)')
parser.add_argument('--df-min', dest='df_min', type=float, default=300/65536, # frequency resolution
                    help='Minimum frequency step between visibilities (MHz)')
parser.add_argument('--dt-max', dest='dt_max', type=float, default=10, # ionospheric time scale
                    help='Maximum time step between visibilities (s)')
parser.add_argument('--df-max', dest='df_max', type=float, default=1.25, # ionoshperic frequency scale
                    help='Maximum frequency step between visibilities (MHz)')
parser.add_argument('--grid-step', dest='grid_step', type=float, default=0.6,
                    help='Distance of visibilities in grid after averaging (cells)')
parser.add_argument('--subbin', dest='subbin', type=float, default=8,
                    help='How finely to subdivide bins for accuracy (antialiasing amount, roughly)')
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
@numba.jit(nopython=True)
def uvw_to_bin(uvw):
    return numpy.floor(counts * uvw / step_sizes+0.5).astype(numpy.int_) + mids

# Create configuration
config = arl.test_support.create_named_configuration(args.config)
ants_xyz = numpy.array(config.xyz)
print('Config:      %s, %d antennas at %s, %.2f' %
      (args.config, len(config.xyz),
       config.location.longitude.to(u.deg).value,
       config.location.latitude.to(u.deg).value))
zenith = astropy.coordinates.AltAz(az=0*u.deg, alt=90*u.deg, location=config.location)
zenith = SkyCoord(zenith, obstime=astropy.time.Time.now()).transform_to('icrs')
ha0 = -args.tsnap/2 /3600/24 * 360 * u.deg
ha1 = args.tsnap/2 /3600/24 * 360 * u.deg
print("Hour angles: %s - %s" % (ha0, ha1))
print("Frequencies: %.1f - %.1f MHz" % (args.freq_min, args.freq_max))
print()

# Fit a plane. "Inspired" by ARL code
def fit_uvwplane(uvw):
    su2 = numpy.sum(uvw[:,0] * uvw[:,0])
    sv2 = numpy.sum(uvw[:,1] * uvw[:,1])
    suv = numpy.sum(uvw[:,0] * uvw[:,1])
    suw = numpy.sum(uvw[:,0] * uvw[:,2])
    svw = numpy.sum(uvw[:,1] * uvw[:,2])
    det = su2 * sv2 - suv ** 2
    p = (sv2 * suw - suv * svw) / det
    q = (su2 * svw - suv * suw) / det
    return p,q

# Fit uv-plane to uvw resolution with phase centre at zenith (most
# likely close enough to optimal, assuming the telescope is built on
# the ground)
print('Zenith:      RA %s, Dec %s' % (zenith.ra, zenith.dec))
pc = zenith
dec = zenith.dec.to(u.rad).value

for _ in range(2):

    # Apply corrections to phase centre to minimise w-term
    p,q = fit_uvwplane(xyz_to_baselines(numpy.array(config.xyz), [0], dec))
    pc = SkyCoord(ra=pc.ra-p*u.rad, dec=pc.dec-q*u.rad)
    print('Phase Centre: RA %s, Dec %s' % (pc.ra, pc.dec))
    dec = pc.dec.to(u.rad).value

# Check fit
print(fit_uvwplane(xyz_to_baselines(numpy.array(config.xyz), [0], dec)))

# Check range
uvw_0 = xyz_to_baselines(numpy.array(config.xyz), [ha0], dec)
uvw_1 = xyz_to_baselines(numpy.array(config.xyz), [ha1], dec)
c = const.c.value
print("u range:     %.2f - %.2f lambda" % (
    numpy.min([uvw_0[:,0], uvw_1[:,0]]) * args.freq_max * 1000000 / c,
    numpy.max([uvw_0[:,0], uvw_1[:,0]]) * args.freq_max * 1000000 / c))
print("v range:     %.2f - %.2f lambda" % (
    numpy.min([uvw_0[:,1], uvw_1[:,1]]) * args.freq_max * 1000000 / c,
    numpy.max([uvw_0[:,1], uvw_1[:,1]]) * args.freq_max * 1000000 / c))
print("w range:     %.2f - %.2f lambda" % (
    numpy.min([uvw_0[:,2], uvw_1[:,2]]) * args.freq_max * 1000000 / c,
    numpy.max([uvw_0[:,2], uvw_1[:,2]]) * args.freq_max * 1000000 / c))
print()

# Create densities
density = numpy.zeros((wcount, ucount, ucount), dtype=int)
@numba.jit(nopython=True)
def fill_density(density, uvw00, nvis, tlen, flen, t_duvw, f_duvw):
    for t in range(int(tlen)):
        for f in range(int(flen)):
            b = uvw_to_bin(uvw00 + t * t_duvw + f * f_duvw)
            density[b[2],b[1],b[0]] += nvis

# Loop through baselines
nvis_total = 0
for a0 in range(ants_xyz.shape[0]):
  print("%d: " % a0, end='', flush=True)
  nbins = 0
  nvis = 0
  for a1 in range(a0+1, ants_xyz.shape[0]):

    # Determine extreme points
    dxyz = ants_xyz[a0] - ants_xyz[a1]
    if dxyz[1] < 0:
        dxyz = -dxyz
    uvw0 = xyz_to_uvw(dxyz, ha0, dec)
    uvw1 = xyz_to_uvw(dxyz, ha1, dec)

    # Scale by frequency
    uvw00 = uvw0 * args.freq_min * 1000000 / c
    uvw01 = uvw0 * args.freq_max * 1000000 / c
    uvw10 = uvw1 * args.freq_min * 1000000 / c
    uvw11 = uvw1 * args.freq_max * 1000000 / c
    assert(numpy.max(numpy.abs([uvw00[2], uvw01[2], uvw10[2], uvw11[2]])) < args.wmax)

    # Determine number of steps for averaging
    duv_t = numpy.sqrt(max(numpy.sum((uvw00 - uvw10)**2),
                           numpy.sum((uvw01 - uvw11)**2)))
    duv_f = numpy.sqrt(max(numpy.sum((uvw00 - uvw01)**2),
                           numpy.sum((uvw10 - uvw11)**2)))
    cells_t = int(numpy.ceil(max(
        1, args.tsnap / args.dt_max, duv_t / ustep / args.grid_step)))
    cells_t = min(cells_t, int(args.tsnap / args.dt_min))
    cells_f = int(numpy.ceil(max(
        1, (args.freq_max - args.freq_min) / args.df_max, duv_f / ustep / args.grid_step)))
    cells_f = min(cells_f, int((args.freq_max - args.freq_min) / args.df_min))
    nvis += cells_t * cells_f

    # Determine how many bins we are covering
    dw_t = numpy.sqrt(max(numpy.sum((uvw00[2] - uvw10[2])**2),
                          numpy.sum((uvw01[2] - uvw11[2])**2)))
    dw_f = numpy.sqrt(max(numpy.sum((uvw00[2] - uvw01[2])**2),
                          numpy.sum((uvw10[2] - uvw11[2])**2)))
    bins_uv_t = counts[0] * duv_t / step_sizes[0]
    bins_uv_f = counts[0] * duv_f / step_sizes[0]
    bins_w_t = counts[2] * dw_t / step_sizes[2]
    bins_w_f = counts[2] * dw_f / step_sizes[2]

    # From that determine number of steps. We deliberately choose the
    # number of steps such that we hit individual bins multiple times
    # so that partial overlaps are represented roughly correctly.
    steps_t = int(numpy.ceil(max(1, bins_uv_t, bins_w_t) * args.subbin))
    steps_f = int(numpy.ceil(max(1, bins_uv_f, bins_w_f) * args.subbin))
    nbins += steps_t * steps_f

    fill_density(density,
                 uvw00,
                 (cells_t * cells_f) // (steps_t * steps_f),
                 steps_t, steps_f,
                 (uvw10 - uvw00) / int(steps_t),
                 (uvw00 - uvw01) / int(steps_f))

  print(" %d visibilities, %d bin updates" % (nvis, nbins))
  nvis_total += nvis

print("done, %d visibilities total\n" % nvis_total)

print("Writing densities to %s..." % args.density.name)
numpy.save(args.density.name, density)
