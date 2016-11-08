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
import h5py

from arl.test_support import *
from arl.visibility_operations import *

# Parse arguments
parser = argparse.ArgumentParser(description='Coalesce binned visibilities.')
parser.add_argument('input', metavar='dirs',
                    nargs='*', type=argparse.FileType('r'),
                    help='input HDF5 files')
parser.add_argument('--oskar', dest='oskar', metavar='VIS',
                    type=argparse.FileType('r'), required=True,
                    help='oskar visibility file for frequency 0')
parser.add_argument('--out-raw', dest='output_raw', metavar='VIS',
                    type=str,
                    help='oskar visibility file for frequency 0')
parser.add_argument('--out', dest='output', metavar='VIS',
                    type=str, required=True,
                    help='oskar visibility file for frequency 0')
parser.add_argument('--ra', dest='ra', metavar='RA', type=float,
                    help='phase centre right ascension (degrees)')
parser.add_argument('--dec', dest='dec', metavar='DC', type=float,
                    help='phase centre declination (degrees)')
parser.add_argument('--a-min', dest='a_min', metavar='A1', type=int, default=0,
                    help='first antenna1 to collect')
parser.add_argument('--a-max', dest='a_max', metavar='A1', type=int, default=65536,
                    help='last antenna1 to collect')
parser.add_argument('--theta', dest='theta', metavar='theta', type=float, required=True,
                    help='Field of view to average for (used to compute cell size)')
parser.add_argument('--step', dest='step', metavar='step', type=float, required=True,
                    help='Step length to average to [cells]')
parser.add_argument('--max-coal-t', dest='maxt', metavar='maxt', type=float, required=True,
                    help='Maximum averaging in time [s]')
parser.add_argument('--max-coal-f', dest='maxf', metavar='maxf', type=float, required=True,
                    help='Maximum averaging in frequency [MHz]')
args = parser.parse_args()

# Open output file
output = h5py.File(args.output, "a")
output_raw = None
if args.output_raw is not None:
    output_raw = h5py.File(args.output_raw, "a")

# Open visibility files
inputs = []
for input in args.input:
    print("Opening", input.name, "...")
    inputs.append(h5py.File(input.name, "r", driver="core"))

# Read oskar file
print("Reading", args.oskar.name, "...")
vis = import_visibility_from_oskar(args.oskar.name)

# Print phase centre
mean_time = astropy.time.Time(numpy.mean(vis.time), format='mjd')
print("Mean time:    %s (%s)" % (mean_time, mean_time.to_datetime()))
print("Phase Centre: RA %.14f DEC %.14f" %
      (vis.phasecentre.ra.to(units.deg).value, vis.phasecentre.dec.to(units.deg).value))

# Determine telescope zenith
altaz = astropy.coordinates.AltAz(
    alt=90*units.deg, az=0*units.deg, obstime=mean_time,
    location=vis.configuration.location)
zenith = altaz.transform_to(astropy.coordinates.ICRS)
print("Zenith:       RA %.14f DEC %.14f" %
      (zenith.ra.to(units.deg).value, zenith.dec.to(units.deg).value))

# Determine baseline plane
ra = args.ra
dec = args.dec
if not ra is None and not dec is None:
    new_pc = SkyCoord(ra=ra, dec=dec, unit=units.deg, frame=astropy.coordinates.ICRS)
else:
    # Crude heuristic to determine the best phase centre. There is
    # probably a better approach, but this seems to converge pretty
    # well.
    def optimise_phase_centre(uvw, pc):
        uvw = numpy.array(uvw) / numpy.sqrt(numpy.sum(uvw**2, axis=1))[:,numpy.newaxis]
        uvw_rot = numpy.transpose([-uvw[:,1],uvw[:,0],uvw.shape[0]*[0]])
        uvw_cross = numpy.cross(uvw, uvw_rot, axis=1)
        x,y,z = numpy.mean(uvw_cross, axis=0) * 2 # seems to give fastest convergance
        return SkyCoord(
            x=numpy.sqrt(1-x**2-y**2),y=x,z=y, # NB different astropy coordinate order
            representation='cartesian',
            frame=pc.skyoffset_frame()).transform_to(astropy.coordinates.ICRS)
    print("Optimising phase centre...", end='', flush=True)
    # Start iteration with zenith
    new_pc = vis.phasecentre
    while True:
        old_pc = new_pc
        new_pc = optimise_phase_centre(phaserotate_visibility(vis, new_pc).uvw, new_pc)
        if abs(new_pc.ra.value-old_pc.ra.value) + abs(new_pc.dec.value-old_pc.dec.value) < 1e-14:
            break
        print('.', end="", flush=True)
    print('')
print("New Centre:   RA %.14f DEC %.14f" %
      (new_pc.ra.to(units.deg).value, new_pc.dec.to(units.deg).value))

# Group frequencies
print("Grouping...")
data_by_antenna = vis.data.group_by(['antenna1', 'antenna2'])
for j, key in enumerate(data_by_antenna.groups.keys):
    a1 = key['antenna1']
    a2 = key['antenna2']
    if a1 < args.a_min or a1 > args.a_max:
        continue
    gdata = data_by_antenna.groups[j]

    # Find group in oskar file
    ntime = gdata['vis'].shape[0]

    # Read baseline from input directories
    v = numpy.ndarray((ntime, 0, 1), dtype=complex)
    freqs = []
    viss = []
    for f in inputs:
        grp = f['vis/%d/%d' % (a1, a2)]
        freqs.append(numpy.array(grp['frequency']))
        viss.append(numpy.array(grp['vis']))

    # Sort
    frequency = numpy.hstack(freqs)
    vis_data = numpy.hstack(viss)

    # Make visibilities
    print("Baseline %d-%d: Got %s visibilities" % (a1, a2, str(vis_data.shape)))
    bl_vis = Visibility(
        vis,
        frequency = frequency,
        vis = vis_data,
        time = gdata['time'],
        uvw = gdata['uvw'],
        weight = numpy.ones((vis_data.shape)),
        antenna1 = gdata['antenna1'],
        antenna2 = gdata['antenna2']
    )
    bl_vis.data.sort("time")

    # Write data
    if output_raw is not None:
        if "vis/%d/%d" % (a1, a2) in output_raw:
            del output_raw["vis/%d/%d" % (a1, a2)]
        export_visibility_to_hdf5(bl_vis, output_raw, "vis/%d/%d" % (a1, a2))

    # Phase rotate
    bl_vis = phaserotate_visibility(bl_vis, new_pc)

    # Determine UVW distance in time at maximum frequency
    max_chan = numpy.argmax(bl_vis.frequency)
    uvw = bl_vis.uvw_lambda(max_chan)
    distance_t = numpy.linalg.norm(uvw[0] - uvw[-1]) * args.theta

    # ... same for frequency
    min_chan = numpy.argmin(bl_vis.frequency)
    uvw2 = bl_vis.uvw_lambda(min_chan)
    distance_f = numpy.linalg.norm(uvw[0] - uvw2[0]) * args.theta

    print("Baseline %d-%d: Distance %f x %f" % (a1, a2, distance_t, distance_f))

    # Decide amount of averaging
    max_time_coal = args.maxt / (24 * 3600 * (numpy.max(bl_vis.time) - numpy.min(bl_vis.time)) / ntime)
    time_coalesce = min(max_time_coal, ntime, ntime / (distance_t / args.step))
    max_freq_coal = args.maxf * 1000000 / ((numpy.max(bl_vis.frequency) - numpy.min(bl_vis.frequency)) / bl_vis.nchan)
    frequency_coalesce = min(max_freq_coal, bl_vis.nchan, bl_vis.nchan / (distance_f / args.step))

    print(" -> Coalescing %d x %d" % (int(time_coalesce), int(frequency_coalesce)))
    cbl_vis = coalesce_visibility(bl_vis, int(time_coalesce), int(frequency_coalesce))

    # Write data
    if "vis/%d/%d" % (a1, a2) in output:
        del output["vis/%d/%d" % (a1, a2)]
    export_visibility_to_hdf5(cbl_vis, output, "vis/%d/%d" % (a1, a2))
