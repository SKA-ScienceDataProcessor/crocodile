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

# Parse arguments
parser = argparse.ArgumentParser(description='Collect and coalesce visibilities for a baseline from OSKAR files.')
parser.add_argument('input', metavar='files', type=argparse.FileType('r'),
                    nargs='*',
                    help='input files')
parser.add_argument('--ra', dest='ra', metavar='RA', type=float,
                    help='phase centre right ascension (degrees)')
parser.add_argument('--dec', dest='dec', metavar='DC', type=float,
                    help='phase centre declination (degrees)')
parser.add_argument('--bl', dest='baselines', metavar='DC', type=str, action='append',
                    help='baselines to gather (antenna pairs, as in "A1-A2")')
parser.add_argument('--theta', dest='theta', metavar='theta', type=float, required=True,
                    help='Field of view to average for (used to compute cell size)')
parser.add_argument('--step', dest='step', metavar='step', type=float, required=True,
                    help='Step length to average to [cells]')
parser.add_argument('--max-coal-t', dest='maxt', metavar='maxt', type=float, required=True,
                    help='Maximum averaging in time [s]')
parser.add_argument('--max-coal-f', dest='maxf', metavar='maxf', type=float, required=True,
                    help='Maximum averaging in frequency [MHz]')
args = parser.parse_args()

# Read oskar file
print("Reading", args.input[0].name, "...")
vis = import_visibility_from_oskar(args.input[0].name)

# Parse baselines
baselines = list(map(lambda b: list(map(int, b.split('-'))), args.baselines))
assert numpy.all(map(lambda b: len(b) == 2, baselines))

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

# Loop through files
bl_table = {}
for i, inp in enumerate(args.input):

    # Read. First one will already be loaded
    if i > 0:
        print("Reading", inp.name, "...")
        vis = import_visibility_from_oskar(inp.name)

    # Phase-rotate
    print("Rotating phase...")
    vis = phaserotate_visibility(vis, new_pc)
    uvw = vis.uvw_lambda(0)
    print("u range:     %.2f - %.2f lambda" % (numpy.min(uvw[:,0]), numpy.max(uvw[:,0])))
    print("v range:     %.2f - %.2f lambda" % (numpy.min(uvw[:,1]), numpy.max(uvw[:,1])))
    print("w range:     %.2f - %.2f lambda" % (numpy.min(uvw[:,2]), numpy.max(uvw[:,2])))

    print("Grouping...")
    data_by_antenna = vis.data.group_by(['antenna1', 'antenna2'])

    # Loop through baselines
    print("Collecting...", end="", flush=True)
    for j, key in enumerate(data_by_antenna.groups.keys):

        # Interested in this baseline?
        a1 = key['antenna1']
        a2 = key['antenna2']
        if not [a1, a2] in baselines:
            continue

        # Select from table
        print(" %d-%d" % (a1, a2), end="", flush=True)
        bl_vis = Visibility(vis, data=data_by_antenna.groups[j])

        # Add to/set collection table
        if (a1, a2) in bl_table:
            bl_table[(a1,a2)] = concatenate_visibility_frequencies(bl_table[(a1,a2)], bl_vis)
        else:
            bl_table[(a1,a2)] = bl_vis

        # Export?
        if i % 10 == 0:
            export_visibility_to_fits(bl_table[(a1,a2)], "v%d-%d.fits" % (a1, a2))
    print(" done")

# Finally, coalesce data and write
for (a1, a2), vis in bl_table.items():
    export_visibility_to_fits(vis, "v%d-%d.fits" % (a1, a2))

for (a1, a2), vis in bl_table.items():

    # Sort by time
    vis.data.sort('time')

    # Determine UVW distance in time at maximum frequency
    max_chan = numpy.argmax(vis.frequency)
    uvw = vis.uvw_lambda(max_chan)
    distance_t = numpy.linalg.norm(uvw[0] - uvw[-1]) * args.theta

    # ... same for frequency
    min_chan = numpy.argmin(vis.frequency)
    uvw2 = vis.uvw_lambda(min_chan)
    distance_f = numpy.linalg.norm(uvw[0] - uvw2[0]) * args.theta

    print("Baseline %d-%d: Distance %f x %f" % (a1, a2, distance_t, distance_f))

    # Decide amount of averaging
    max_time_coal = args.maxt / (24 * 3600 * (numpy.max(vis.time) - numpy.min(vis.time)) / vis.ntime)
    time_coalesce = min(max_time_coal, vis.ntime, vis.ntime / (distance_t / args.step))
    max_freq_coal = args.maxf * 1000000 / ((numpy.max(vis.frequency) - numpy.min(vis.frequency)) / vis.nchan)
    frequency_coalesce = min(max_freq_coal, vis.nchan, vis.nchan / (distance_f / args.step))

    print(" -> Coalescing %d x %d" % (int(time_coalesce), int(frequency_coalesce)))
    cvis = coalesce_visibility(vis, int(time_coalesce), int(frequency_coalesce))
    export_visibility_to_fits(cvis, "c%d-%d.fits" % (a1, a2))
