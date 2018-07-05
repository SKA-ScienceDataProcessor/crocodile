# Tim Cornwell <realtimcornwell@gmail.com>
#
# Definition of structures needed by the function interface. These are mostly
# subclasses of astropy classes.
#
import numpy as numpy

from astropy.table import Table

import os

import numpy

import astropy.units as units
from astropy.coordinates import SkyCoord, ICRS, EarthLocation
from astropy.table import Table, Column, vstack
from astropy.wcs import WCS
from astropy.io.fits import HDUList, PrimaryHDU, BinTableHDU, table_to_hdu
import astropy.io.fits as fits
from astropy import units

import h5py

from crocodile.simulate import *

from arl.data_models import *
from arl.parameters import crocodile_path

from util.read_oskar_vis import OskarVis

import logging
log = logging.getLogger(__name__)

def filter_configuration(fc: Configuration, params={}):
    """ Filter a configuration e.g. remove certain antennas

    :param fc:
    :type Configuration:
    :param params: Dictionary containing parameters
    :returns: Configuration
    """
    log.error("filter_configuration: No filter implemented yet")
    return fc


def create_configuration_from_array(antxyz: numpy.array, name: str = None, location: EarthLocation = None,
                                    mount: str = 'alt-az', names: str = '%d', meta: dict = None, params={}):
    """ Define from parts

    :param name:
    :param antxyz: locations of antennas numpy.array[...,3]
    :type numpy.array:
    :param location: Location of array centre (reference for antenna locations)
    :type EarthLocation:
    :param mount: Mount type e.g. 'altaz'
    :type str:
    :param names: Generator for names e.g. 'SKA1_MID%d'
    :type generator:
    :type meta:
    :type dict:
    :returns: Configuration
    """
    fc = Configuration()
    assert len(antxyz) == 2, "Antenna array has wrong shape"
    fc.data = Table(data=[names, antxyz, mount], names=["names", "xyz", "mount"], meta=meta)
    fc.location = location
    return fc


def create_configuration_from_file(antfile: str, name: str = None, location: EarthLocation = None, mount: str = 'altaz',
                                   names: str = "%d", frame: str = 'local',
                                   meta: dict = None,
                                   params={}):
    """ Define from a file

    :param antfile: Antenna file name
    :type str:
    :param name: Name of array e.g. 'LOWBD2'
    :type str:
    :param location:
    :type EarthLocation:
    :param mount: mount type: 'altaz', 'xy'
    :type str:
    :param frame: 'local' | 'global'
    :type str:
    :param meta: Any meta info
    :type dict:
    :returns: Configuration
    """
    fc = Configuration()
    fc.name = name
    fc.location = location
    antxyz = numpy.genfromtxt(antfile, delimiter=",")
    assert antxyz.shape[1] == 3, ("Antenna array has wrong shape %s" % antxyz.shape)
    nants = antxyz.shape[0]
    if frame == 'local':
        latitude = location.geodetic[1].to(units.rad).value
        antxyz = xyz_at_latitude(antxyz, latitude)
    xyz = Column(antxyz, name="xyz")

    anames = [names % ant for ant in range(nants)]
    mounts = Column(numpy.repeat(mount, nants), name="mount")
    fc.data = Table(data=[anames, xyz, mounts], names=["names", "xyz", "mount"], meta=meta)
    fc.frame = frame
    return fc


def create_LOFAR_configuration(antfile: str, meta: dict = None,
                               params={}):
    """ Define from the LOFAR configuration file

    :param antfile:
    :type str:
    :param name:
    :type str:
    :param meta:
    :type dict:
    :param params: Dictionary containing parameters
    :returns: Configuration
    """
    fc = Configuration()
    antxyz = numpy.genfromtxt(antfile, skip_header=2, usecols=[1, 2, 3], delimiter=",")
    nants = antxyz.shape[0]
    assert antxyz.shape[1] == 3, "Antenna array has wrong shape %s" % antxyz.shape
    anames = numpy.genfromtxt(antfile, dtype='str', skip_header=2, usecols=[0], delimiter=",")
    mounts = Column(numpy.repeat('XY', nants), name="mount")
    fc.data = Table(data=[anames, antxyz, mounts], names=["names", "xyz", "mount"], meta=meta)
    fc.location = EarthLocation(x=[3826923.9] * units.m, y=[460915.1] * units.m, z=[5064643.2] * units.m)
    return fc


def create_named_configuration(name: str = 'LOWBD2', params={}):
    """ Standard configurations e.g. LOWBD2, MIDBD2

    :param name: name of Configuration LOWBD2, LOWBD1, LOFAR, VLAA
    :type str:
    :returns: Configuration
    """
    
    if name == 'LOWBD2':
        location = EarthLocation(lon="116.4999", lat="-26.7000", height=300.0)
        fc = create_configuration_from_file(antfile=crocodile_path("data/configurations/LOWBD2.csv"),
                                            location=location, mount='xy',
                                            names='LOWBD2_%d', name=name)
    elif name == 'LOWBD1':
        location = EarthLocation(lon="116.4999", lat="-26.7000", height=300.0)
        fc = create_configuration_from_file(antfile=crocodile_path("data/configurations/LOWBD1.csv"),
                                            location=location, mount='xy',
                                            names='LOWBD1_%d', name=name)
    elif name == 'LOFAR':
        fc = create_LOFAR_configuration(antfile=crocodile_path("data/configurations/LOFAR.csv"))
    elif name == 'VLAA':
        location = EarthLocation(lon="-107.6184", lat="34.0784", height=2124.0)
        fc = create_configuration_from_file(antfile=crocodile_path("data/configurations/VLA_A_hor_xyz.csv"),
                                            location=location,
                                            mount='altaz', names='VLA_%d',name=name)
    elif name == 'VLAA_north': # Pseudo-VLAA at north pole
        location = EarthLocation(lon="-107.6184", lat="90.000", height=2124.0)
        fc = create_configuration_from_file(antfile=crocodile_path("data/configurations/VLA_A_hor_xyz.csv"),
                                            location=location,
                                            mount='altaz', names='VLA_%d', name=name)
    elif name == 'LOWBD2_north': # Pseudo-SKA-LOW at north pole
        location = EarthLocation(lon="116.4999", lat="90.000", height=300.0)
        fc = create_configuration_from_file(antfile=crocodile_path("data/configurations/LOWBD2.csv"),
                                            location=location, mount='xy',
                                            names='LOWBD2_%d', name=name)
    else:
        fc = Configuration()
        raise UserWarning("No such Configuration %s" % name)
    return fc

def import_visibility_from_ms(msfile: str, params={}) -> Visibility:
    """ Import a visibility set from a measurement set

    :param msfile: Name of measurement set
    :type str:
    :returns: Visibility
    """
    log.error('test_support.import_visibility_from_ms: not yet implemented')
    return Visibility()


def export_visibility_to_ms(vis: Visibility, msfile: str = None, params={}) -> Visibility:
    """ Export a visibility set to a measurement set

    :param vis: Name of visibility set
    :param Visibility:
    :param msfile: Name of output measurement set
    :type str:
    :returns: Visibility
    """
    log.error('test_support.visibility_from_ms: not yet implemented')

def import_visibility_from_oskar(oskar_file: str, params={}) -> Visibility:
    """ Import a visibility set from an OSKAR visibility file

    :param oskar_file: Name of OSKAR visibility file
    :type str:
    :returns: Visibility
    """

    # Extract data from Oskar file
    oskar_vis = OskarVis(oskar_file)
    ra,dec = oskar_vis.phase_centre()
    a1,a2 = oskar_vis.stations(flatten=True)

    # Make configuration
    location = EarthLocation(lon = oskar_vis.telescope_lon,
                             lat = oskar_vis.telescope_lat,
                             height = oskar_vis.telescope_alt)
    antxyz = numpy.transpose([oskar_vis.station_x,
                              oskar_vis.station_y,
                              oskar_vis.station_z])
    name = oskar_vis.telescope_path
    if name == '':
        name = 'oskar-import'
    config = Configuration(
        name     = name,
        location = location,
        xyz      = antxyz
    )

    # Assume exactly one frequency and polarisation - that is the only
    # supported case right now.
    amps = oskar_vis.amplitudes(flatten=True)
    amps = amps.reshape(list(amps.shape) + [1,1])

    # Construct visibilities
    return Visibility(
        frequency     = [oskar_vis.frequency(i) for i in range(oskar_vis.num_channels)],
        phasecentre   = SkyCoord(frame=ICRS, ra=ra, dec=dec, unit=units.deg),
        configuration = config,
        uvw           = numpy.transpose(oskar_vis.uvw(flatten=True)),
        time          = oskar_vis.times(flatten=True),
        antenna1      = a1,
        antenna2      = a2,
        vis           = amps,
        weight        = numpy.ones(amps.shape))

def configuration_to_hdu(configuration : Configuration) -> BinTableHDU:

    # Convert to HDU
    hdu = table_to_hdu(configuration.data)

    # Save rest of data into header fields (offensively ad-hoc, obviously)
    hdu.header['NAME'] = configuration.name
    hdu.header['LOC_LAT'] = configuration.location.latitude.value
    hdu.header['LOC_LON'] = configuration.location.longitude.value
    hdu.header['LOC_HGT'] = configuration.location.height.value

    return hdu

def visibility_to_hdu(vis: Visibility) -> BinTableHDU:

    # Convert to HDU
    hdu = table_to_hdu(vis.data)

    # Save rest of data into header fields (offensively ad-hoc, obviously)
    pc = vis.phasecentre
    hdu.header['PC_RA'] = pc.ra.to(units.deg).value
    hdu.header['PC_DEC'] = pc.dec.to(units.deg).value
    hdu.header['FREQ'] = ','.join(map(str, vis.frequency))
    hdu.header['CONFIG'] = vis.configuration.name

    return hdu

def export_visibility_to_fits(vis: Visibility, fits_file: str):

    hdu = HDUList([
        PrimaryHDU(),
        configuration_to_hdu(vis.configuration),
        visibility_to_hdu(vis)
    ])
    with open(fits_file, "w") as f:
        hdu.writeto(f, checksum=True)

def export_configuration_to_hdf5(cfg: Configuration, f: h5py.File, path: str = '/'):

    grp = f.create_group(path)
    grp.attrs['type'] = 'Configuration'
    grp.attrs['name'] = cfg.name
    grp.attrs['location'] = [cfg.location.lat.value,
                             cfg.location.lon.value,
                             cfg.location.height.value ]
    for col in cfg.data.columns:
        c = cfg.data[col]
        # Unicode wide strings are not supported, convert to ASCII
        if c.dtype.kind == 'U':
            c = c.astype("S")
        grp.create_dataset(col, data=c)

def export_visibility_to_hdf5(vis: Visibility, f: h5py.File, path: str = '/', maxshape={}):

    grp = f.create_group(path)
    grp.attrs['type'] = 'Visibility'
    grp.attrs['phasecentre'] = [vis.phasecentre.ra.to(units.deg).value,
                                vis.phasecentre.dec.to(units.deg).value]
    if vis.configuration is not None:
        grp.attrs['configuration'] = vis.configuration.name
    freq = numpy.array(vis.frequency)
    grp.create_dataset('frequency', data=freq, maxshape=maxshape.get('frequency'))
    for col in vis.data.columns:
        if col == 'weight' and numpy.all(vis.data[col] == 1.0):
            continue
        grp.create_dataset(col, data=vis.data[col], maxshape=maxshape.get(col))


def import_configuration_from_hdf5(f: h5py.File, path: str = '/'):
    """Import telescope configuration from a HDF5 file.

    :param f: Open HDF5 file to import data from
    :param path: Group name to load data from
    :returns: Configuration object
    """

    # Access group, make sure it is the right type
    grp = f[path]
    assert grp.attrs['type'] == 'Configuration'

    # Read table columns
    table = Table()
    for col in ['names', 'xyz', 'mount']:
        table[col] = numpy.array(grp[col])

    return Configuration(
        name = grp.attrs['name'],
        location = EarthLocation(lat=grp.attrs['location'][0],
                                 lon=grp.attrs['location'][1],
                                 height=grp.attrs['location'][2]),
        data = table
        )


def import_visibility_from_hdf5(f: h5py.File, path: str = '/', cfg: Configuration = None, cols = None):
    """Import visibilities from a HDF5 file.

    :param f: Open HDF5 file to import data from
    :param path: Group name to load data from
    :param cfg: Configuration to set for visibilities
    :returns: Visibilities object
    """

    # Access group, make sure it is the right type
    grp = f[path]
    assert grp.attrs['type'] == 'Visibility'

    # Read table columns
    table = Table()
    if cols is None:
        cols = ['uvw', 'time', 'antenna1', 'antenna2', 'vis', 'weight']
    for col in cols:

        # Default weights to 1 if they are not found
        if not col in grp and col == 'weight':
            table[col] = numpy.ones(table['vis'].shape)
        else:
            table[col] = numpy.array(grp[col])

    return Visibility(
        frequency = grp['frequency'],
        phasecentre = SkyCoord(ra=grp.attrs["phasecentre"][0],
                               dec=grp.attrs["phasecentre"][1],
                               frame=ICRS, unit=units.deg),
        data = table
        )


def import_visibility_baselines_from_hdf5(f: h5py.File, cfg: Configuration = None, cols = None):
    """Import visibilities for multiple baselines from a HDF5 group. This
    means that we load every visibility set contained within the given group.

    :param f: Open HDF5 file to import data from
    :param cfg: Configuration to set for visibilities
    :returns: Visibilities object
    """

    # Collect visibilities
    viss = []
    for name in f:

        # Get group
        grp = f[name]
        if not isinstance(grp, h5py.Group):
            continue

        # Visibilities?
        if 'type' in grp.attrs and grp.attrs.get('type', '') == 'Visibility':
            viss.append(import_visibility_from_hdf5(f, grp.name, cols))
            print('.', end='', flush=True)

        else:
            # Otherwise recurse
            viss += import_visibility_baselines_from_hdf5(grp, cols)

    return viss


def hdu_to_configuration(hdu: BinTableHDU) -> Configuration:

    lat = hdu.header.get('LOC_LAT')
    lon = hdu.header.get('LOC_LON')
    hgt = hdu.header.get('LOC_HGT')
    loc = None
    if not lat is None and not lon is None and not hgt is None:
        loc = EarthLocation(lat=lat, lon=lon, height=hgt)
    return Configuration(
        data = Table(hdu.data),
        name = hdu.header.get('NAME'),
        location = loc
        )

def hdu_to_visibility(hdu: BinTableHDU, configuration: Configuration = None) -> Visibility:

    # Decode phase centre, if any
    pc_ra = hdu.header.get('PC_RA')
    pc_dec = hdu.header.get('PC_DEC')
    pc = None
    if not pc_ra is None and not pc_dec is None:
        pc = SkyCoord(ra=pc_ra, dec=pc_dec, frame=ICRS, unit=units.deg)

    # Check configuration name (additional security?)
    if not configuration is None:
        assert(configuration.name == hdu.header.get('CONFIG'))

    return Visibility(
        data = Table(hdu.data),
        frequency = list(map(float, hdu.header['FREQ'].split(','))),
        phasecentre = pc,
        configuration = configuration
    )

def import_visibility_from_fits(fits_file: str) -> Visibility:

    with fits.open(fits_file) as hdulist:

        # TODO: Check that it is the right kind of file...

        config = hdu_to_configuration(hdulist[1])
        return hdu_to_visibility(hdulist[2], config)

def import_image_from_fits(fitsfile: str):
    """ Read an Image from fits
    
    :param fitsfile:
    :type str:
    :returns: Image
    """
    hdulist = fits.open(crocodile_path(fitsfile))
    fim = Image()
    fim.data = hdulist[0].data
    fim.wcs = WCS(crocodile_path(fitsfile))
    hdulist.close()
    log.debug("import_image_from_fits: Max, min in %s = %.6f, %.6f" % (fitsfile, fim.data.max(), fim.data.min()))
    return fim

def create_test_image(canonical=True):
    """Create a useful test image

    This is the test image M31 widely used in ALMA and other simulations. It is actually part of an Halpha region in
    M31.

    :param canonical: Make the image into a 4 dimensional image
    :returns: Image
    """
    im = import_image_from_fits(crocodile_path("data/models/M31.MOD"))
    if canonical:
        im = replicate_image(im)
    return im
