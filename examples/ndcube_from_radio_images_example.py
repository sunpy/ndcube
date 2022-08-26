"""
=================
Creating an NDCube from solar radio interferometric images.
=================

How to make an NDCube from solar radio interferometric images.

The example uses `dkist` https://docs.dkist.nso.edu/projects/python-tools/en/latest/ and `gwcs`
to create a single wcs object that takes into account the changing crval of 
solar radio images using `~dkist.wcs.models.VaryingCelestialTransform`. 
The following example uses data from the NenuFAR radio telescope
https://nenufar.obs-nancay.fr/en/homepage-en/
"""
import glob

import numpy as np

from multiprocessing import Pool

from astropy.modeling import models
from astropy import coordinates as coord
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u
from astropy.io import fits
from astropy.time import Time, TimeDelta
from gwcs import wcs
from gwcs import coordinate_frames as cf
from ndcube import NDCube
from sunpy.coordinates.frames import Helioprojective
from sunpy.coordinates import sun

from dkist.wcs.models import VaryingCelestialTransform

##############################################################################
# We first need to load some data. These exist on a machine somewhere and 
# I don't know how to share them. 
# Here we take every image at a single frequency

ims = glob.glob("/data/mpearse/radio_images/SB158/*image.fits")
ims.sort()

def get_data(file):
    with fits.open(file) as hdu:
        header = hdu[0].header
        data = np.squeeze(hdu[0].data)
    
    obstime = Time(header['date-obs']) # observational time
    freq = header['crval3']*u.Hz # frequency of observation
    wavelength = freq.to(u.m, equivalencies=u.spectral())
    #data from Jy/beam to T_b
    beam_semi_major_axis = 0.5*header['BMAJ']*u.deg #nothing in fits to suggest degrees, you just have to know
    beam_semi_minor_axis = 0.5*header['BMIN']*u.deg
    beam_area = ((np.pi*beam_semi_major_axis*beam_semi_minor_axis)/(4*np.log(2)))
    data = data *(u.Jy/beam_area)
    equiv = u.brightness_temperature(freq)
    data = data.to(u.K, equivalencies=equiv)
    
    return data, obstime

with Pool() as pool:
    datatime = pool.map(get_data, ims)

data = np.array([d[0] for d in datatime])
time = Time(np.array([t[1] for t in datatime]))
###########################################################################
# The reference coordinate for radio observations typically changes as the
# telescope pointing is updated to stay somewhat centred on the sun.
# Thus, we will need create look up tables for the different reference 
# coordinates and corresponding rotation matrices. We will also need the
# location of the NenuFAR array.

nenufar_ITRF = np.array((4323915,165533.67,4670321.7))
array_loc = EarthLocation.from_geocentric(*nenufar_ITRF, u.m)

def get_lookup_tables(file):
    with fits.open(file) as hdu:
        header = hdu[0].header
    obstime = Time(header['DATE-OBS'])

    array_gcrs = SkyCoord(array_loc.get_gcrs(obstime))

    # reference coordinate from FITS file
    reference_coord = SkyCoord(header['CRVAL1']*u.deg, header['CRVAL2']*u.deg,
                           frame='gcrs',
                           obstime=obstime,
                           obsgeoloc=array_gcrs.cartesian,
                           obsgeovel=array_gcrs.velocity.to_cartesian(),
                           distance=array_gcrs.hcrs.distance,
                           equinox='J2000')

    reference_coord_arcsec = reference_coord.transform_to(Helioprojective(observer=array_gcrs))
    crval = [reference_coord_arcsec.Tx.arcsec, reference_coord_arcsec.Ty.arcsec]
    P = sun.P(obstime)
    pc = np.array([[np.cos(-P), -np.sin(-P)],
                   [np.sin(-P), np.cos(P)]])
    
    return crval, pc

with Pool() as pool:
    lookup_tables = pool.map(get_lookup_tables, ims)
    
crval_table = np.array([lt[0] for lt in lookup_tables])
crval_table = crval_table*u.arcsec

pc_table = np.array([lt[1] for lt in lookup_tables])
pc_table = pc_table*u.deg
##########################################################################
# We now define the `~dkist.wcs.models.VaryingCelestialTransform` to create
# our wcs object. WARNING this is very finnicky and will likely change in the
# future. It's convenient here to get some meta data from the first image

with fits.open(ims[0]) as hdu:
    header = hdu[0].header
cdelt1 = (np.abs(header['CDELT1'])*u.deg).to(u.arcsec)/u.pix
cdelt2 = (np.abs(header['CDELT2'])*u.deg).to(u.arcsec)/u.pix

vct = VaryingCelestialTransform(crval_table = crval_table,
                                pc_table = pc_table,
                                crpix = [header['CRPIX1']*u.pix, header['CRPIX2']*u.pix],
                                cdelt = [cdelt1, cdelt2],
                                lon_pole=180*u.deg,
                               )

##########################################################################
# We now follow from 
# https://gwcs.readthedocs.io/en/latest/#a-step-by-step-example-of-constructing-an-imaging-gwcs-object
# to construct our wcs object that will be used by `NDCube`

detector_frame = cf.Frame2D(name="detector", axes_names=("x", "y"),
                            unit=(u.pix, u.pix))

time_frame = cf.TemporalFrame(time, axes_names=('time'), axes_order=(2,))

detector_time_frame = cf.CompositeFrame([time_frame, detector_frame])

sky_frame = cf.CelestialFrame(reference_frame=Helioprojective, name='helioprojective',
                              axes_names=['pos.helioprojective.lat', 'pos.helioprojective.lon'],
                              unit=(u.arcsec, u.arcsec))


pipeline = [(detector_time_frame, vct),
            (sky_frame, None)
           ]

wcsobj = wcs.WCS(pipeline)

##########################################################################
# Finally we create our 'NDCube' object
cube = NDCube(data, wcsobj, unit=u.K)

##########################################################################
# Hooray!