"""
=================
Creating an NDCube from solar radio interferometric images.
=================

How to make an NDCube from solar radio interferometric images.

The example uses `dkist`
https://docs.dkist.nso.edu/projects/python-tools/en/latest/ and `gwcs`
to create a single wcs object that takes into account the changing crval of
solar radio images using `~dkist.wcs.models.VaryingCelestialTransform`.
The following example uses data from the NenuFAR radio telescope
https://nenufar.obs-nancay.fr/en/homepage-en/
"""
import numpy as np
from astropy import units as u
from astropy.modeling import models
from astropy.time import Time, TimeDelta
from dkist.wcs.models import CoupledCompoundModel, VaryingCelestialTransform
from gwcs import coordinate_frames as cf
from gwcs import wcs
from sunpy.coordinates.frames import Helioprojective

from ndcube import NDCube

##############################################################################
# We first need to load some data. We'll create some fake data for this
# example of shape TIME, SPACE, SPACE.
# We will also need the corresponding time array.

data = np.random.rand(1440, 1024, 1024)
t_arr = np.arange(1440)
time_array = Time("2022-09-01T12:00:00") + TimeDelta(t_arr, format='sec')
###########################################################################
# The reference coordinate for radio observations typically changes as the
# telescope pointing is updated to stay somewhat centred on the sun.
# Thus, we will need create look up tables for the different reference
# coordinates and corresponding rotation matrices.
# Here we generate these randomly to have shape TIME, 2 for the reference
# coordinates and TIME, 2, 2 for the rotation matrices.

crval_table = np.random.rand(1440, 2)
crval_table = crval_table*u.arcsec

pc_table = np.random.rand(1440, 2, 2)
pc_table = pc_table*u.deg
##########################################################################
# Let's also define the reference pixel and the pixel scale
cdelt1 = 88.8*(u.arcsec/u.pix)
cdelt2 = 88.8*(u.arcsec/u.pix)
crpix1 = 513*u.pix
crpix2 = 513*u.pix

##########################################################################
# We now need a model to convert from a given pixel (x, y)
# at a given time sample (z) to the relevant
# world coordinate (lat, lon, time).
# This requires `dkist.wcs.models.VaryingCelestialTransform` to create
# a wcs object for the changing reference coordinates/rotation matrices
# and also `astropy.modeling.models.Linear1D` to map from z to time.
# These are combined using `dkist.wcs.models.CoupledCompoundModel`.
# WARNING `~dkist.wcs.models.VaryingCelestialTransform`
# and `~dkist.wcs.models.CoupledCompoundModel`
# are likely change in the future and thus this method may need to be updated.


vct = VaryingCelestialTransform(crval_table=crval_table,
                                pc_table=pc_table,
                                crpix=[crpix1, crpix2],
                                cdelt=[cdelt1, cdelt2],
                                lon_pole=180*u.deg,
                                )
dt = time_array[-1] - time_array[0]
m = dt.sec/len(time_array)

tmodel = models.Linear1D(slope=m*u.s/u.pix, intercept=0*u.s)

ccm = CoupledCompoundModel('&', vct, tmodel)
##########################################################################
# We now follow from
# https://gwcs.readthedocs.io/en/latest/#a-step-by-step-example-of-constructing-an-imaging-gwcs-object
# to construct our wcs object that will be used by `NDCube`

timepix_frame = cf.CoordinateFrame(naxes=1,
                                   axes_type="TIME",
                                   axes_order=(2,),
                                   name="detector_time",
                                   axes_names="z",
                                   unit=(u.pix))


# something funny here if unit = (u.pix, u.pix)
# coordinate is returned as pix^2
detector_frame = cf.Frame2D(name="detector",
                            axes_names=("x", "y"),
                            unit=(u.pix, u.pix))

detector_time_frame = cf.CompositeFrame([detector_frame, timepix_frame],
                                        name="detector_time_frame")

time_frame = cf.TemporalFrame(time_array[0],
                              axes_names=('time'),
                              axes_order=(2,),
                              unit=(u.s))

sky_frame = cf.CelestialFrame(reference_frame=Helioprojective,
                              name='helioprojective',
                              axes_names=['pos.helioprojective.lat',
                                          'pos.helioprojective.lon'],
                              unit=(u.arcsec, u.arcsec))

sky_time_frame = cf.CompositeFrame([sky_frame, time_frame],
                                   name="sky_time_frame")

pipeline = [(detector_time_frame, ccm),
            (sky_time_frame, None)
            ]

wcsobj = wcs.WCS(pipeline)

##########################################################################
# Finally we create our 'NDCube' object

cube = NDCube(data, wcsobj, unit=u.K)
print(cube)

##########################################################################
# Hooray!
