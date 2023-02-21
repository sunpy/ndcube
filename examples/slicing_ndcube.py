"""
=======================================================
Examples of cropping an NDCube
=======================================================

One of the powerful aspects of having coordinate-aware data stored as an
`~ndcube.NDCube` is the ability to crop and slice the data and coordinates in a
standarised and easy way.

For example, there may be a region of interest you would like to crop out along a certain dimension
of your cube. In this example, this methods to slice an `~ndcube.NDCube` are illustrated.

"""
import astropy.wcs
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames

from ndcube import NDCube

##############################################################################
# Lets begin by creating an example `~ndcube.NDCube` object.
# For this case, we'll generate an `~ndcube.NDCube` that consists of 3 dimensions
# space-space-wavelength. This is analagous to say an observation that takes images in multiple
# wavelengths.
# Lets first define a 3-D numpy array, and then define the WCS information that describe the
# data. We'll just create an array of random numbers, and a WCS which consists of the coordinate information
# which in this case will be in Helioprojective (i.e. an observation of the Sun) in latitute and longitude,
# and over several wavelengths in the range of 10.2 - 11 angstrom.

# Define the data of random numbers. Here the spatial dimensions are (45, 45) and the wavelength dimension is 5.
data = np.random.rand(5, 45, 45)
# Define the WCS
wcs = astropy.wcs.WCS(naxis=3)
wcs.wcs.ctype = 'HPLT-TAN', 'HPLN-TAN', "WAVE"
wcs.wcs.cunit = 'arcsec', 'arcsec', 'Angstrom'
wcs.wcs.cdelt = 10, 10, 0.2
wcs.wcs.crpix = 2, 2, 0
wcs.wcs.crval = 1, 1, 10
wcs.wcs.cname = 'HPC lat', 'HPC lon', 'wavelength'
# Instantiate the `~ndcube.NDCube`
example_cube = NDCube(data, wcs=wcs)

##############################################################################
# So we now have created an `~ndcube.NDCube` named `example_cube`.
# You may have noticed that the order of the WCS is reversed to the array order - this
# is normal convention, and something to remember throughout.
# Now lets first inspect the cube.

##############################################################################
# Here we can inspect the cube by plotting it
example_cube.plot()

##############################################################################
# We can also inspect the dimesions of the cube:
example_cube.dimensions

##############################################################################
# We can also inspect the axis world coordinates of the pixels:
example_cube.axis_world_coords_values()


##############################################################################
# Cropping the cube
# -------------------
# An `~ndcube.NDCube` can be sliced and cropped both by array indexing (similar to the way a numpy
# array in indexed) or by real world coordinates. Lets begin by cropping by index.

##############################################################################
# Cropping by pixel index
# -------------------
# Lets say crop the `example_cube` so that we slice out only one wavelength, we can do that
# by indexing as such
sliced_cube = example_cube[1, :, :]
# here we can see we are left with a 2-D cube which is an image at one wavelength.
sliced_cube.dimensions

# We can also index a region of interest of the cube at a particular wavelength.
# Again note that we are cropping here based on the `pixel` index rather then than coordinate
# real world value

sliced_cube = example_cube[1, 10:20, 20:40]
sliced_cube.dimensions

# now we can inspect the sliced cube, and see its now a smaller region of interest.
sliced_cube.plot()

##############################################################################
# Cropping cube using world coordinate values using `ndcube.NDCube.crop()`
# -------------------
# Now in many use cases its more useful to crop a cube to a region of interest based
# on some specific real world coordinates such as coordinates of points in space or over some spectral
# range. This is achieved by the `ndcube.NDCube.crop()` method which takes high-level astropy coordinate objects,
# such as an `astropy.coordinate.SkyCoord` object. The `.crop()` returns the smallest cube in pixel space that contains all
# the passed points.

##############################################################################
# Lets first define some points over which to crop the `example_cube`. The points are
# defined as iterables of size two that define a point in the cube.
# Lets first define two points in space (lat and long) we want to crop but keeping all wavelengths:
point1 = [SkyCoord(0*u.arcsec, 0*u.arcsec, frame=frames.Helioprojective), None]
point2 = [SkyCoord(200*u.arcsec, 100*u.arcsec, frame=frames.Helioprojective), None]

cropped_cube = example_cube.crop(point1, point2)
cropped_cube.dimesions

cropped_cube.plot()

##############################################################################
# Now lets say we also want to crop over at a  particular wavelength.
# Lets define a new point, and then include them to be passed in the `.crop` method.
point3 = [None, SpectralCoord(10.2*u.angstrom)]

cropped_cube = example_cube.crop(point1, point2, point3)
cropped_cube.dimesions

cropped_cube.plot()

# Now lets say we also want to crop over at a  particular wavelength range.
# Lets define another new point, and then include them to be passed in the `.crop` method.
point4 = [None, SpectralCoord(10.6*u.angstrom)]

cropped_cube = example_cube.crop(point1, point2, point3, point4)
cropped_cube.dimesions

##############################################################################
# Here we can just see how powerful this can be to easily crop over different world coordinates.
