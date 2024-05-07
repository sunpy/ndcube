"""
=======================================================
How to create an NDCube from data stored in a FITS file
=======================================================

This example shows how you load in data from a FITS file to create an `~ndcube.NDCube`.
Here we will use an example of a single image.
"""

import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS

from ndcube import NDCube

##############################################################################
# We first download the example file that we will use here to show how to create an
# `~ndcube.NDCube` from data stored in a FITS file.
# Here we are using an example file from ``astropy``.

image_file = get_pkg_data_filename('tutorials/FITS-images/HorseHead.fits')

###########################################################################
# Lets extract the image data and the header information from the FITS file.
# This can be achieved by using the functionality within `astropy.io.fits`.
# In this file the image information is located in the Primary HDU (extension 0).

image_data = fits.getdata(image_file)
image_header = fits.getheader(image_file)

##########################################################################
# To create an `~ndcube.NDCube` object, we need both the data array and a
# WCS object (e.g. an `~astropy.wcs.WCS`). Here the data WCS information is
# within the header, which we can pass to :class:`~astropy.wcs.WCS` to create a WCS object.

example_ndcube = NDCube(image_data, WCS(image_header))

##########################################################################
# Now we have created an `~ndcube.NDCube` from this data.
# We can inspect the `~ndcube.NDCube`, such as the WCS.

print(example_ndcube.wcs)

##########################################################################
# and we can also inspect the dimensions.

print(example_ndcube.shape)

##########################################################################
# We can also quickly visualize the data using the :meth:`~ndcube.NDCube.plot` method.

example_ndcube.plot()

plt.show()
