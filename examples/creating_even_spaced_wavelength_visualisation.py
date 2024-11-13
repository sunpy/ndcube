"""
================================================
Combining a celestial WCS with a wavelength axis
================================================

The goal of this example is to construct a spectral-image cube of AIA images at different wavelengths.

This will showcase how to add an arbitrarily spaced wavelength dimension
to a celestial WCS.
"""
import matplotlib.pyplot as plt

import astropy.units as u

import sunpy.data.sample
import sunpy.map

from ndcube import NDCube
from ndcube.extra_coords import QuantityTableCoordinate
from ndcube.wcs.wrappers import CompoundLowLevelWCS

#############################################################################
# We will use the sample data that ``sunpy`` provides to construct a sequence of AIA
# image files for different wavelengths using `sunpy.map.Map`.

aia_files = [sunpy.data.sample.AIA_094_IMAGE,
             sunpy.data.sample.AIA_131_IMAGE,
             sunpy.data.sample.AIA_171_IMAGE,
             sunpy.data.sample.AIA_193_IMAGE,
             sunpy.data.sample.AIA_211_IMAGE,
             sunpy.data.sample.AIA_304_IMAGE,
             sunpy.data.sample.AIA_335_IMAGE,
             sunpy.data.sample.AIA_1600_IMAGE]
# `sequence=True` causes a sequence of maps to be returned, one for each image file.
sequence_of_maps = sunpy.map.Map(aia_files, sequence=True)
# Sort the maps in the sequence in order of wavelength.
sequence_of_maps.maps = sorted(sequence_of_maps.maps, key=lambda m: m.wavelength)

#############################################################################
# Using an `astropy.units.Quantity` of the wavelengths of the images, we can construct
# a 1D lookup-table WCS via `.QuantityTableCoordinate`.
# This is then combined with the celestial WCS into a single 3D WCS via `.CompoundLowLevelWCS`.

wavelengths = u.Quantity([m.wavelength for m in sequence_of_maps])
wavelengths_wcs = QuantityTableCoordinate(wavelengths, physical_types="em.wl", names="wavelength").wcs
cube_wcs = CompoundLowLevelWCS(wavelengths_wcs, sequence_of_maps[0].wcs)

#############################################################################
# Combine the new 3D WCS with the stack of AIA images using `ndcube.NDCube`.
# Note that because we set the wavelength to the first axis
# in the WCS (cube_wcs), the final data cube is stacked such
# that wavelength corresponds to the array axis which is last.
# This is due to the convention that WCS axis ordering is reversed
# compared to data array axis ordering.

my_cube = NDCube(sequence_of_maps.as_array(), wcs=cube_wcs)
# Produce an interactive plot of the spectral-image stack.
my_cube.plot(plot_axes=['y', 'x', None])

plt.show()
