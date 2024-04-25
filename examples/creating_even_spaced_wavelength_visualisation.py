"""
================================================
Combining a celestial WCS with a wavelength axis
================================================

The goal of this example is to construct a spectral-image cube of AIA images at different wavelength.

This will showcase how to add an arbitrarily spaced wavelength dimension
to a celestial WCS.
"""
import astropy.units as u
import matplotlib.pyplot as plt
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
sequence_of_maps.maps = list(sorted(sequence_of_maps.maps, key=lambda m: m.wavelength))

#############################################################################
# Create an AstroPy Quantity of the wavelengths of the images and use it to build a
# 1-D lookup-table WCS via `QuantityTableCoordinate`.
# This is then combined with the celestial WCS into a single 3-D WCS
# via CompoundLowLevelWCS.

waves = u.Quantity([m.wavelength for m in maps])
wave_wcs = QuantityTableCoordinate(waves, physical_types="em.wl", names="wavelength").wcs
cube_wcs = CompoundLowLevelWCS(wave_wcs, maps[0].wcs)
# In the above WCS, we have put the WCS wavelength axis first. Therefore, the last axis
# of the associated spectral-image data array will have to be last.

#############################################################################
# Combine the new 3-D WCS with the stack of AIA images via NDCube.

my_cube = NDCube(maps.as_array(), wcs=cube_wcs)
# Produce an interactive plot of the spectral-image stack.
my_cube.plot(plot_axes=['y', 'x', None])

plt.show()
