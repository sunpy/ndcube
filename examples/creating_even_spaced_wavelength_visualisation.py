"""
================================================
Combining a celestial WCS with a wavelength axis
================================================

The goal of this example is to construct a spectral-image cube of AIA images at different wavelength.

This will showcase how to add an arbitrarily spaced wavelength dimension
to a celestial WCS.
"""
import matplotlib.pyplot as plt

import astropy.units as u

import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a

from ndcube import NDCube
from ndcube.extra_coords import QuantityTableCoordinate
from ndcube.wcs.wrappers import CompoundLowLevelWCS

#############################################################################
# Use SunPy's Fido to search for the desired data and then read the data files with sunpy Map.
# `sequence=True` causes a sequence of maps to be returned, one for each image file.
aia_files = Fido.fetch(Fido.search(a.Time("2023/01/01", "2023/01/01 00:00:11"), a.Instrument.aia))
maps = sunpy.map.Map(aia_files, sequence=True)

#############################################################################
# Sort the maps in the sequence in order of wavelength.
maps.maps = list(sorted(maps.maps, key=lambda m: m.wavelength))

#############################################################################
# Create an AstroPy Quantity of the wavelengths of the images and use it to build a
# 1-D lookup-table WCS via `QuantityTableCoordinate`.
# This is then combined with the celestial WCS into a single 3-D WCS
# via CompoundLowLevelWCS.
waves = u.Quantity([m.wavelength for m in maps])
wave_wcs = QuantityTableCoordinate(waves, physical_types="em.wl", names="wavelength").wcs
cube_wcs = CompoundLowLevelWCS(wave_wcs, maps[0].wcs)  # We have put the WCS wavelength axis first. Therefore, the last axis of the associated spectral-image data array will have to be last.

#############################################################################
# Combine the new 3-D WCS with the stack of AIA images via NDCube.
my_cube = NDCube(maps.as_array(), wcs=cube_wcs)
# Produce an interactive plot of the spectral-image stack.
my_cube.plot(plot_axes=['y', 'x', None])
plt.show()
