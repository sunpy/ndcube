"""
=================
Creating an even spaced wavelength dimension in AIA
=================

This example shows you how to create an illustration of the changing intensity across a
band of wavelength observations. The result is an interactive figure of the observation.

The example uses `sunpy.Fido` to retrieve the data and `sunpy.Map` handle the data and its meta.
`NDCube` is used to stack the images into a cube in order the adjustment through wavelength.

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
# Use SunPy's Fido to search for the desired data and then load it into map to utilise the benefits of WCS.
# `sequence=True` allows a series of maps to be handled.
aia_files = Fido.fetch(Fido.search(a.Time("2023/01/01", "2023/01/01 00:00:11"), a.Instrument.aia))
maps = sunpy.map.Map(aia_files, sequence=True)

#############################################################################
# Access the map sequenece on the map object and ensure they are sorted by wavelength.
maps.maps = list(sorted(maps.maps, key=lambda m: m.wavelength))

#############################################################################
# Utilise the AstroPy Quantity functionality to create a set of quantites matching the wavelengths of the observations.
# These are then utilised to create a lookup-table type object via `QuantityTableCoordinate`, creating a corresponding wavelength
# for each pixel in the wavelength (think depth) dimension. Finally, this is then passed to CompoundLowLevelWCS, which constructs the final cube orientation.
waves = u.Quantity([m.wavelength for m in maps])
wave_wcs = QuantityTableCoordinate(waves, physical_types="em.wl", names="wavelength").wcs
cube_wcs = CompoundLowLevelWCS(wave_wcs, maps[0].wcs)

#############################################################################
# Lets plot some stuff!
my_cube = NDCube(maps.as_array(), wcs=cube_wcs)
my_cube.plot(plot_axes=['y', 'x', None])
plt.show()
