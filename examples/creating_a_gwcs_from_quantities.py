"""
===============================================
How to create an gWCS from quantities and times
===============================================

This example shows to create a gWCS from quantities in this example time and energy.

"""

import numpy as np
from matplotlib import pyplot as plt

import astropy.units as u
from astropy.time import Time

from ndcube import NDCube
from ndcube.extra_coords import QuantityTableCoordinate, TimeTableCoordinate
from ndcube.wcs.wrappers import CompoundLowLevelWCS

##############################################################################
# First we create our coordinate arrays.

energy = np.arange(10) * u.keV
time = Time('2020-01-01 00:00:00') + np.arange(9)*u.s

##############################################################################
# Then we use use
# `~ndcube.extra_coords.table_coord.QuantityTableCoordinate` and
# `~ndcube.extra_coords.table_coord.TimeTableCoordinate` to create table coordinates.

energy_coord = QuantityTableCoordinate(energy, names='energy', physical_types='em.energy')
print(energy_coord)

time_coord = TimeTableCoordinate(time, names='time', physical_types='time')
print(time_coord)

##############################################################################
# Create new `~ndcube.wcs.wrappers.compound_wcs.CompoundLowLevelWCS` instance using the previously created table
# coordinates WCSs (note the ordering).

wcs = CompoundLowLevelWCS(time_coord.wcs, energy_coord.wcs, )
print(wcs)

##############################################################################
# Finally we make a data array and create new `~ndcube.NDCube` with this data and the gWCS we just created.

data = np.random.rand(len(time), len(energy))
cube = NDCube(data=data, wcs=wcs)
print(cube)

##############################################################################
# Make a plot

cube.plot()
plt.show()
