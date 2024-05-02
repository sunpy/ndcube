"""
===============================================
How to create an gWCS from quantities and times
===============================================

This example shows to create a gWCS from quantities in this example time and energy.

"""

import numpy as np

import astropy.units as u
from astropy.time import Time

from ndcube import ExtraCoords, NDCube
from ndcube.extra_coords import QuantityTableCoordinate, TimeTableCoordinate

##############################################################################
# First we create our coordinate arrays.

energy = np.arange(10) * u.keV
time = Time('2020-01-01 00:00:00') + np.arange(9)*u.s

##############################################################################
# Then we use use
# `~ndcube.extra_coords.table_coord.QuantityTableCoordinate` and
# `~ndcube.extra_coords.table_coord.TimeTableCoordinate` to create table coordinates.

energy_coord = QuantityTableCoordinate(energy, names='energy', physical_types='em.energy')
energy_coord

time_coord = TimeTableCoordinate(time, names='time', physical_types='time')
time_coord

##############################################################################
# Create new `~ndcube.extra_coords.extra_coords.ExtraCoords` instance and add the previously created table coordinates
# and extract the gWCS.

extra_coords = ExtraCoords()
extra_coords.add('energy', array_dimension=1, lookup_table=energy_coord)
extra_coords.add('time', array_dimension=0, lookup_table=time_coord)
wcs = extra_coords.wcs
wcs

##############################################################################
# Finally we make a data array and create new `~ndcube.NDCube` with this data and the gWCS we just created.

data = np.random.rand(len(time), len(energy))
cube = NDCube(data=data, wcs=wcs)
cube
