"""
===============================================
How to create an GWCS from quantities and times
===============================================

This example shows how to create a GWCS from astropy quantities.
"""
import numpy as np
from matplotlib import pyplot as plt

import astropy.units as u
from astropy.time import Time

from ndcube import NDCube
from ndcube.extra_coords import QuantityTableCoordinate, TimeTableCoordinate

##############################################################################
# We aim to create coordinates that are focused around time and energies using astropy quantities.

energy = np.arange(10) * u.keV
time = Time('2020-01-01 00:00:00') + np.arange(9)*u.s

##############################################################################
# Then, we need to turn these into lookup tables using
# `~ndcube.extra_coords.table_coord.QuantityTableCoordinate` and
# `~ndcube.extra_coords.table_coord.TimeTableCoordinate` to create table coordinates.

energy_coord = QuantityTableCoordinate(energy, names='energy', physical_types='em.energy')
print(energy_coord)

time_coord = TimeTableCoordinate(time, names='time', physical_types='time')
print(time_coord)

##############################################################################
# Now we need to combine table coordinates created above and extract the ``.wcs`` from the result.

wcs = (time_coord & energy_coord).wcs
print(wcs)

##############################################################################
# Now, we have all the pieces required to construct a `~ndcube.NDCube` with this data and the GWCS we just created.

data = np.random.rand(len(time), len(energy))
cube = NDCube(data=data, wcs=wcs)
print(cube)

##############################################################################
# Finally, we will plot the cube.

cube.plot()

plt.show()
