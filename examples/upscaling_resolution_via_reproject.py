"""
=====================================
Upscaling the resolution of an NDCube
=====================================

This example shows how to increase the resolution of an NDCube by reprojecting to a finer grid.
"""

import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS

from sunpy.data.sample import AIA_171_IMAGE
from sunpy.visualization.colormaps import cm

from ndcube import NDCube

##############################################################################
# We start by creating an NDCube from sample solar data provided by SunPy.
# Here we use an AIA 171 image, but the same approach can be applied to other datasets, including those with non celestial axes.

hdul = fits.open(AIA_171_IMAGE)
cube = NDCube(hdul[1].data, WCS(hdul[1].header))

###########################################################################
# Next, we define a new WCS with a finer pixel scale, note that while it is obvious that the CDELT values are changed to reflect the finer scale,
# the CRPIX values also need to be adjusted as the reference pixel position changes with the new scale.
# You can use any value for the scale factor, including non-integer values, greater or less than 1.
# You can also scale each axis independently.

scale_factor = 1.5
new_wcs = cube.wcs.deepcopy()
new_wcs.wcs.cdelt /= scale_factor
new_wcs.wcs.crpix *= scale_factor

###########################################################################
# Now we can reproject the original cube to the new WCS with higher resolution.
new_shape = tuple(int(s * scale_factor) for s in cube.data.shape)
reprojected_cube = cube.reproject_to(new_wcs, shape_out=new_shape)

###########################################################################
# Our new NDCube now has a higher resolution.
# We can compare the shapes of the original and reprojected cubes with new pixel axes.
print(cube.data.shape, reprojected_cube.data.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

ax1.imshow(cube.data, origin='lower', cmap=cm.sdoaia171, vmin=10, vmax=10000)
ax1.set_title('Original')
ax1.set_xlabel('X [pixel]')
ax1.set_ylabel('Y [pixel]')
ax2.imshow(reprojected_cube.data, origin='lower', cmap=cm.sdoaia171, vmin=10, vmax=10000)
ax2.set_title('Reprojected (Upscaled)')
ax2.set_xlabel('X [pixel]')
ax2.set_ylabel('Y [pixel]')
plt.tight_layout()
plt.show()
