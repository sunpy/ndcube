import astropy.wcs
import numpy as np

from ndcube import NDCube

# Define data array
data = np.random.rand(3, 4, 5)
data = k
# Define WCS transformations in an astropy WCS object.
wcs = astropy.wcs.WCS(naxis=3)
wcs.wcs.ctype = 'WAVE', 'HPLT-TAN', 'HPLN-TAN'
wcs.wcs.cunit = 'Angstrom', 'deg', 'deg'
wcs.wcs.cdelt = 0.2, 0.5, 0.4
wcs.wcs.crpix = 0, 2, 2
wcs.wcs.crval = 10, 0.5, 1

# Instantiate a simple ndcube
my_cube = NDCube(data, wcs=wcs)
