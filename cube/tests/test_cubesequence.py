from __future__ import absolute_import
from sunpycube.cube.datacube import Cube, CubeSequence
from sunpycube.cube import cube_utils as cu
from sunpy.map.mapbase import GenericMap
from sunpycube.spectra.spectrum import Spectrum
from sunpycube.spectra.spectrogram import Spectrogram
from sunpy.lightcurve.lightcurve import LightCurve
import numpy as np
from sunpycube.wcs_util import WCS
import pytest
import astropy.units as u


# sample data for tests
# TODO: use a fixture reading from a test file. file TBD.
data = np.array([[[1, 2, 3, 4], [2, 4, 5, 3], [0, -1, 2, 3]],
                 [[2, 4, 5, 1], [10, 5, 2, 2], [10, 3, 3, 0]]])

data2 = np.array([[[11, 22, 33, 44], [22, 44, 55, 33], [0, -1, 22, 33]],
                  [[22, 44, 55, 11], [10, 55, 22, 22], [10, 33, 33, 0]]])

ht = {'CTYPE1': 'HPLT-TAN', 'CUNIT1': 'deg', 'CDELT1': 0.5, 'CRPIX1': 0, 'CRVAL1': 0, 'NAXIS1': 2,
      'CTYPE2': 'WAVE    ', 'CUNIT2': 'Angstrom', 'CDELT2': 0.2, 'CRPIX2': 0, 'CRVAL2': 0, 'NAXIS2': 3,
      'CTYPE3': 'TIME    ', 'CUNIT3': 'min', 'CDELT3': 0.4, 'CRPIX3': 0, 'CRVAL3': 0, 'NAXIS3': 4}

hm = {
    'CTYPE1': 'WAVE    ', 'CUNIT1': 'Angstrom', 'CDELT1': 0.2, 'CRPIX1': 0, 'CRVAL1': 10, 'NAXIS1': 4,
    'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5, 'NAXIS2': 3,
    'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 2,
}

wt = WCS(header=ht, naxis=3)
wm = WCS(header=hm, naxis=3)

cube1 = Cube(data, wt)
cube2 = Cube(data, wm)
cube3 = Cube(data2, wt)
cube4 = Cube(data2, wm)

seq = CubeSequence([cube1, cube2, cube3, cube4])


@pytest.mark.parametrize("test_input,expected", [
    (seq[0], Cube),
    (seq[1], Cube),
    (seq[2], Cube),
    (seq[3], Cube),
    (seq[0:1], CubeSequence),
    (seq[1:3], CubeSequence),
    (seq[0:2], CubeSequence),
    (seq[slice(0, 2)], CubeSequence),
    (seq[slice(0, 3)], CubeSequence),
])
def test_slice_first_index_sequence(test_input, expected):
    assert isinstance(test_input, expected)


@pytest.mark.parametrize("test_input,expected", [
    (seq[0:1].shape[0], 1),
    (seq[1:3].shape[0], 2),
    (seq[0:2].shape[0], 2),
    (seq[0::].shape[0], 4),
    (seq[slice(0, 2)].shape[0], 2),
    (seq[slice(0, 3)].shape[0], 3),
])
def test_slice_first_index_sequence(test_input, expected):
    assert test_input == expected
