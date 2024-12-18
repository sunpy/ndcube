import pytest

from astropy.wcs.wcsapi.conftest import Celestial2DLowLevelWCS as ApyCelestial2DLowLevelWCS
from astropy.wcs.wcsapi.conftest import *  # NOQA


class Celestial2DLowLevelWCS(ApyCelestial2DLowLevelWCS):
    def __init__(self):
        self._pixel_bounds = (-1, 5), (1, 7)

    @property
    def pixel_bounds(self):
        return self._pixel_bounds

    @pixel_bounds.setter
    def pixel_bounds(self, val):
        self._pixel_bounds = val


@pytest.fixture
def celestial_2d_ape14_wcs():
    return Celestial2DLowLevelWCS()
