import unittest

import numpy as np
import pytest
from astropy.wcs import WCS

from ndcube import utils
from ndcube.tests import helpers

ht_with_celestial = {
    'CTYPE4': 'HPLN-TAN', 'CUNIT4': 'deg', 'CDELT4': 1, 'CRPIX4': 0, 'CRVAL4': 0, 'NAXIS4': 1,
    'CNAME4': 'redundant axis', 'CROTA4': 0,
    'CTYPE3': 'HPLT-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.5, 'CRPIX3': 0, 'CRVAL3': 0, 'NAXIS3': 2,
    'CTYPE2': 'WAVE    ', 'CUNIT2': 'Angstrom', 'CDELT2': 0.2, 'CRPIX2': 0, 'CRVAL2': 0,
    'NAXIS2': 3,
    'CTYPE1': 'TIME    ', 'CUNIT1': 'min', 'CDELT1': 0.4, 'CRPIX1': 0, 'CRVAL1': 0, 'NAXIS1': 4}

hm = {'CTYPE1': 'WAVE    ', 'CUNIT1': 'Angstrom', 'CDELT1': 0.2, 'CRPIX1': 0, 'CRVAL1': 10,
      'NAXIS1': 4,
      'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5,
      'NAXIS2': 3,
      'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 2}
wm = WCS(header=hm)

hm_reindexed_102 = {
    'CTYPE2': 'WAVE    ', 'CUNIT2': 'Angstrom', 'CDELT2': 0.2, 'CRPIX2': 0, 'CRVAL2': 10,
    'NAXIS2': 4,
    'CTYPE1': 'HPLT-TAN', 'CUNIT1': 'deg', 'CDELT1': 0.5, 'CRPIX1': 2, 'CRVAL1': 0.5, 'NAXIS1': 3,
    'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 2}
wm_reindexed_102 = WCS(header=hm_reindexed_102)


@pytest.fixture
def axis_correlation_matrix():
    return _axis_correlation_matrix()


def _axis_correlation_matrix():
    shape = (4, 4)
    acm = np.zeros(shape, dtype=bool)
    for i in range(min(shape)):
        acm[i, i] = True
    acm[0, 1] = True
    acm[1, 0] = True
    acm[-1, 0] = True
    return acm


class TestWCS():
    def __init__(self):
        self.world_axis_physical_types = [
            'custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat', 'em.wl', 'time']
        self.axis_correlation_matrix = _axis_correlation_matrix()


TEST_WCS = TestWCS()


@pytest.mark.parametrize("test_input,expected", [
    ((wm, 0), (0, 1)),
    ((wm, 1), (0, 1)),
    ((wm, 2), (2,)),
    ((wm, 1), (0, 1))
])
def test_get_dependent_data_axes(test_input, expected):
    output = utils.wcs.get_dependent_data_axes(*test_input)
    assert output == expected


@pytest.mark.parametrize("test_input,expected", [
    ((wm, 0), (0,)),
    ((wm, 1), (1, 2)),
    ((wm, 2), (1, 2)),
])
def test_get_dependent_wcs_axes(test_input, expected):
    output = utils.wcs.get_dependent_wcs_axes(*test_input)
    assert output == expected


def test_reflect_axis_index():
    test_input = np.array([1, 4, -2])
    naxes = 5
    expected = np.array([3, 0, 1])
    output = utils.wcs.reflect_axis_index(test_input, naxes)
    assert all(output == expected)


def test_pixel_axis_to_world_axes(axis_correlation_matrix):
    output = utils.wcs.pixel_axis_to_world_axes(0, axis_correlation_matrix)
    expected = np.array([0, 1, 3])
    assert all(output == expected)


def test_world_axis_to_pixel_axes(axis_correlation_matrix):
    output = utils.wcs.world_axis_to_pixel_axes(1, axis_correlation_matrix)
    expected = np.array([0, 1])
    assert all(output == expected)


def test_pixel_axis_to_physical_types():
    output = utils.wcs.pixel_axis_to_physical_types(0, TEST_WCS)
    expected = np.array(['custom:pos.helioprojective.lon',
                         'custom:pos.helioprojective.lat', 'time'])
    print(output, expected)
    assert all(output == expected)


def test_physical_type_to_pixel_axes():
    output = utils.wcs.physical_type_to_pixel_axes('lon', TEST_WCS)
    expected = np.array([0, 1])
    assert all(output == expected)


@pytest.mark.parametrize("test_input,expected", [('wl', 2), ('em.wl', 2)])
def test_physical_type_to_world_axis(test_input, expected):
    world_axis_physical_types = ['custom:pos.helioprojective.lon',
                                 'custom:pos.helioprojective.lat', 'em.wl', 'time']
    output = utils.wcs.physical_type_to_world_axis(test_input, world_axis_physical_types)
    assert output == expected


def test_get_dependent_pixel_axes(axis_correlation_matrix):
    output = utils.wcs.get_dependent_pixel_axes(0, axis_correlation_matrix)
    expected = np.array([0, 1, 3])
    assert all(output == expected)


def test_get_dependent_array_axes(axis_correlation_matrix):
    output = utils.wcs.get_dependent_array_axes(3, axis_correlation_matrix)
    expected = np.array([0, 2, 3])
    assert all(output == expected)


def test_get_dependent_world_axes(axis_correlation_matrix):
    output = utils.wcs.get_dependent_world_axes(3, axis_correlation_matrix)
    expected = np.array([0, 3])
    print(output, expected)
    assert all(output == expected)


def test_get_dependent_physical_types():
    output = utils.wcs.get_dependent_physical_types("time", TEST_WCS)
    expected = np.array(['custom:pos.helioprojective.lon', 'time'])
    assert all(output == expected)
