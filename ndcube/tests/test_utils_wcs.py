import pytest
import unittest

import numpy as np
import astropy.wcs

from ndcube import utils
from ndcube.tests import helpers


ht = {'CTYPE3': 'HPLT-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.5, 'CRPIX3': 0, 'CRVAL3': 0, 'NAXIS3': 2,
      'CTYPE2': 'WAVE    ', 'CUNIT2': 'Angstrom', 'CDELT2': 0.2, 'CRPIX2': 0, 'CRVAL2': 0,
      'NAXIS2': 3,
      'CTYPE1': 'TIME    ', 'CUNIT1': 'min', 'CDELT1': 0.4, 'CRPIX1': 0, 'CRVAL1': 0, 'NAXIS1': 4}
wt = utils.wcs.WCS(header=ht, naxis=3)

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
wm = utils.wcs.WCS(header=hm, naxis=3)

hm_reindexed_102 = {
    'CTYPE2': 'WAVE    ', 'CUNIT2': 'Angstrom', 'CDELT2': 0.2, 'CRPIX2': 0, 'CRVAL2': 10,
    'NAXIS2': 4,
    'CTYPE1': 'HPLT-TAN', 'CUNIT1': 'deg', 'CDELT1': 0.5, 'CRPIX1': 2, 'CRVAL1': 0.5, 'NAXIS1': 3,
    'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 2}
wm_reindexed_102 = utils.wcs.WCS(header=hm_reindexed_102, naxis=3)


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


@pytest.fixture
def test_wcs():
    return TestWCS()


class TestWCS():
    def __init__(self):
        self.world_axis_physical_types = [
            'custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat', 'em.wl', 'time']
        self.axis_correlation_matrix = _axis_correlation_matrix()


@pytest.mark.parametrize("test_input,expected", [(ht, True), (hm, False)])
def test_wcs_needs_augmenting(test_input, expected):
    assert utils.wcs.WCS._needs_augmenting(test_input) is expected


@pytest.mark.parametrize("test_input,expected", [((ht, 3), ht_with_celestial)])
def test_wcs_augment(test_input, expected):
    unit_tester = unittest.TestCase()
    unit_tester.assertEqual(utils.wcs.WCS._augment(*test_input), expected)


@pytest.mark.parametrize(
    "test_input,expected",
    [({}, False),
     ([slice(1, 5), slice(-1, -5, -2)], True)])
def test_all_slice(test_input, expected):
    assert utils.wcs._all_slice(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [({}, []),
     ((slice(1, 2), slice(1, 3), 2, slice(2, 4), 8),
      [slice(1, 2, None), slice(1, 3, None), slice(2, 3, None),
       slice(2, 4, None), slice(8, 9, None)])])
def test_slice_list(test_input, expected):
    assert utils.wcs._slice_list(test_input) == expected


@pytest.mark.parametrize("test_input,expected", [
    ((wm, np.array([1, 0, 2])), wm_reindexed_102),
    ((wm, np.array([1, 0, -1])), wm_reindexed_102)
])
def test_reindex_wcs(test_input, expected):
    print(utils.wcs.reindex_wcs(*test_input))
    print(expected)
    helpers.assert_wcs_are_equal(utils.wcs.reindex_wcs(*test_input), expected)


@pytest.mark.parametrize("test_input", [
    (TypeError, wm, 0),
    (TypeError, wm, np.array(['spam', 'eggs', 'ham'])),
])
def test_reindex_wcs_errors(test_input):
    with pytest.raises(test_input[0]):
        utils.wcs.reindex_wcs(*test_input[1:])


@pytest.mark.parametrize("test_input,expected", [
    ((wm, 0, [False, False, False]), (0, 1)),
    ((wm, 1, [False, False, False]), (0, 1)),
    ((wm, 2, [False, False, False]), (2,)),
    ((wm, 1, [False, False, True]), (1,))
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


@pytest.mark.parametrize("test_input,expected", [
    (wm, np.array([[True, False, False], [False, True, True], [False, True, True]])),
    (wt, np.array([[True, False, False, False], [False, True, False, False],
                   [False, False, True, True], [False, False, True, True]])),
    (wm_reindexed_102, np.array([[True, False, True], [False, True, False],
                                 [True, False, True]]))
])
def test_axis_correlation_matrix(test_input, expected):
    assert (utils.wcs.axis_correlation_matrix(test_input) == expected).all()


def test_convert_between_array_and_pixel_axes():
    test_input = np.array([1, 4, -2])
    naxes = 5
    expected = np.array([3, 0, 1])
    output = utils.wcs.convert_between_array_and_pixel_axes(test_input, naxes)
    assert all(output == expected)


def test_pixel_axis_to_world_axes(axis_correlation_matrix):
    output = utils.wcs.pixel_axis_to_world_axes(0, axis_correlation_matrix)
    expected = np.array([0, 1, 3])
    assert all(output == expected)


@pytest.mark.parametrize("test_input,expected", [('wl', 2), ('em.wl', 2)])
def test_physical_type_to_world_axis(test_input, expected):
    world_axis_physical_types = ['custom:pos.helioprojective.lon',
                                 'custom:pos.helioprojective.lat', 'em.wl', 'time']
    output = utils.wcs.physical_type_to_world_axis(test_input, world_axis_physical_types)
    assert output == expected
