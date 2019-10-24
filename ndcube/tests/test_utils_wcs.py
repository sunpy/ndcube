import pytest
import unittest

import numpy as np
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
    helpers.assert_wcs_are_equal(utils.wcs.reindex_wcs(*test_input), expected)


@pytest.mark.parametrize("test_input", [
    (TypeError, wm, 0),
    (TypeError, wm, np.array(['spam', 'eggs', 'ham'])),
])
def test_reindex_wcs_errors(test_input):
    with pytest.raises(test_input[0]):
        utils.wcs.reindex_wcs(*test_input[1:])


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
