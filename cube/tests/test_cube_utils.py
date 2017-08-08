# -*- coding: utf-8 -*-
import pytest
from sunpycube.cube import cube_utils as cu
import numpy as np
from sunpycube.wcs_util import WCS
from astropy import units as u

ht = {
    'CTYPE1': 'HPLT-TAN', 'CUNIT1': 'deg', 'CDELT1': 0.5, 'CRPIX1': 0, 'CRVAL1': 0, 'NAXIS1': 2,
    'CTYPE2': 'WAVE    ', 'CUNIT2': 'Angstrom', 'CDELT2': 0.2, 'CRPIX2': 0, 'CRVAL2': 0, 'NAXIS2': 3,
    'CTYPE3': 'TIME    ', 'CUNIT3': 'min', 'CDELT3': 0.4, 'CRPIX3': 0, 'CRVAL3': 0, 'NAXIS3': 4
}
wt = WCS(header=ht, naxis=3)

hm = {
    'CTYPE1': 'WAVE    ', 'CUNIT1': 'Angstrom', 'CDELT1': 0.2, 'CRPIX1': 0, 'CRVAL1': 10, 'NAXIS1': 4,
    'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5, 'NAXIS2': 3,
    'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 2,
}
wm = WCS(header=hm, naxis=3)

data = np.array([[[1, 2, 3, 4], [2, 4, 5, 3], [0, -1, 2, 3]],
                 [[2, 4, 5, 1], [10, 5, 2, 2], [10, 3, 3, 0]]])


def test_select_order():
    lists = [['TIME', 'WAVE', 'HPLT-TAN', 'HPLN-TAN'],
             ['WAVE', 'HPLT-TAN', 'UTC', 'HPLN-TAN'],
             ['HPLT-TAN', 'TIME', 'HPLN-TAN'],
             ['HPLT-TAN', 'DEC--TAN', 'WAVE'],
             [],
             ['UTC', 'TIME', 'WAVE', 'HPLT-TAN']]

    results = [[0, 1, 2, 3],
               [2, 0, 1, 3],
               [1, 0, 2],  # Second order is initial order
               [2, 0, 1],
               [],
               [1, 0, 2, 3]]

    for (l, r) in zip(lists, results):
        assert cu.select_order(l) == r


@pytest.mark.parametrize("test_input,expected", [
    (cu._convert_cube_like_index_to_sequence_indices(5, np.array([8, 16, 24, 32])), (0, 5)),
    (cu._convert_cube_like_index_to_sequence_indices(8, np.array([8, 16, 24, 32])), (1, 0)),
    (cu._convert_cube_like_index_to_sequence_indices(20, np.array([8, 16, 24, 32])), (2, 4)),
    (cu._convert_cube_like_index_to_sequence_indices(50, np.array([8, 16, 24, 32])), (3, 7)),
])
def test_convert_cube_like_index_to_sequence_indices(test_input, expected):
    assert test_input == expected


@pytest.mark.parametrize("test_input,expected", [
    (cu._convert_cube_like_slice_to_sequence_slices(
        slice(2, 5), np.array([8, 16, 24, 32])), (slice(0, 1), slice(2, 5))),
    (cu._convert_cube_like_slice_to_sequence_slices(
        slice(5, 15), np.array([8, 16, 24, 32])), (slice(0, 2), [slice(5, 8), slice(0, 7)])),
    (cu._convert_cube_like_slice_to_sequence_slices(
        slice(5, 16), np.array([8, 16, 24, 32])), (slice(0, 2), [slice(5, 8)])),
    (cu._convert_cube_like_slice_to_sequence_slices(
        slice(5, 23), np.array([8, 16, 24, 32])), (slice(0, 3), [slice(5, 8), slice(0, 7)])),
    (cu._convert_cube_like_slice_to_sequence_slices(
        slice(5, 100), np.array([8, 16, 24, 32])), (slice(0, 4), [slice(5, 8)])),
])
def test_convert_cube_like_slice_to_sequence_slices(test_input, expected):
    assert test_input == expected
