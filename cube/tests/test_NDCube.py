# -*- coding: utf-8 -*-
'''
Tests for NDCube
'''
from sunpycube.cube.NDCube import NDCube
from sunpycube.cube import cube_utils as cu
from sunpycube.wcs_util import WCS, _wcs_slicer, assert_wcs_are_equal
from collections import namedtuple
import pytest
import numpy as np
import astropy.units as u

DimensionPair = namedtuple('DimensionPair', 'dimensions axes')

# sample data for tests
# TODO: use a fixture reading from a test file. file TBD.
ht = {'CTYPE3': 'HPLT-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.5, 'CRPIX3': 0, 'CRVAL3': 0, 'NAXIS3': 2,
      'CTYPE2': 'WAVE    ', 'CUNIT2': 'Angstrom', 'CDELT2': 0.2, 'CRPIX2': 0, 'CRVAL2': 0, 'NAXIS2': 3,
      'CTYPE1': 'TIME    ', 'CUNIT1': 'min', 'CDELT1': 0.4, 'CRPIX1': 0, 'CRVAL1': 0, 'NAXIS1': 4}
wt = WCS(header=ht, naxis=3)
data = np.array([[[1, 2, 3, 4], [2, 4, 5, 3], [0, -1, 2, 3]],
                 [[2, 4, 5, 1], [10, 5, 2, 2], [10, 3, 3, 0]]])

hm = {
    'CTYPE1': 'WAVE    ', 'CUNIT1': 'Angstrom', 'CDELT1': 0.2, 'CRPIX1': 0, 'CRVAL1': 10, 'NAXIS1': 4,
    'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5, 'NAXIS2': 3,
    'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 2,
}
wm = WCS(header=hm, naxis=3)

h4 = {
    'CTYPE1': 'TIME    ', 'CUNIT1': 'min', 'CDELT1': 0.6, 'CRPIX1': 0, 'CRVAL1': 0, 'NAXIS1': 4,
    'CTYPE2': 'WAVE    ', 'CUNIT2': 'nm', 'CDELT2': 0.2, 'CRPIX2': 0, 'CRVAL2': 10, 'NAXIS2': 3,
    'CTYPE3': 'HPLT-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.5, 'CRPIX3': 0, 'CRVAL3': 0, 'NAXIS3': 2,
    'CTYPE4': 'HPLN-TAN', 'CUNIT4': 'deg', 'CDELT4': 0.4, 'CRPIX4': 2, 'CRVAL4': 1, 'NAXIS4': 2
}
w4 = WCS(header=h4, naxis=4)
data4 = np.array([[[[1, 2, 3, 4], [2, 4, 5, 3], [0, -1, 2, 3]],
                   [[2, 4, 5, 1], [10, 5, 2, 2], [10, 3, 3, 0]]],

                  [[[4, 6, 3, 5], [1, 0, 5, 3], [-4, 8, 7, 6]],
                   [[7, 3, 2, 6], [1, 7, 8, 7], [2, 4, 0, 1]]]])
mask_cubem = data > 0
mask_cube = data >= 0
cubem = NDCube(data, wcs=wm, mask=mask_cubem)
cube = NDCube(data, wcs=wt, missing_axis=[False, False, False, True], mask=mask_cube)
hcube = NDCube(data4, wcs=w4)


@pytest.mark.parametrize("test_input,expected,mask,wcs", [
    (cubem[:, 1], NDCube, mask_cubem[:, 1], _wcs_slicer(
        wm, [False, False, False], (slice(None, None, None), 1))),
    (cubem[:, 0:2], NDCube, mask_cubem[:, 0:2], _wcs_slicer(
        wm, [False, False, False], (slice(None, None, None), slice(0, 2, None)))),
    (cubem[:, :], NDCube, mask_cubem[:, :], _wcs_slicer(
        wm, [False, False, False], (slice(None, None, None), slice(None, None, None)))),
    (cubem[1, 1], NDCube, mask_cubem[1, 1], _wcs_slicer(wm, [False, False, False], (1, 1))),
    (cubem[1, 0:2], NDCube, mask_cubem[1, 0:2], _wcs_slicer(
        wm, [False, False, False], (1, slice(0, 2, None)))),
    (cubem[1, :], NDCube, mask_cubem[1, :], _wcs_slicer(
        wm, [False, False, False], (1, slice(None, None, None)))),
    (cube[:, 1], NDCube, mask_cube[:, 1], _wcs_slicer(
        wt, [True, False, False, False], (slice(None, None, None), 1))),
    (cube[:, 0:2], NDCube, mask_cube[:, 0:2], _wcs_slicer(
        wt, [True, False, False, False], (slice(None, None, None), slice(0, 2, None)))),
    (cube[:, :], NDCube, mask_cube[:, :], _wcs_slicer(
        wt, [True, False, False, False], (slice(None, None, None), slice(None, None, None)))),
    (cube[1, 1], NDCube, mask_cube[1, 1], _wcs_slicer(wt, [True, False, False, False], (1, 1))),
    (cube[1, 0:2], NDCube, mask_cube[1, 0:2], _wcs_slicer(
        wt, [True, False, False, False], (1, slice(0, 2, None)))),
    (cube[1, :], NDCube, mask_cube[1, :], _wcs_slicer(
        wt, [True, False, False, False], (1, slice(0, 2, None)))),
])
def test_slicing_second_axis_type(test_input, expected, mask, wcs):
    assert isinstance(test_input, expected)
    assert np.all(test_input.mask == mask)
    assert_wcs_are_equal(test_input.wcs, wcs[0])
    assert test_input.missing_axis == wcs[1]


@pytest.mark.parametrize("test_input,expected", [
    (cubem[:, 1].dimensions, DimensionPair(dimensions=u.Quantity(
        (2, 4), unit=u.pix), axes=['HPLN-TAN', 'WAVE'])),
    (cubem[:, 0:2].dimensions, DimensionPair(dimensions=u.Quantity(
        (2, 2, 4), unit=u.pix), axes=['HPLN-TAN', 'HPLT-TAN', 'WAVE'])),
    (cubem[:, :].dimensions, DimensionPair(dimensions=u.Quantity(
        (2, 3, 4), unit=u.pix), axes=['HPLN-TAN', 'HPLT-TAN', 'WAVE'])),
    (cubem[1, 1].dimensions, DimensionPair(dimensions=u.Quantity((4,), unit=u.pix), axes=['WAVE'])),
    (cubem[1, 0:2].dimensions, DimensionPair(dimensions=u.Quantity(
        (2, 4), unit=u.pix), axes=['HPLT-TAN', 'WAVE'])),
    (cubem[1, :].dimensions, DimensionPair(dimensions=u.Quantity(
        (3, 4), unit=u.pix), axes=['HPLT-TAN', 'WAVE'])),
    (cube[:, 1].dimensions, DimensionPair(dimensions=u.Quantity(
        (2, 4), unit=u.pix), axes=['HPLT-TAN', 'TIME'])),
    (cube[:, 0:2].dimensions, DimensionPair(dimensions=u.Quantity(
        (2, 2, 4), unit=u.pix), axes=['HPLT-TAN', 'WAVE', 'TIME'])),
    (cube[:, :].dimensions, DimensionPair(dimensions=u.Quantity(
        (2, 3, 4), unit=u.pix), axes=['HPLT-TAN', 'WAVE', 'TIME'])),
    (cube[1, 1].dimensions, DimensionPair(dimensions=u.Quantity((4,), unit=u.pix), axes=['TIME'])),
    (cube[1, 0:2].dimensions, DimensionPair(
        dimensions=u.Quantity((2, 4), unit=u.pix), axes=['WAVE', 'TIME'])),
    (cube[1, :].dimensions, DimensionPair(dimensions=u.Quantity(
        (3, 4), unit=u.pix), axes=['WAVE', 'TIME'])),
])
def test_slicing_second_axis(test_input, expected):
    assert test_input[1] == expected[1]
    assert np.all(test_input[0].value == expected[0].value)
    assert test_input[0].unit == expected[0].unit


@pytest.mark.parametrize("test_input,expected,mask,wcs", [
    (cubem[1], NDCube, mask_cubem[1], _wcs_slicer(wm, [False, False, False], 1)),
    (cubem[0:2], NDCube, mask_cubem[0:2], _wcs_slicer(
        wm, [False, False, False], slice(0, 2, None))),
    (cubem[:], NDCube, mask_cubem[:], _wcs_slicer(
        wm, [False, False, False], slice(None, None, None))),
    (cube[1], NDCube, mask_cube[1], _wcs_slicer(wt, [True, False, False, False], 1)),
    (cube[0:2], NDCube, mask_cube[0:2], _wcs_slicer(
        wt, [True, False, False, False], slice(0, 2, None))),
    (cube[:], NDCube, mask_cube[:], _wcs_slicer(
        wt, [True, False, False, False], slice(None, None, None))),
])
def test_slicing_first_axis_type(test_input, expected, mask, wcs):
    assert isinstance(test_input, expected)
    assert np.all(test_input.mask == mask)
    assert_wcs_are_equal(test_input.wcs, wcs[0])
    assert test_input.missing_axis == wcs[1]


@pytest.mark.parametrize("test_input,expected", [
    (cubem[1].dimensions, DimensionPair(dimensions=u.Quantity(
        (3, 4), unit=u.pix), axes=['HPLT-TAN', 'WAVE'])),
    (cubem[0:2].dimensions, DimensionPair(dimensions=u.Quantity(
        (2, 3, 4), unit=u.pix), axes=['HPLN-TAN', 'HPLT-TAN', 'WAVE'])),
    (cubem[:].dimensions, DimensionPair(dimensions=u.Quantity(
        (2, 3, 4), unit=u.pix), axes=['HPLN-TAN', 'HPLT-TAN', 'WAVE'])),
    (cube[1].dimensions, DimensionPair(dimensions=u.Quantity((3, 4), unit=u.pix), axes=['WAVE', 'TIME'])),
    (cube[0:2].dimensions, DimensionPair(dimensions=u.Quantity(
        (2, 3, 4), unit=u.pix), axes=['HPLT-TAN', 'WAVE', 'TIME'])),
    (cube[:].dimensions, DimensionPair(dimensions=u.Quantity(
        (2, 3, 4), unit=u.pix), axes=['HPLT-TAN', 'WAVE', 'TIME'])),
])
def test_slicing_first_axis_dimensions(test_input, expected):
    assert test_input[1] == expected[1]
    assert np.all(test_input[0].value == expected[0].value)
    assert test_input[0].unit == expected[0].unit


@pytest.mark.parametrize("test_input,expected,mask,wcs", [
    (cubem[:, :, 1], NDCube, mask_cubem[:, :, 1], _wcs_slicer(
        wm, [False, False, False], (slice(None, None, None), slice(None, None, None), 1))),
    (cubem[:, :, 0:2], NDCube, mask_cubem[:, :, 0:2], _wcs_slicer(wm, [False, False, False],
                                                                  (slice(None, None, None), slice(None, None, None), slice(0, 2, None)))),
    (cubem[:, :, :], NDCube, mask_cubem[:, :, :], _wcs_slicer(wm, [False, False, False],
                                                              (slice(None, None, None), slice(None, None, None), slice(None, None, None)))),
    (cubem[:, 1, 1], NDCube, mask_cubem[:, 1, 1], _wcs_slicer(
        wm, [False, False, False], (slice(None, None, None), 1, 1))),
    (cubem[:, 1, 0:2], NDCube, mask_cubem[:, 1, 0:2], _wcs_slicer(
        wm, [False, False, False], (slice(None, None, None), 1, slice(0, 2, None)))),
    (cubem[:, 1, :], NDCube, mask_cubem[:, 1, :], _wcs_slicer(
        wm, [False, False, False], (slice(None, None, None), 1, slice(None, None, None)))),
    (cubem[1, :, 1], NDCube, mask_cubem[1, :, 1], _wcs_slicer(
        wm, [False, False, False], (slice(None, None, None), 1, slice(None, None, None)))),
    (cubem[1, :, 0:2], NDCube, mask_cubem[1, :, 0:2], _wcs_slicer(
        wm, [False, False, False], (1, slice(None, None, None), slice(0, 2, None)))),
    (cubem[1, :, :], NDCube, mask_cubem[1, :, :], _wcs_slicer(
        wm, [False, False, False], (1, slice(None, None, None), slice(None, None, None)))),
    (cubem[1, 1, 1], NDCube, mask_cubem[1, 1, 1],
     _wcs_slicer(wm, [False, False, False], (1, 1, 1))),
    (cubem[1, 1, 0:2], NDCube, mask_cubem[1, 1, 0:2], _wcs_slicer(
        wm, [False, False, False], (1, 1, slice(0, 2, None)))),
    (cubem[1, 1, :], NDCube, mask_cubem[1, 1, :], _wcs_slicer(
        wm, [False, False, False], (1, 1, slice(None, None, None)))),
    (cube[:, :, 1], NDCube, mask_cube[:, :, 1], _wcs_slicer(
        wt, [True, False, False, False], (slice(None, None, None), slice(None, None, None), 1))),
    (cube[:, :, 0:2], NDCube, mask_cube[:, :, 0:2], _wcs_slicer(wt, [True, False, False,
                                                                     False], (slice(None, None, None), slice(None, None, None), slice(0, 2, None)))),
    (cube[:, :, :], NDCube, mask_cube[:, :, :], _wcs_slicer(wt, [True, False, False, False],
                                                            (slice(None, None, None), slice(None, None, None), slice(None, None, None)))),
    (cube[:, 1, 1], NDCube, mask_cube[:, 1, 1], _wcs_slicer(
        wt, [True, False, False, False], (slice(None, None, None), 1, 1))),
    (cube[:, 1, 0:2], NDCube, mask_cube[:, 1, 0:2], _wcs_slicer(
        wt, [True, False, False, False], (slice(None, None, None), 1, slice(0, 2, None)))),
    (cube[:, 1, :], NDCube, mask_cube[:, 1, :], _wcs_slicer(
        wt, [True, False, False, False], (slice(None, None, None), 1, slice(None, None, None)))),
    (cube[1, :, 1], NDCube, mask_cube[1, :, 1], _wcs_slicer(
        wt, [True, False, False, False], (1, slice(None, None, None), 1))),
    (cube[1, :, 0:2], NDCube, mask_cube[1, :, 0:2], _wcs_slicer(
        wt, [True, False, False, False], (1, slice(None, None, None), slice(0, 2, None)))),
    (cube[1, :, :], NDCube, mask_cube[1, :, :], _wcs_slicer(
        wt, [True, False, False, False], (1, slice(None, None, None), slice(None, None, None)))),
    (cube[1, 1, 1], NDCube, mask_cube[1, 1, 1], _wcs_slicer(
        wt, [True, False, False, False], (1, 1, 1))),
    (cube[1, 1, 0:2], NDCube, mask_cube[1, 1, 0:2], _wcs_slicer(
        wt, [True, False, False, False], (1, 1, slice(0, 2, None)))),
    (cube[1, 1, :], NDCube, mask_cube[1, 1, :], _wcs_slicer(
        wt, [True, False, False, False], (1, 1, slice(0, 2, None)))),
])
def test_slicing_third_axis(test_input, expected, mask, wcs):
    assert isinstance(test_input, expected)
    assert np.all(test_input.mask == mask)
    assert_wcs_are_equal(test_input.wcs, wcs[0])
    assert test_input.missing_axis == wcs[1]


@pytest.mark.parametrize("test_input,expected", [
    (cubem[:, :, 1].dimensions, DimensionPair(dimensions=u.Quantity(
        (2, 3), unit=u.pix), axes=['HPLN-TAN', 'HPLT-TAN'])),
    (cubem[:, :, 0:2].dimensions, DimensionPair(
        dimensions=u.Quantity((2, 3, 2), unit=u.pix), axes=['HPLN-TAN', 'HPLT-TAN', 'WAVE'])),
    (cubem[:, :, :].dimensions, DimensionPair(
        dimensions=u.Quantity((2, 3, 4), unit=u.pix), axes=['HPLN-TAN', 'HPLT-TAN', 'WAVE'])),
    (cubem[:, 1, 1].dimensions, DimensionPair(
        dimensions=u.Quantity((2,), unit=u.pix), axes=['HPLN-TAN'])),
    (cubem[:, 1, 0:2].dimensions, DimensionPair(
        dimensions=u.Quantity((2, 2), unit=u.pix), axes=['HPLN-TAN', 'WAVE'])),
    (cubem[:, 1, :].dimensions, DimensionPair(dimensions=u.Quantity(
        (2, 4), unit=u.pix), axes=['HPLN-TAN', 'WAVE'])),
    (cubem[1, :, 1].dimensions, DimensionPair(
        dimensions=u.Quantity((3,), unit=u.pix), axes=['HPLT-TAN'])),
    (cubem[1, :, 0:2].dimensions, DimensionPair(
        dimensions=u.Quantity((3, 2), unit=u.pix), axes=['HPLT-TAN', 'WAVE'])),
    (cubem[1, :, :].dimensions, DimensionPair(dimensions=u.Quantity(
        (3, 4), unit=u.pix), axes=['HPLT-TAN', 'WAVE'])),
    (cubem[1, 1, 1].dimensions, DimensionPair(dimensions=u.Quantity((), unit=u.pix), axes=[])),
    (cubem[1, 1, 0:2].dimensions, DimensionPair(
        dimensions=u.Quantity((2,), unit=u.pix), axes=['WAVE'])),
    (cubem[1, 1, :].dimensions, DimensionPair(
        dimensions=u.Quantity((4,), unit=u.pix), axes=['WAVE'])),
    (cube[:, :, 1].dimensions, DimensionPair(dimensions=u.Quantity(
        (2, 3), unit=u.pix), axes=['HPLT-TAN', 'WAVE'])),
    (cube[:, :, 0:2].dimensions, DimensionPair(dimensions=u.Quantity(
        (2, 3, 2), unit=u.pix), axes=['HPLT-TAN', 'WAVE', 'TIME'])),
    (cube[:, :, :].dimensions, DimensionPair(dimensions=u.Quantity(
        (2, 3, 4), unit=u.pix), axes=['HPLT-TAN', 'WAVE', 'TIME'])),
    (cube[:, 1, 1].dimensions, DimensionPair(
        dimensions=u.Quantity((2,), unit=u.pix), axes=['HPLT-TAN'])),
    (cube[:, 1, 0:2].dimensions, DimensionPair(
        dimensions=u.Quantity((2, 2), unit=u.pix), axes=['HPLT-TAN', 'TIME'])),
    (cube[:, 1, :].dimensions, DimensionPair(dimensions=u.Quantity(
        (2, 4), unit=u.pix), axes=['HPLT-TAN', 'TIME'])),
    (cube[1, :, 1].dimensions, DimensionPair(dimensions=u.Quantity((3,), unit=u.pix), axes=['WAVE'])),
    (cube[1, :, 0:2].dimensions, DimensionPair(
        dimensions=u.Quantity((3, 2), unit=u.pix), axes=['WAVE', 'TIME'])),
    (cube[1, :, :].dimensions, DimensionPair(
        dimensions=u.Quantity((3, 4), unit=u.pix), axes=['WAVE', 'TIME'])),
    (cube[1, 1, 1].dimensions, DimensionPair(dimensions=u.Quantity((), unit=u.pix), axes=[])),
    (cube[1, 1, 0:2].dimensions, DimensionPair(
        dimensions=u.Quantity((2,), unit=u.pix), axes=['TIME'])),
    (cube[1, 1, :].dimensions, DimensionPair(dimensions=u.Quantity((4,), unit=u.pix
                                                                   ), axes=['TIME'])),
])
def test_slicing_third_axis(test_input, expected):
    assert test_input[1] == expected[1]
    assert np.all(test_input[0].value == expected[0].value)
    assert test_input[0].unit == expected[0].unit
