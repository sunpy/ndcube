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

DimensionPair = namedtuple('DimensionPair', 'shape axis_types')

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

mask_cubem = data > 0
mask_cube = data >= 0
cubem = NDCube(data, wcs=wm, mask=mask_cubem, uncertainty=data)
cube = NDCube(data, wcs=wt, missing_axis=[False, False,
                                          False, True], mask=mask_cube, uncertainty=[2, 3])


@pytest.mark.parametrize("test_input,expected,mask,wcs,uncertainty,dimensions", [
    (cubem[:, 1], NDCube, mask_cubem[:, 1], _wcs_slicer(
        wm, [False, False, False], (slice(None, None, None), 1)), data[:, 1], DimensionPair(shape=u.Quantity((2, 4), unit=u.pix), axis_types=['HPLN-TAN', 'WAVE'])),
    (cubem[:, 0:2], NDCube, mask_cubem[:, 0:2], _wcs_slicer(
        wm, [False, False, False], (slice(None, None, None), slice(0, 2, None))), data[:, 0:2], DimensionPair(shape=u.Quantity((2, 2, 4), unit=u.pix), axis_types=['HPLN-TAN', 'HPLT-TAN', 'WAVE'])),
    (cubem[:, :], NDCube, mask_cubem[:, :], _wcs_slicer(
        wm, [False, False, False], (slice(None, None, None), slice(None, None, None))), data[:, :], DimensionPair(shape=u.Quantity((2, 3, 4), unit=u.pix), axis_types=['HPLN-TAN', 'HPLT-TAN', 'WAVE'])),
    (cubem[1, 1], NDCube, mask_cubem[1, 1], _wcs_slicer(
        wm, [False, False, False], (1, 1)), data[1, 1], DimensionPair(shape=u.Quantity((4,), unit=u.pix), axis_types=['WAVE'])),
    (cubem[1, 0:2], NDCube, mask_cubem[1, 0:2], _wcs_slicer(
        wm, [False, False, False], (1, slice(0, 2, None))), data[1, 0:2], DimensionPair(shape=u.Quantity((2, 4), unit=u.pix), axis_types=['HPLT-TAN', 'WAVE'])),
    (cubem[1, :], NDCube, mask_cubem[1, :], _wcs_slicer(
        wm, [False, False, False], (1, slice(None, None, None))), data[1, :], DimensionPair(shape=u.Quantity((3, 4), unit=u.pix), axis_types=['HPLT-TAN', 'WAVE'])),
    (cube[:, 1], NDCube, mask_cube[:, 1], _wcs_slicer(
        wt, [True, False, False, False], (slice(None, None, None), 1)), np.array([2, 3]), DimensionPair(shape=u.Quantity((2, 4), unit=u.pix), axis_types=['HPLT-TAN', 'TIME'])),
    (cube[:, 0:2], NDCube, mask_cube[:, 0:2], _wcs_slicer(
        wt, [True, False, False, False], (slice(None, None, None), slice(0, 2, None))), np.array([2, 3]), DimensionPair(shape=u.Quantity((2, 2, 4), unit=u.pix), axis_types=['HPLT-TAN', 'WAVE', 'TIME'])),
    (cube[:, :], NDCube, mask_cube[:, :], _wcs_slicer(
        wt, [True, False, False, False], (slice(None, None, None), slice(None, None, None))), np.array([2, 3]), DimensionPair(shape=u.Quantity((2, 3, 4), unit=u.pix), axis_types=['HPLT-TAN', 'WAVE', 'TIME'])),
    (cube[1, 1], NDCube, mask_cube[1, 1], _wcs_slicer(
        wt, [True, False, False, False], (1, 1)), np.array([2, 3]), DimensionPair(shape=u.Quantity((4,), unit=u.pix), axis_types=['TIME'])),
    (cube[1, 0:2], NDCube, mask_cube[1, 0:2], _wcs_slicer(
        wt, [True, False, False, False], (1, slice(0, 2, None))), np.array([2, 3]), DimensionPair(shape=u.Quantity((2, 4), unit=u.pix), axis_types=['WAVE', 'TIME'])),
    (cube[1, :], NDCube, mask_cube[1, :], _wcs_slicer(
        wt, [True, False, False, False], (1, slice(0, 2, None))), np.array([2, 3]), DimensionPair(shape=u.Quantity((3, 4), unit=u.pix), axis_types=['WAVE', 'TIME'])),
])
def test_slicing_second_axis_type(test_input, expected, mask, wcs, uncertainty, dimensions):
    assert isinstance(test_input, expected)
    assert np.all(test_input.mask == mask)
    assert_wcs_are_equal(test_input.wcs, wcs[0])
    assert test_input.missing_axis == wcs[1]
    assert test_input.uncertainty.array.shape == uncertainty.shape
    assert test_input.dimensions[1] == dimensions[1]
    assert np.all(test_input.dimensions[0].value == dimensions[0].value)
    assert test_input.dimensions[0].unit == dimensions[0].unit


@pytest.mark.parametrize("test_input,expected,mask,wcs,uncertainty,dimensions", [
    (cubem[1], NDCube, mask_cubem[1], _wcs_slicer(wm, [False, False, False], 1), data[1], DimensionPair(shape=u.Quantity(
        (3, 4), unit=u.pix), axis_types=['HPLT-TAN', 'WAVE'])),
    (cubem[0:2], NDCube, mask_cubem[0:2], _wcs_slicer(
        wm, [False, False, False], slice(0, 2, None)), data[0:2], DimensionPair(shape=u.Quantity(
            (2, 3, 4), unit=u.pix), axis_types=['HPLN-TAN', 'HPLT-TAN', 'WAVE'])),
    (cubem[:], NDCube, mask_cubem[:], _wcs_slicer(
        wm, [False, False, False], slice(None, None, None)), data[:], DimensionPair(shape=u.Quantity(
            (2, 3, 4), unit=u.pix), axis_types=['HPLN-TAN', 'HPLT-TAN', 'WAVE'])),
    (cube[1], NDCube, mask_cube[1], _wcs_slicer(
        wt, [True, False, False, False], 1), np.array([2, 3]), DimensionPair(shape=u.Quantity(
            (3, 4), unit=u.pix), axis_types=['WAVE', 'TIME'])),
    (cube[0:2], NDCube, mask_cube[0:2], _wcs_slicer(
        wt, [True, False, False, False], slice(0, 2, None)), np.array([2, 3]), DimensionPair(shape=u.Quantity(
            (2, 3, 4), unit=u.pix), axis_types=['HPLT-TAN', 'WAVE', 'TIME'])),
    (cube[:], NDCube, mask_cube[:], _wcs_slicer(
        wt, [True, False, False, False], slice(None, None, None)), np.array([2, 3]), DimensionPair(shape=u.Quantity(
            (2, 3, 4), unit=u.pix), axis_types=['HPLT-TAN', 'WAVE', 'TIME'])),
])
def test_slicing_first_axis_type(test_input, expected, mask, wcs, uncertainty, dimensions):
    assert isinstance(test_input, expected)
    assert np.all(test_input.mask == mask)
    assert_wcs_are_equal(test_input.wcs, wcs[0])
    assert test_input.missing_axis == wcs[1]
    assert test_input.uncertainty.array.shape == uncertainty.shape
    assert test_input.dimensions[1] == dimensions[1]
    assert np.all(test_input.dimensions[0].value == dimensions[0].value)
    assert test_input.dimensions[0].unit == dimensions[0].unit


@pytest.mark.parametrize("test_input,expected,mask,wcs,uncertainty,dimensions", [
    (cubem[:, :, 1], NDCube, mask_cubem[:, :, 1], _wcs_slicer(
        wm, [False, False, False], (slice(None, None, None), slice(None, None, None), 1)), data[:, :, 1], DimensionPair(shape=u.Quantity(
            (2, 3), unit=u.pix), axis_types=['HPLN-TAN', 'HPLT-TAN'])),
    (cubem[:, :, 0:2], NDCube, mask_cubem[:, :, 0:2], _wcs_slicer(wm, [False, False, False],
                                                                  (slice(None, None, None), slice(None, None, None), slice(0, 2, None))), data[:, :, 0:2], DimensionPair(
        shape=u.Quantity((2, 3, 2), unit=u.pix), axis_types=['HPLN-TAN', 'HPLT-TAN', 'WAVE'])),
    (cubem[:, :, :], NDCube, mask_cubem[:, :, :], _wcs_slicer(wm, [False, False, False],
                                                              (slice(None, None, None), slice(None, None, None), slice(None, None, None))), data[:, :, :], DimensionPair(
        shape=u.Quantity((2, 3, 4), unit=u.pix), axis_types=['HPLN-TAN', 'HPLT-TAN', 'WAVE'])),
    (cubem[:, 1, 1], NDCube, mask_cubem[:, 1, 1], _wcs_slicer(
        wm, [False, False, False], (slice(None, None, None), 1, 1)), data[:, 1, 1], DimensionPair(
        shape=u.Quantity((2,), unit=u.pix), axis_types=['HPLN-TAN'])),
    (cubem[:, 1, 0:2], NDCube, mask_cubem[:, 1, 0:2], _wcs_slicer(
        wm, [False, False, False], (slice(None, None, None), 1, slice(0, 2, None))), data[:, 1, 0:2], DimensionPair(
        shape=u.Quantity((2, 2), unit=u.pix), axis_types=['HPLN-TAN', 'WAVE'])),
    (cubem[:, 1, :], NDCube, mask_cubem[:, 1, :], _wcs_slicer(
        wm, [False, False, False], (slice(None, None, None), 1, slice(None, None, None))), data[:, 1, :], DimensionPair(shape=u.Quantity(
            (2, 4), unit=u.pix), axis_types=['HPLN-TAN', 'WAVE'])),
    (cubem[1, :, 1], NDCube, mask_cubem[1, :, 1], _wcs_slicer(
        wm, [False, False, False], (1, slice(None, None, None), 1)), data[1, :, 1], DimensionPair(
        shape=u.Quantity((3,), unit=u.pix), axis_types=['HPLT-TAN'])),
    (cubem[1, :, 0:2], NDCube, mask_cubem[1, :, 0:2], _wcs_slicer(
        wm, [False, False, False], (1, slice(None, None, None), slice(0, 2, None))), data[1, :, 0:2], DimensionPair(
        shape=u.Quantity((3, 2), unit=u.pix), axis_types=['HPLT-TAN', 'WAVE'])),
    (cubem[1, :, :], NDCube, mask_cubem[1, :, :], _wcs_slicer(
        wm, [False, False, False], (1, slice(None, None, None), slice(None, None, None))), data[1, :, :], DimensionPair(shape=u.Quantity(
            (3, 4), unit=u.pix), axis_types=['HPLT-TAN', 'WAVE'])),
    (cubem[1, 1, 1], NDCube, mask_cubem[1, 1, 1],
     _wcs_slicer(wm, [False, False, False], (1, 1, 1)), data[1, 1, 1], DimensionPair(shape=u.Quantity((), unit=u.pix), axis_types=[])),
    (cubem[1, 1, 0:2], NDCube, mask_cubem[1, 1, 0:2], _wcs_slicer(
        wm, [False, False, False], (1, 1, slice(0, 2, None))), data[1, 1, 0:2], DimensionPair(
        shape=u.Quantity((2,), unit=u.pix), axis_types=['WAVE'])),
    (cubem[1, 1, :], NDCube, mask_cubem[1, 1, :], _wcs_slicer(
        wm, [False, False, False], (1, 1, slice(None, None, None))), data[1, 1, :], DimensionPair(
        shape=u.Quantity((4,), unit=u.pix), axis_types=['WAVE'])),
    (cube[:, :, 1], NDCube, mask_cube[:, :, 1], _wcs_slicer(
        wt, [True, False, False, False], (slice(None, None, None), slice(None, None, None), 1)), np.array([2, 3]), DimensionPair(shape=u.Quantity(
            (2, 3), unit=u.pix), axis_types=['HPLT-TAN', 'WAVE'])),
    (cube[:, :, 0:2], NDCube, mask_cube[:, :, 0:2], _wcs_slicer(wt, [True, False, False,
                                                                     False], (slice(None, None, None), slice(None, None, None), slice(0, 2, None))), np.array([2, 3]), DimensionPair(shape=u.Quantity(
                                                                         (2, 3, 2), unit=u.pix), axis_types=['HPLT-TAN', 'WAVE', 'TIME'])),
    (cube[:, :, :], NDCube, mask_cube[:, :, :], _wcs_slicer(wt, [True, False, False, False],
                                                            (slice(None, None, None), slice(None, None, None), slice(None, None, None))), np.array([2, 3]), DimensionPair(shape=u.Quantity(
                                                                (2, 3, 4), unit=u.pix), axis_types=['HPLT-TAN', 'WAVE', 'TIME'])),
    (cube[:, 1, 1], NDCube, mask_cube[:, 1, 1], _wcs_slicer(
        wt, [True, False, False, False], (slice(None, None, None), 1, 1)), np.array([2, 3]), DimensionPair(
        shape=u.Quantity((2,), unit=u.pix), axis_types=['HPLT-TAN'])),
    (cube[:, 1, 0:2], NDCube, mask_cube[:, 1, 0:2], _wcs_slicer(
        wt, [True, False, False, False], (slice(None, None, None), 1, slice(0, 2, None))), np.array([2, 3]), DimensionPair(
        shape=u.Quantity((2, 2), unit=u.pix), axis_types=['HPLT-TAN', 'TIME'])),
    (cube[:, 1, :], NDCube, mask_cube[:, 1, :], _wcs_slicer(
        wt, [True, False, False, False], (slice(None, None, None), 1, slice(None, None, None))), np.array([2, 3]), DimensionPair(shape=u.Quantity(
            (2, 4), unit=u.pix), axis_types=['HPLT-TAN', 'TIME'])),
    (cube[1, :, 1], NDCube, mask_cube[1, :, 1], _wcs_slicer(
        wt, [True, False, False, False], (1, slice(None, None, None), 1)), np.array([2, 3]), DimensionPair(
        shape=u.Quantity((3,), unit=u.pix), axis_types=['WAVE'])),
    (cube[1, :, 0:2], NDCube, mask_cube[1, :, 0:2], _wcs_slicer(
        wt, [True, False, False, False], (1, slice(None, None, None), slice(0, 2, None))), np.array([2, 3]), DimensionPair(
        shape=u.Quantity((3, 2), unit=u.pix), axis_types=['WAVE', 'TIME'])),
    (cube[1, :, :], NDCube, mask_cube[1, :, :], _wcs_slicer(
        wt, [True, False, False, False], (1, slice(None, None, None), slice(None, None, None))), np.array([2, 3]), DimensionPair(
        shape=u.Quantity((3, 4), unit=u.pix), axis_types=['WAVE', 'TIME'])),
    (cube[1, 1, 1], NDCube, mask_cube[1, 1, 1], _wcs_slicer(
        wt, [True, False, False, False], (1, 1, 1)), np.array([2, 3]), DimensionPair(shape=u.Quantity((), unit=u.pix), axis_types=[])),
    (cube[1, 1, 0:2], NDCube, mask_cube[1, 1, 0:2], _wcs_slicer(
        wt, [True, False, False, False], (1, 1, slice(0, 2, None))), np.array([2, 3]), DimensionPair(
        shape=u.Quantity((2,), unit=u.pix), axis_types=['TIME'])),
    (cube[1, 1, :], NDCube, mask_cube[1, 1, :], _wcs_slicer(
        wt, [True, False, False, False], (1, 1, slice(0, 2, None))), np.array([2, 3]), DimensionPair(shape=u.Quantity((4,), unit=u.pix
                                                                                                                        ), axis_types=['TIME'])),
])
def test_slicing_third_axis(test_input, expected, mask, wcs, uncertainty, dimensions):
    assert isinstance(test_input, expected)
    assert np.all(test_input.mask == mask)
    assert_wcs_are_equal(test_input.wcs, wcs[0])
    assert test_input.missing_axis == wcs[1]
    assert test_input.uncertainty.array.shape == uncertainty.shape
    assert test_input.dimensions[1] == dimensions[1]
    assert np.all(test_input.dimensions[0].value == dimensions[0].value)
    assert test_input.dimensions[0].unit == dimensions[0].unit


@pytest.mark.parametrize("test_input,expected", [
    (cubem[1].pixel_to_world([u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix)])[0], wm.all_pix2world(
        u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix), wm.wcs.crpix[2]-1, 0)[-2]),
    (cubem[1].pixel_to_world([u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix)])[
     1], wm.all_pix2world(u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix), wm.wcs.crpix[2]-1, 0)[0]),
    (cubem[0:2].pixel_to_world([u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix)])[
     0], wm.all_pix2world(u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix), 0)[-1]),
    (cubem[0:2].pixel_to_world([u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix)])[
     1], wm.all_pix2world(u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix), 0)[1]),
    (cubem[0:2].pixel_to_world([u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix)])[
     2], wm.all_pix2world(u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix), 0)[0]),
    (cube[1].pixel_to_world([u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix)])[0], wt.all_pix2world(
        u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix), wt.wcs.crpix[2]-1, wt.wcs.crpix[3]-1, 0)[1]),
    (cube[1].pixel_to_world([u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix)])[1], wt.all_pix2world(
        u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix), wt.wcs.crpix[2]-1, wt.wcs.crpix[3]-1, 0)[0]),
    (cube[0:2].pixel_to_world([u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix)])[
     0], wt.all_pix2world(u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix), wt.wcs.crpix[3]-1, 0)[2]),
    (cube[0:2].pixel_to_world([u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix)])[
     1], wt.all_pix2world(u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix), wt.wcs.crpix[3]-1, 0)[1]),
    (cube[0:2].pixel_to_world([u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix)])[
     2], wt.all_pix2world(u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix), u.Quantity(np.arange(4), unit=u.pix), wt.wcs.crpix[3]-1, 0)[0]),
])
def test_pixel_to_world(test_input, expected):
    assert np.all(test_input.value == expected)


@pytest.mark.parametrize("test_input,expected", [
    (cubem[1].world_to_pixel([u.Quantity(np.arange(4), unit=u.deg), u.Quantity(np.arange(4), unit=u.m)], origin=1)[
     0], wm.all_world2pix(u.Quantity(np.arange(4), unit=u.deg), u.Quantity(np.arange(4), unit=u.m), wm.wcs.crpix[2]-1, 1)[1]),
    (cubem[1].world_to_pixel([u.Quantity(np.arange(4), unit=u.deg), u.Quantity(np.arange(4), unit=u.m)])[
     1], wm.all_world2pix(u.Quantity(np.arange(4), unit=u.deg), u.Quantity(np.arange(4), unit=u.m), wm.wcs.crpix[2]-1, 0)[0]),
    (cubem[0:2].world_to_pixel([u.Quantity(np.arange(4), unit=u.deg), u.Quantity(np.arange(4), unit=u.deg), u.Quantity(np.arange(4), unit=u.m)])[
     0], wm.all_world2pix(u.Quantity(np.arange(4), unit=u.deg), u.Quantity(np.arange(4), unit=u.deg), u.Quantity(np.arange(4), unit=u.m), 0)[-1]),
    (cubem[0:2].world_to_pixel([u.Quantity(np.arange(4), unit=u.deg), u.Quantity(np.arange(4), unit=u.deg), u.Quantity(np.arange(4), unit=u.m)])[
     1], wm.all_world2pix(u.Quantity(np.arange(4), unit=u.deg), u.Quantity(np.arange(4), unit=u.deg), u.Quantity(np.arange(4), unit=u.m), 0)[1]),
    (cubem[0:2].world_to_pixel([u.Quantity(np.arange(4), unit=u.deg), u.Quantity(np.arange(4), unit=u.deg), u.Quantity(np.arange(4), unit=u.m)])[
     2], wm.all_world2pix(u.Quantity(np.arange(4), unit=u.deg), u.Quantity(np.arange(4), unit=u.deg), u.Quantity(np.arange(4), unit=u.m), 0)[0]),
    (cube[1].world_to_pixel([u.Quantity(np.arange(4), unit=u.m), u.Quantity(np.arange(4), unit=u.min)])[0], wt.all_world2pix(
        u.Quantity(np.arange(4), unit=u.m), u.Quantity(np.arange(4), unit=u.min), wt.wcs.crpix[2]-1, wt.wcs.crpix[3]-1, 0)[1]),
    (cube[1].world_to_pixel([u.Quantity(np.arange(4), unit=u.m), u.Quantity(np.arange(4), unit=u.min)])[1], wt.all_world2pix(
        u.Quantity(np.arange(4), unit=u.m), u.Quantity(np.arange(4), unit=u.min), wt.wcs.crpix[2]-1, wt.wcs.crpix[3]-1, 0)[0]),
    (cube[0:2].world_to_pixel([u.Quantity(np.arange(4), unit=u.deg), u.Quantity(np.arange(4), unit=u.m), u.Quantity(np.arange(4), unit=u.min)])[
     0], wt.all_world2pix(u.Quantity(np.arange(4), unit=u.deg), u.Quantity(np.arange(4), unit=u.m), u.Quantity(np.arange(4), unit=u.min), wt.wcs.crpix[3]-1, 0)[2]),
    (cube[0:2].world_to_pixel([u.Quantity(np.arange(4), unit=u.deg), u.Quantity(np.arange(4), unit=u.m), u.Quantity(np.arange(4), unit=u.min)])[
     1], wt.all_world2pix(u.Quantity(np.arange(4), unit=u.deg), u.Quantity(np.arange(4), unit=u.m), u.Quantity(np.arange(4), unit=u.min), wt.wcs.crpix[3]-1, 0)[1]),
    (cube[0:2].world_to_pixel([u.Quantity(np.arange(4), unit=u.deg), u.Quantity(np.arange(4), unit=u.m), u.Quantity(np.arange(4), unit=u.min)])[
     2], wt.all_world2pix(u.Quantity(np.arange(4), unit=u.deg), u.Quantity(np.arange(4), unit=u.m), u.Quantity(np.arange(4), unit=u.min), wt.wcs.crpix[3]-1, 0)[0]),
])
def test_world_to_pixel(test_input, expected):
    assert np.allclose(test_input.value, expected)
