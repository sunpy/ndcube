# -*- coding: utf-8 -*-

"""
Helpers for testing ndcube.
"""
import unittest

import numpy as np

from ndcube import utils

__all__ = ['assert_extra_coords_equal',
           'assert_metas_equal',
           'assert_cubes_equal',
           'assert_cubesequences_equal',
           'assert_wcs_are_equal']


def assert_extra_coords_equal(test_input, extra_coords):
    assert test_input.keys() == extra_coords.keys()
    for key in list(test_input.keys()):
        assert test_input[key]['axis'] == extra_coords[key]['axis']
        assert (test_input[key]['value'] == extra_coords[key]['value']).all()


def assert_metas_equal(test_input, expected_output):
    assert test_input.keys() == expected_output.keys()
    for key in list(test_input.keys()):
        assert test_input[key] == expected_output[key]


def assert_cubes_equal(test_input, expected_cube):
    unit_tester = unittest.TestCase()
    assert type(test_input) == type(expected_cube)
    assert np.all(test_input.mask == expected_cube.mask)
    assert_wcs_are_equal(test_input.wcs, expected_cube.wcs)
    assert test_input.missing_axes == expected_cube.missing_axes
    assert test_input.uncertainty.array.shape == expected_cube.uncertainty.array.shape
    assert test_input.world_axis_physical_types == expected_cube.world_axis_physical_types
    assert all(test_input.dimensions.value == expected_cube.dimensions.value)
    assert test_input.dimensions.unit == expected_cube.dimensions.unit
    assert_extra_coords_equal(test_input.extra_coords, expected_cube.extra_coords)


def assert_cubesequences_equal(test_input, expected_sequence):
    assert type(test_input) == type(expected_sequence)
    assert_metas_equal(test_input.meta, expected_sequence.meta)
    assert test_input._common_axis == expected_sequence._common_axis
    for i, cube in enumerate(test_input.data):
        assert_cubes_equal(cube, expected_sequence.data[i])


def assert_wcs_are_equal(wcs1, wcs2):
    """
    Assert function for testing two wcs object.
    Used in testing NDCube.
    """
    assert list(wcs1.wcs.ctype) == list(wcs2.wcs.ctype)
    assert list(wcs1.wcs.crval) == list(wcs2.wcs.crval)
    assert list(wcs1.wcs.crpix) == list(wcs2.wcs.crpix)
    assert list(wcs1.wcs.cdelt) == list(wcs2.wcs.cdelt)
    assert list(wcs1.wcs.cunit) == list(wcs2.wcs.cunit)
    assert wcs1.wcs.naxis == wcs2.wcs.naxis
