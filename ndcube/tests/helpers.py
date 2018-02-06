# -*- coding: utf-8 -*-

"""
Helpers for testing ndcube.
"""

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
    assert type(test_input) == type(expected_cube)
    assert np.all(test_input.mask == expected_cube.mask)
    utils.wcs.assert_wcs_are_equal(test_input.wcs, expected_cube.wcs)
    assert test_input.missing_axis == expected_cube.missing_axis
    assert test_input.uncertainty.array.shape == expected_cube.uncertainty.array.shape
    assert test_input.dimensions[1] == expected_cube.dimensions[1]
    assert np.all(test_input.dimensions[0].value == expected_cube.dimensions[0].value)
    assert test_input.dimensions[0].unit == expected_cube.dimensions[0].unit
    assert_extra_coords_equal(test_input._extra_coords, expected_cube._extra_coords)


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
