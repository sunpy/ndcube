# -*- coding: utf-8 -*-

"""
Helpers for testing ndcube.
"""
import unittest

import numpy as np

from ndcube import utils
from astropy.io import fits

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

def comparerecords(a, b):
    """
    Compare two record arrays
    Does this field by field, using approximation testing for float columns
    (Complex not yet handled.)
    Column names not compared, but column types and sizes are.
    
    Note: This helper method has been taken from `astropy.io.fits.tests.test_table`
    """

    nfieldsa = len(a.dtype.names)
    nfieldsb = len(b.dtype.names)
    if nfieldsa != nfieldsb:
        print("number of fields don't match")
        return False
    for i in range(nfieldsa):
        fielda = a.field(i)
        fieldb = b.field(i)
        if fielda.dtype.char == 'S':
            fielda = decode_ascii(fielda)
        if fieldb.dtype.char == 'S':
            fieldb = decode_ascii(fieldb)
        if (not isinstance(fielda, type(fieldb)) and not
            isinstance(fieldb, type(fielda))):
            print("type(fielda): ", type(fielda), " fielda: ", fielda)
            print("type(fieldb): ", type(fieldb), " fieldb: ", fieldb)
            print('field {0} type differs'.format(i))
            return False
        if len(fielda) and isinstance(fielda[0], np.floating):
            if not comparefloats(fielda, fieldb):
                print("fielda: ", fielda)
                print("fieldb: ", fieldb)
                print('field {0} differs'.format(i))
                return False
        elif (isinstance(fielda, fits.column._VLF) or
              isinstance(fieldb, fits.column._VLF)):
            for row in range(len(fielda)):
                if np.any(fielda[row] != fieldb[row]):
                    print('fielda[{0}]: {1}'.format(row, fielda[row]))
                    print('fieldb[{0}]: {1}'.format(row, fieldb[row]))
                    print('field {0} differs in row {1}'.format(i, row))
        else:
            if np.any(fielda != fieldb):
                print("fielda: ", fielda)
                print("fieldb: ", fieldb)
                print('field {0} differs'.format(i))
                return False
    return True