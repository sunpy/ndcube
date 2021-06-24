
"""
Helpers for testing ndcube.
"""
import unittest
from pathlib import Path
from functools import wraps

import astropy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
import sunpy
from astropy.wcs.wcsapi.fitswcs import SlicedFITSWCS
from astropy.wcs.wcsapi.low_level_api import BaseLowLevelWCS
from astropy.wcs.wcsapi.wrappers.sliced_wcs import sanitize_slices
from numpy.testing import assert_equal

from ndcube import NDCube, NDCubeSequence

__all__ = ['figure_test',
           'get_hash_library_name',
           'assert_extra_coords_equal',
           'assert_metas_equal',
           'assert_cubes_equal',
           'assert_cubesequences_equal',
           'assert_wcs_are_equal']


def get_hash_library_name():
    """
    Generate the hash library name for this env.
    """
    ft2_version = f"{mpl.ft2font.__freetype_version__.replace('.', '')}"
    sunpy_version = "dev" if "dev" in sunpy.__version__ else sunpy.__version__.replace('.', '')
    mpl_version = "dev" if "+" in mpl.__version__ else mpl.__version__.replace('.', '')
    astropy_version = "dev" if "dev" in astropy.__version__ else astropy.__version__.replace('.', '')
    return f"figure_hashes_mpl_{mpl_version}_ft_{ft2_version}_astropy_{astropy_version}_sunpy_{sunpy_version}.json"


def figure_test(test_function):
    """
    A decorator for a test that verifies the hash of the current figure or the
    returned figure, with the name of the test function as the hash identifier
    in the library. A PNG is also created in the 'result_image' directory,
    which is created on the current path.

    All such decorated tests are marked with `pytest.mark.mpl_image` for convenient filtering.
    """
    hash_library_name = get_hash_library_name()
    hash_library_file = Path(__file__).parent / ".." / "visualization" / "tests"  / hash_library_name

    @pytest.mark.remote_data
    @pytest.mark.mpl_image_compare(hash_library=hash_library_file.resolve(),
                                   savefig_kwargs={'metadata': {'Software': None}},
                                   style='default')
    @wraps(test_function)
    def test_wrapper(*args, **kwargs):
        ret = test_function(*args, **kwargs)
        if ret is None:
            ret = plt.gcf()
        return ret
    return test_wrapper


def assert_extra_coords_equal(test_input, extra_coords):
    assert test_input.keys() == extra_coords.keys()
    for key in list(test_input.keys()):
        assert test_input[key]['axis'] == extra_coords[key]['axis']
        assert (test_input[key]['value'] == extra_coords[key]['value']).all()


def assert_metas_equal(test_input, expected_output):
    if not (test_input is None and expected_output is None):
        assert test_input.keys() == expected_output.keys()
        for key in list(test_input.keys()):
            assert test_input[key] == expected_output[key]


def assert_cubes_equal(test_input, expected_cube):
    unittest.TestCase()
    assert isinstance(test_input, type(expected_cube))
    assert np.all(test_input.mask == expected_cube.mask)
    assert_wcs_are_equal(test_input.wcs, expected_cube.wcs)
    if test_input.uncertainty:
        assert test_input.uncertainty.array.shape == expected_cube.uncertainty.array.shape
    assert all(test_input.dimensions.value == expected_cube.dimensions.value)
    assert test_input.dimensions.unit == expected_cube.dimensions.unit
    if type(test_input.extra_coords) is not type(expected_cube.extra_coords):
        raise AssertionError("NDCube extra_coords not of same type: {0} != {1}".format(
            type(test_input.extra_coords), type(expected_cube.extra_coords)))
    if test_input.extra_coords is not None:
        assert_extra_coords_equal(test_input.extra_coords, expected_cube.extra_coords)


def assert_cubesequences_equal(test_input, expected_sequence):
    assert isinstance(test_input, type(expected_sequence))
    assert_metas_equal(test_input.meta, expected_sequence.meta)
    assert test_input._common_axis == expected_sequence._common_axis
    for i, cube in enumerate(test_input.data):
        assert_cubes_equal(cube, expected_sequence.data[i])


def assert_wcs_are_equal(wcs1, wcs2):
    """
    Assert function for testing two wcs object.

    Used in testing NDCube.
    Also checks if both the wcs objects are instance
    of `SlicedLowLevelWCS`
    """

    if not isinstance(wcs1, BaseLowLevelWCS):
        wcs1 = wcs1.low_level_wcs
    if not isinstance(wcs2, BaseLowLevelWCS):
        wcs2 = wcs2.low_level_wcs
    # Check the APE14 attributes of both the WCS
    assert wcs1.pixel_n_dim == wcs2.pixel_n_dim
    assert wcs1.world_n_dim == wcs2.world_n_dim
    assert wcs1.array_shape == wcs2.array_shape
    assert wcs1.pixel_shape == wcs2.pixel_shape
    assert wcs1.world_axis_physical_types == wcs2.world_axis_physical_types
    assert wcs1.world_axis_units == wcs2.world_axis_units
    assert_equal(wcs1.axis_correlation_matrix, wcs2.axis_correlation_matrix)
    assert wcs1.pixel_bounds == wcs2.pixel_bounds


def create_sliced_wcs(wcs, item, dim):
    """
    Creates a sliced `SlicedFITSWCS` object from the given slice item
    """

    # Sanitize the slices
    item = sanitize_slices(item, dim)
    return SlicedFITSWCS(wcs, item)


def assert_collections_equal(collection1, collection2):
    assert collection1.keys() == collection2.keys()
    assert collection1.aligned_axes == collection2.aligned_axes
    for cube1, cube2 in zip(collection1.values(), collection2.values()):
        # Check cubes are same type.
        assert type(cube1) is type(cube2)
        if isinstance(cube1, NDCube):
            assert_cubes_equal(cube1, cube2)
        elif isinstance(cube1, NDCubeSequence):
            assert_cubesequences_equal(cube1, cube2)
        else:
            raise TypeError("Unsupported Type in NDCollection: {0}".format(type(cube1)))
