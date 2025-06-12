
"""
Helpers for testing ndcube.
"""
from pathlib import Path
from functools import wraps

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_animators
import numpy as np
import pytest
from numpy.testing import assert_equal

import astropy
import astropy.units as u
from astropy.wcs.wcsapi import BaseHighLevelWCS
from astropy.wcs.wcsapi.fitswcs import SlicedFITSWCS
from astropy.wcs.wcsapi.low_level_api import BaseLowLevelWCS
from astropy.wcs.wcsapi.wrappers.sliced_wcs import sanitize_slices

from ndcube import NDCube, NDCubeSequence
from ndcube.meta import NDMeta

__all__ = [
    'assert_cubes_equal',
    'assert_cubesequences_equal',
    'assert_extra_coords_equal',
    'assert_metas_equal',
    'assert_wcs_are_equal',
    'figure_test',
    'get_hash_library_name',
]


def get_hash_library_name():
    """
    Generate the hash library name for this env.
    """
    ft2_version = f"{mpl.ft2font.__freetype_version__.replace('.', '')}"
    animators_version = "dev" if (("dev" in mpl_animators.__version__) or ("rc" in mpl_animators.__version__)) else mpl_animators.__version__.replace('.', '')
    mpl_version = "dev" if (("dev" in mpl.__version__) or ("rc" in mpl.__version__)) else mpl.__version__.replace('.', '')
    astropy_version = "dev" if (("dev" in astropy.__version__) or ("rc" in astropy.__version__)) else astropy.__version__.replace('.', '')
    return f"figure_hashes_mpl_{mpl_version}_ft_{ft2_version}_astropy_{astropy_version}_animators_{animators_version}.json"


def figure_test(test_function):
    """
    A decorator for a test that verifies the hash of the current figure or the
    returned figure, with the name of the test function as the hash identifier
    in the library. A PNG is also created in the 'result_image' directory,
    which is created on the current path.

    All such decorated tests are marked with ``pytest.mark.mpl_image`` for convenient filtering.
    """
    hash_library_name = get_hash_library_name()
    hash_library_file = Path(__file__).parent / ".." / "visualization" / "tests" / hash_library_name

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
    assert set(test_input.keys()) == set(extra_coords.keys())
    if extra_coords._lookup_tables is None:
        assert test_input._lookup_tables is None
    for ec_idx, key in enumerate(extra_coords.keys()):
        test_idx = np.where(np.asarray(test_input.keys()) == key)[0][0]
        assert test_input.mapping[test_idx] == extra_coords.mapping[ec_idx]
        if extra_coords._lookup_tables is not None:
            test_table = test_input._lookup_tables[test_idx][1].table
            ec_table = extra_coords._lookup_tables[ec_idx][1].table
            if not isinstance(ec_table, tuple):
                test_table = (test_table,)
                ec_table = (ec_table,)
            for test_tab, ec_tab in zip(test_table, ec_table):
                if ec_tab.isscalar:
                    assert test_tab == ec_tab
                else:
                    assert all(test_tab == ec_tab)
    if extra_coords._wcs is None:
        assert test_input._wcs is None
    else:
        assert_wcs_are_equal(test_input._wcs, extra_coords._wcs)


def assert_metas_equal(test_input, expected_output):
    if type(test_input) is not type(expected_output):
        raise AssertionError(
            "input and expected are of different type. "
            f"input: {type(test_input)}; expected: {type(expected_output)}")
    multi_element_msg = "more than one element is ambiguous"
    if isinstance(test_input, NDMeta) and isinstance(expected_output, NDMeta):
        assert test_input.keys() == expected_output.keys()

        if test_input.data_shape is None or expected_output.data_shape is None:
            assert test_input.data_shape == expected_output.data_shape
        else:
            assert np.allclose(test_input.data_shape, expected_output.data_shape)

        for test_value, expected_value in zip(test_input.values(), expected_output.values()):
            try:
                assert test_value == expected_value
            except ValueError as err:  # noqa: PERF203
                if multi_element_msg in err.args[0]:
                    if test_value.dtype.kind in ('S', 'U'):
                        # If the values are strings, we can compare them as arrays.
                        assert np.array_equal(test_value, expected_value)
                    else:
                        assert np.allclose(test_value, expected_value)
        for key in test_input.axes.keys():
            assert all(test_input.axes[key] == expected_output.axes[key])
    else:
        if not (test_input is None and expected_output is None):
            assert test_input.keys() == expected_output.keys()
            for key in list(test_input.keys()):
                assert test_input[key] == expected_output[key]


def assert_cubes_equal(test_input, expected_cube, check_data=True):
    assert isinstance(test_input, type(expected_cube))
    assert np.all(test_input.mask == expected_cube.mask)
    if check_data:
        np.testing.assert_array_equal(test_input.data, expected_cube.data)
    assert_wcs_are_equal(test_input.wcs, expected_cube.wcs)
    if test_input.uncertainty:
        assert test_input.uncertainty.array.shape == expected_cube.uncertainty.array.shape
    assert np.all(test_input.shape == expected_cube.shape)
    assert_metas_equal(test_input.meta, expected_cube.meta)
    if type(test_input.extra_coords) is not type(expected_cube.extra_coords):
        raise AssertionError(f"NDCube extra_coords not of same type: "
                             f"{type(test_input.extra_coords)} != {type(expected_cube.extra_coords)}")
    if test_input.extra_coords is not None:
        assert_extra_coords_equal(test_input.extra_coords, expected_cube.extra_coords)


def assert_cubesequences_equal(test_input, expected_sequence, check_data=True):
    assert isinstance(test_input, type(expected_sequence))
    assert_metas_equal(test_input.meta, expected_sequence.meta)
    assert test_input._common_axis == expected_sequence._common_axis
    for i, cube in enumerate(test_input.data):
        assert_cubes_equal(cube, expected_sequence.data[i], check_data=check_data)


def assert_wcs_are_equal(wcs1, wcs2):
    """
    Assert function for testing two wcs object.

    Used in testing NDCube.
    Also checks if both the wcs objects are instance
    of `~astropy.wcs.wcsapi.SlicedLowLevelWCS`.
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
    if wcs1.pixel_shape is not None:
        random_idx = np.random.randint(wcs1.pixel_shape,size=[10,wcs1.pixel_n_dim])
        # SlicedLowLevelWCS vs BaseHighLevelWCS don't have the same pixel_to_world method
        low_level_wcs1 = wcs1.low_level_wcs if isinstance(wcs1, BaseHighLevelWCS) else wcs1
        low_level_wcs2 = wcs2.low_level_wcs if isinstance(wcs2, BaseHighLevelWCS) else wcs2
        np.testing.assert_array_equal(low_level_wcs1.pixel_to_world_values(*random_idx.T), low_level_wcs2.pixel_to_world_values(*random_idx.T))

def create_sliced_wcs(wcs, item, dim):
    """
    Creates a sliced `SlicedFITSWCS` object from the given slice item
    """

    # Sanitize the slices
    item = sanitize_slices(item, dim)
    return SlicedFITSWCS(wcs, item)


def assert_collections_equal(collection1, collection2, check_data=True):
    assert collection1.keys() == collection2.keys()
    assert collection1.aligned_axes == collection2.aligned_axes
    for cube1, cube2 in zip(collection1.values(), collection2.values()):
        # Check cubes are same type.
        assert type(cube1) is type(cube2)
        if isinstance(cube1, NDCube):
            assert_cubes_equal(cube1, cube2, check_data=check_data)
        elif isinstance(cube1, NDCubeSequence):
            assert_cubesequences_equal(cube1, cube2, check_data=check_data)
        else:
            raise TypeError(f"Unsupported Type in NDCollection: {type(cube1)}")

def ndmeta_et0_pr01(shape):
    return NDMeta({"salutation": "hello",
                   "exposure time": u.Quantity([2.] * shape[0], unit=u.s),
                   "pixel response": (100 * np.ones((shape[0], shape[1]), dtype=float)) * u.percent},
                   axes={"exposure time": 0, "pixel response": (0, 1)}, data_shape=shape)


def ndmeta_et0_pr02(shape):
    return NDMeta({"salutation": "hello",
                   "exposure time": u.Quantity([2.] * shape[0], unit=u.s),
                   "pixel response": (100 * np.ones((shape[0], shape[2]), dtype=float)) * u.percent},
                   axes={"exposure time": 0, "pixel response": (0, 2)}, data_shape=shape)
