import unittest

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time, TimeDelta

from ndcube import NDCube, NDCubeSequence


def derive_sliced_cube_dims(orig_cube_dims, tuple_item):
    expected_cube_dims = list(orig_cube_dims)
    len_cube_item = len(tuple_item) - 1
    if len_cube_item > 0:
        cube_item = tuple_item[1:]
        for i, s in zip(np.arange(len_cube_item)[::-1], cube_item[::-1]):
            if isinstance(s, int):
                del expected_cube_dims[i]
            else:
                expected_cube_dims[i] = float(s.stop - s.start) * u.pix
    expected_cube_dims *= u.pix
    return expected_cube_dims


@pytest.mark.parametrize("ndc, item",
                         (
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[0:1], ),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[0:1, 0:2]),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[0:1, 1]),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[1:3, 1, 0:2])
                         ),
                         indirect=('ndc',))
def test_slice_sequence_axis(ndc, item):
    # Calculate expected dimensions of cubes with sequence after slicing.
    tuple_item = item if isinstance(item, tuple) else (item,)
    expected_cube0_dims = derive_sliced_cube_dims(ndc.data[tuple_item[0]][0].dimensions, tuple_item)
    # Assert output is as expected.
    sliced_sequence = ndc[item]
    assert isinstance(sliced_sequence, NDCubeSequence)
    assert int(sliced_sequence.dimensions[0].value) == tuple_item[0].stop - tuple_item[0].start
    assert (sliced_sequence[0].dimensions == expected_cube0_dims).all()


@pytest.mark.parametrize("ndc, item",
                         (
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[0]),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[1, 0:1]),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[2, 1]),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[3, 1, 0:2])
                         ),
                         indirect=("ndc",))
def test_extract_ndcube(ndc, item):
    cube = ndc[item]
    tuple_item = item if isinstance(item, tuple) else (item,)
    expected_cube_dims = derive_sliced_cube_dims(ndc.data[tuple_item[0]].dimensions, tuple_item)
    assert isinstance(cube, NDCube)
    assert (cube.dimensions == expected_cube_dims).all()


@pytest.mark.parametrize("ndc, item, expected_common_axis",
                         (
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[:, 0], 0),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[:, 0:1, 0:2], 1),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[:, :, :, 1], 1),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[:, :, 0], None)
                         ),
                         indirect=("ndc",))
def test_slice_common_axis(ndc, item, expected_common_axis):
    sliced_sequence = ndc[item]
    assert sliced_sequence._common_axis == expected_common_axis


@pytest.mark.parametrize("ndc, item, expected_dimensions",
                         (
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[1:7], (3 * u.pix,
                                                                             2 * u.pix,
                                                                             [2., 3., 1.] * u.pix,
                                                                             4 * u.pix)),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[1:7, 0], (3 * u.pix,
                                                                                [2., 3., 1.] * u.pix,
                                                                                4 * u.pix)),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[2:4], (2 * u.pix,
                                                                             2 * u.pix,
                                                                             1 * u.pix,
                                                                             4 * u.pix)),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[0:6], (2 * u.pix,
                                                                             2 * u.pix,
                                                                             3 * u.pix,
                                                                             4 * u.pix)),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[0:6, :, 0], (3 * u.pix,
                                                                                   2 * u.pix,
                                                                                   4 * u.pix)),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[0::, 0, 0], (4 * u.pix,
                                                                                   2 * u.pix))
                         ),
                         indirect=("ndc",))
def test_index_as_cube(ndc, item, expected_dimensions):
    sliced_sequence = ndc.index_as_cube[item]
    sliced_dims = sliced_sequence.dimensions
    for dim, expected_dim in zip(sliced_dims, expected_dimensions):
        (dim == expected_dim).all()


@pytest.mark.parametrize("ndc, axis, expected_dimensions",
                         (
                             ("ndcubesequence_4c_ln_lt_l", 0, (8 * u.pix,
                                                               3 * u.pix,
                                                               4 * u.pix)),
                             ("ndcubesequence_4c_ln_lt_l_cax1", 1, (12 * u.pix,
                                                                    2 * u.pix,
                                                                    4 * u.pix)),
                             ("ndcubesequence_4c_ln_lt_l", 2, (16 * u.pix,
                                                               2 * u.pix,
                                                               3 * u.pix))
                         ),
                         indirect=("ndc",))
def test_explode_along_axis(ndc, axis, expected_dimensions):
    exploded_sequence = ndc.explode_along_axis(axis)
    assert exploded_sequence.dimensions == expected_dimensions
    assert exploded_sequence._common_axis is None


@pytest.mark.parametrize("ndc, axis", (("ndcubesequence_4c_ln_lt_l_cax1", 0),), indirect=("ndc",))
def test_explode_along_axis_error(ndc, axis):
    with pytest.raises(ValueError):
        ndc.explode_along_axis(axis)


@pytest.mark.parametrize("ndc, expected_dimensions",
                         (
                             ("ndcubesequence_4c_ln_lt_l_cax1", (4 * u.pix,
                                                                 2. * u.pix,
                                                                 3. * u.pix,
                                                                 4. * u.pix)),
                         ),
                         indirect=("ndc",))
def test_dimensions(ndc, expected_dimensions):
    unit_tester = unittest.TestCase()
    unit_tester.assertEqual(ndc.dimensions, expected_dimensions)


@pytest.mark.parametrize("ndc, expected_dimensions",
                         (
                             ("ndcubesequence_4c_ln_lt_l_cax1", [2., 12, 4] * u.pix),
                         ),
                         indirect=("ndc",))
def test_cube_like_dimensions(ndc, expected_dimensions):
    assert (ndc.cube_like_dimensions == expected_dimensions).all()


@pytest.mark.parametrize("ndc", (("ndcubesequence_4c_ln_lt_l",)), indirect=("ndc",))
def test_cube_like_dimensions_error(ndc):
    with pytest.raises(TypeError):
        ndc.cube_like_dimensions


@pytest.mark.parametrize("ndc", (("ndcubesequence_3c_l_ln_lt_cax1",)), indirect=("ndc",))
def test_common_axis_coords(ndc):
    common_axis_length = int(ndc.cube_like_dimensions[ndc._common_axis].value)
    base_time = Time('2000-01-01', format='fits', scale='utc')
    expected = {'time': [base_time + TimeDelta(60 * i, format='sec')
                         for i in range(common_axis_length)]}
    output = ndc.common_axis_coords


@pytest.mark.parametrize("ndc", (("ndcubesequence_4c_ln_lt_l",)), indirect=("ndc",))
def test_sequence_axis_coords(ndc):
    expected = {'global coord': [None, <Quantity 0. pix>, None],
                'distance': [<Quantity 1. um>, <Quantity 2. um>, None]}
    output = ndc.sequence_axis_coords
    assert output == expected
