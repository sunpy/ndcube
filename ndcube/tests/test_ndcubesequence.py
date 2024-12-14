
import warnings

import numpy as np
import pytest

import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time, TimeDelta

from ndcube import NDCube, NDCubeSequence, NDMeta
from ndcube.tests import helpers


def derive_sliced_cube_dims(orig_cube_dims, tuple_item):
    expected_cube_dims = list(orig_cube_dims)
    len_cube_item = len(tuple_item) - 1
    if len_cube_item > 0:
        cube_item = tuple_item[1:]
        for i, s in zip(np.arange(len_cube_item)[::-1], cube_item[::-1]):
            if isinstance(s, int):
                del expected_cube_dims[i]
            else:
                expected_cube_dims[i] = float(s.stop - s.start)
    return tuple(expected_cube_dims)


@pytest.mark.parametrize(("ndc", "item"),
                         [
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[0:1], ),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[0:1, 0:2]),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[0:1, 1]),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[1:3, 1, 0:2])
                         ],
                         indirect=('ndc',))
def test_slice_sequence_axis(ndc, item):
    # Calculate expected dimensions of cubes with sequence after slicing.
    tuple_item = item if isinstance(item, tuple) else (item,)
    expected_cube0_dims = derive_sliced_cube_dims(ndc.data[tuple_item[0]][0].shape, tuple_item)
    # Assert output is as expected.
    sliced_sequence = ndc[item]
    assert isinstance(sliced_sequence, NDCubeSequence)
    assert int(sliced_sequence.shape[0]) == tuple_item[0].stop - tuple_item[0].start
    assert np.all(sliced_sequence[0].shape == expected_cube0_dims)


@pytest.mark.parametrize(("ndc", "item"),
                         [
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[0]),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[1, 0:1]),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[2, 1]),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[3, 1, 0:2])
                         ],
                         indirect=("ndc",))
def test_extract_ndcube(ndc, item):
    cube = ndc[item]
    tuple_item = item if isinstance(item, tuple) else (item,)
    expected_cube_dims = derive_sliced_cube_dims(ndc.data[tuple_item[0]].shape, tuple_item)
    assert isinstance(cube, NDCube)
    assert np.all(cube.shape == expected_cube_dims)


@pytest.mark.parametrize(("ndc", "item", "expected_common_axis"),
                         [
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[:, 0], 0),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[:, 0:1, 0:2], 1),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[:, :, :, 1], 1),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[:, :, 0], None)
                         ],
                         indirect=("ndc",))
def test_slice_common_axis(ndc, item, expected_common_axis):
    sliced_sequence = ndc[item]
    assert sliced_sequence._common_axis == expected_common_axis


@pytest.mark.parametrize(("ndc", "item", "expected_shape"),
                         [
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[:, 1:7], (3, 2, (2, 3, 1), 4)),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[0, 1:7], (3, (2, 3, 1), 4)),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[:, 2:4], (2, 2, 1, 4)),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[:, 0:6], (2, 2, 3, 4)),
                             ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[0, 0:6], (2, 3, 4)),
                         ],
                         indirect=("ndc",))
def test_index_as_cube(ndc, item, expected_shape):
    assert (ndc.index_as_cube[item].shape == expected_shape)


@pytest.mark.parametrize(("ndc", "axis", "expected_shape"),
                         [
                             ("ndcubesequence_4c_ln_lt_l", 0, (8,
                                                               3,
                                                               4)),
                             ("ndcubesequence_4c_ln_lt_l_cax1", 1, (12,
                                                                    2,
                                                                    4))
                         ],
                         indirect=("ndc",))
def test_explode_along_axis_common_axis_none(ndc, axis, expected_shape):
    exploded_sequence = ndc.explode_along_axis(axis)
    assert np.all(exploded_sequence.shape == expected_shape)
    assert exploded_sequence._common_axis is None


@pytest.mark.parametrize("ndc", (['ndcubesequence_4c_ln_lt_l_cax1']), indirect=("ndc",))
def test_explode_along_axis_common_axis_same(ndc):
    exploded_sequence = ndc.explode_along_axis(2)
    assert exploded_sequence.shape == (16, 2, 3)
    assert exploded_sequence._common_axis == ndc._common_axis


@pytest.mark.parametrize("ndc", (['ndcubesequence_4c_ln_lt_l_cax1']), indirect=("ndc",))
def test_explode_along_axis_common_axis_changed(ndc):
    exploded_sequence = ndc.explode_along_axis(0)
    assert exploded_sequence.shape == (8, 3, 4)
    assert exploded_sequence._common_axis == ndc._common_axis - 1


@pytest.mark.parametrize(("ndc", "expected_shape"),
                         [
                             ("ndcubesequence_4c_ln_lt_l_cax1", (4,
                                                                 2.,
                                                                 3.,
                                                                 4.)),
                         ],
                         indirect=("ndc",))
def test_shape(ndc, expected_shape):
    np.testing.assert_array_equal(ndc.shape, expected_shape)


@pytest.mark.parametrize(("ndc", "expected_shape"),
                         [
                             ("ndcubesequence_4c_ln_lt_l_cax1", [2., 12, 4]),
                         ],
                         indirect=("ndc",))
def test_cube_like_shape(ndc, expected_shape):
    assert np.all(ndc.cube_like_shape == expected_shape)


@pytest.mark.parametrize(("ndc", "expected_dimensions"),
                         [
                             ("ndcubesequence_4c_ln_lt_l_cax1", tuple(u.Quantity(d, unit=u.pix) for d in [2. , 12, 4])),
                         ],
                         indirect=("ndc",))
def test_cube_like_dimensions(ndc, expected_dimensions):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ndc_dim, exp_dim in zip(ndc.cube_like_dimensions, expected_dimensions):
            assert_quantity_allclose(ndc_dim, exp_dim)


@pytest.mark.parametrize("ndc", (["ndcubesequence_4c_ln_lt_l"]), indirect=("ndc",))
def test_cube_like_shape_error(ndc):
    with pytest.raises(TypeError):
        ndc.cube_like_shape


@pytest.mark.parametrize("ndc", (["ndcubesequence_3c_l_ln_lt_cax1"]), indirect=("ndc",))
def test_common_axis_coords(ndc):
    # Construct expected skycoord
    common_coords = [cube.axis_world_coords('lon') for cube in ndc]
    expected_skycoords = []
    for cube_coords in common_coords:
        expected_skycoords += [cube_coords[0][i] for i in range(len(cube_coords[0]))]
    # Construct expected Times
    base_time = Time('2000-01-01', format='fits', scale='utc')
    expected_times = [base_time + TimeDelta(60*i, format='sec') for i in range(15)]
    # Run test function.
    output = ndc.common_axis_coords
    #  Check right number of coords returned.
    assert len(output) == 2
    output_skycoords, output_times = output
    # Check SkyCoords are equal.
    for output_coord, expected_coord in zip(output_skycoords, expected_skycoords):
        assert all(output_coord == expected_coord)
    # Check times are equal
    for output_time, expected_time in zip(output_times, expected_times):
        td = output_time - expected_time
        assert u.allclose(td.to(u.s), 0*u.s, atol=1e-10*u.s)


@pytest.mark.parametrize("ndc", (["ndcubesequence_3c_l_ln_lt_cax1"]), indirect=("ndc",))
def test_sequence_axis_coords(ndc):
    expected = {'distance': [1*u.m, 2*u.m, 3*u.m]}
    output = ndc.sequence_axis_coords
    assert output == expected


def test_crop(ndcubesequence_4c_ln_lt_l):
    seq = ndcubesequence_4c_ln_lt_l
    intervals = seq[0].wcs.array_index_to_world([1, 2], [0, 1], [0, 2])
    lower_corner = [coord[0] for coord in intervals]
    upper_corner = [coord[-1] for coord in intervals]
    expected = seq[:, 1:3, 0:2, 0:3]
    output = seq.crop(lower_corner, upper_corner)
    helpers.assert_cubesequences_equal(output, expected)


def test_crop_by_values(ndcubesequence_4c_ln_lt_l):
    seq = ndcubesequence_4c_ln_lt_l
    intervals = seq[0].wcs.array_index_to_world_values([1, 2], [0, 1], [0, 2])
    units = [u.m, u.deg, u.deg]
    lower_corner = [coord[0] * unit for coord, unit in zip(intervals, units)]
    upper_corner = [coord[-1] * unit for coord, unit in zip(intervals, units)]
    # Ensure some quantities are in units different from each other
    # and those stored in the WCS.
    lower_corner[0] = lower_corner[0].to(units[0])
    lower_corner[-1] = lower_corner[-1].to(units[-1])
    upper_corner[-1] = upper_corner[-1].to(units[-1])
    expected = seq[:, 1:3, 0:2, 0:3]
    output = seq.crop_by_values(lower_corner, upper_corner)
    helpers.assert_cubesequences_equal(output, expected)


def test_slice_meta(ndcubesequence_4c_ln_lt_l_cax1):
    seq = ndcubesequence_4c_ln_lt_l_cax1
    sliced_seq = seq[:, :, 0]
    expected_meta = NDMeta({"salutation": "hello",
                            "exposure time": u.Quantity([2] * 4, unit=u.s),
                            "pixel response": u.Quantity([100] * 4, unit=u.percent)},
                           axes={"exposure time": 0, "pixel response": 0}, data_shape=(4, 2, 4))
    helpers.assert_metas_equal(sliced_seq.meta, expected_meta)
