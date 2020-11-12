import datetime
import unittest
from collections import namedtuple
from copy import deepcopy

import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS

from ndcube import NDCube, NDCubeSequence

# sample data for tests
# TODO: use a fixture reading from a test file. file TBD.
data = np.array([[[1, 2, 3, 4], [2, 4, 5, 3], [0, -1, 2, 3]],
                 [[2, 4, 5, 1], [10, 5, 2, 2], [10, 3, 3, 0]]])

data2 = np.array([[[11, 22, 33, 44], [22, 44, 55, 33], [0, -1, 22, 33]],
                  [[22, 44, 55, 11], [10, 55, 22, 22], [10, 33, 33, 0]]])

ht = {'CTYPE3': 'UNKNOWN', 'CUNIT3': 'deg', 'CDELT3': 0.5, 'CRPIX3': 0, 'CRVAL3': 0, 'NAXIS3': 2,
      'CTYPE2': 'WAVE    ', 'CUNIT2': 'Angstrom', 'CDELT2': 0.2, 'CRPIX2': 0, 'CRVAL2': 0,
      'NAXIS2': 3,
      'CTYPE1': 'TIME    ', 'CUNIT1': 'min', 'CDELT1': 0.4, 'CRPIX1': 0, 'CRVAL1': 0, 'NAXIS1': 4}

hm = {
    'CTYPE1': 'WAVE    ', 'CUNIT1': 'Angstrom', 'CDELT1': 0.2, 'CRPIX1': 0, 'CRVAL1': 10,
    'NAXIS1': 4,
    'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5, 'NAXIS2': 3,
    'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 2,
}

wt = WCS(header=ht, naxis=3)
wm = WCS(header=hm, naxis=3)

cube1 = NDCube(data, wt, extra_coords=[
    ('pix', 0, u.Quantity(range(data.shape[0]), unit=u.pix)),
    ('distance', None, u.Quantity(0, unit=u.cm)),
    ('time', None, datetime.datetime(2000, 1, 1, 0, 0))])

cube2 = NDCube(data, wm, extra_coords=[
    ('pix', 0, u.Quantity(np.arange(1, data.shape[0] + 1), unit=u.pix) +
     cube1.extra_coords['pix']['value'][-1]),
    ('distance', None, u.Quantity(1, unit=u.cm)),
    ('time', None, datetime.datetime(2000, 1, 1, 0, 1))])

cube3 = NDCube(data2, wt, extra_coords=[
    ('pix', 0, u.Quantity(np.arange(1, data2.shape[0] + 1), unit=u.pix) +
     cube2.extra_coords['pix']['value'][-1]),
    ('distance', None, u.Quantity(2, unit=u.cm)),
    ('time', None, datetime.datetime(2000, 1, 1, 0, 2))])

cube4 = NDCube(data2, wm, extra_coords=[
    ('pix', 0, u.Quantity(np.arange(1, data2.shape[0] + 1), unit=u.pix) +
     cube3.extra_coords['pix']['value'][-1]),
    ('distance', None, u.Quantity(3, unit=u.cm)),
    ('time', None, datetime.datetime(2000, 1, 1, 0, 3))])

cube2_no_no = NDCube(data, wm, extra_coords=[
    ('pix', 0, u.Quantity(np.arange(1, data.shape[0] + 1), unit=u.pix) +
     cube1.extra_coords['pix']['value'][-1]),
    ('time', None, datetime.datetime(2000, 1, 1, 0, 1))])

cube3_no_time = NDCube(data2, wt, extra_coords=[
    ('pix', 0, u.Quantity(np.arange(1, data2.shape[0] + 1), unit=u.pix) +
     cube2.extra_coords['pix']['value'][-1]),
    ('distance', None, u.Quantity(2, unit=u.cm))])

cube3_diff_compatible_unit = NDCube(
    data2, wt, extra_coords=[
        ('pix', 0, u.Quantity(np.arange(data2.shape[0]), unit=u.pix) +
         cube2.extra_coords['pix']['value'][-1]),
        ('distance', None, u.Quantity(2, unit=u.cm).to('m')),
        ('time', None, datetime.datetime(2000, 1, 1, 0, 2))])

cube3_diff_incompatible_unit = NDCube(
    data2, wt, extra_coords=[
        ('pix', 0, u.Quantity(np.arange(data2.shape[0]), unit=u.pix) +
         cube2.extra_coords['pix']['value'][-1]),
        ('distance', None, u.Quantity(2, unit=u.s)),
        ('time', None, datetime.datetime(2000, 1, 1, 0, 2))])

cube1_time_common = NDCube(
    data, wt,
    extra_coords=[('time', 1, [datetime.datetime(2000, 1, 1) + datetime.timedelta(minutes=i)
                               for i in range(data.shape[1])])])

cube2_time_common = NDCube(data, wm, extra_coords=[
    ('time', 1,
     [cube1_time_common.extra_coords["time"]["value"][-1] + datetime.timedelta(minutes=i)
      for i in range(1, data.shape[1] + 1)])])

cube1_no_extra_coords = NDCube(data, wt)
cube3_no_extra_coords = NDCube(data2, wt)

seq = NDCubeSequence([cube1, cube2, cube3, cube4], common_axis=0)
seq_bad_common_axis = NDCubeSequence([cube1, cube2, cube3, cube4], common_axis=None)
seq_time_common = NDCubeSequence([cube1_time_common, cube2_time_common], common_axis=1)
seq1 = NDCubeSequence([cube1, cube2, cube3, cube4])
seq2 = NDCubeSequence([cube1, cube2_no_no, cube3_no_time, cube4])
seq3 = NDCubeSequence([cube1, cube2, cube3_diff_compatible_unit, cube4])
seq4 = NDCubeSequence([cube1, cube2, cube3_diff_incompatible_unit, cube4])
seq_no_extra_coords = NDCubeSequence([cube1_no_extra_coords, cube3_no_extra_coords], common_axis=0)

nan_extra_coord = u.Quantity(range(4), unit=u.cm)
nan_extra_coord.value[1] = np.nan
nan_time_extra_coord = np.array([datetime.datetime(2000, 1, 1) + datetime.timedelta(minutes=i)
                                 for i in range(len(seq.data))])
nan_time_extra_coord[2] = np.nan


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
    print(sliced_sequence[0].dimensions, expected_cube0_dims, item, tuple_item)
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
    orig_cube_dims = ndc.data[tuple_item[0]].dimensions
    expected_cube_dims = derive_sliced_cube_dims(ndc.data[tuple_item[0]].dimensions, tuple_item)
    assert isinstance(cube, NDCube)
    print(tuple_item, orig_cube_dims, cube.dimensions, expected_cube_dims, orig_cube_dims[(slice(0, 1),)])
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
            ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[1:7],
                (3 * u.pix, 2 * u.pix, [2., 3., 1.] * u.pix, 4 * u.pix)),
            ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[1:7, 0],
                (3 * u.pix, [2., 3., 1.] * u.pix, 4 * u.pix)),
            ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[2:4],
                (2 * u.pix, 2 * u.pix, 1 * u.pix, 4 * u.pix)),
            ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[0:6],
                (2 * u.pix, 2 * u.pix, 3 * u.pix, 4 * u.pix)),
            ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[0:6, :, 0],
                (3 * u.pix, 2 * u.pix, 4 * u.pix)),
            ("ndcubesequence_4c_ln_lt_l_cax1", np.s_[0::, 0, 0],
                (4 * u.pix, 2 * u.pix))
        ),
        indirect=("ndc",))
def test_index_as_cube(ndc, item, expected_dimensions):
    sliced_sequence = ndc.index_as_cube[item]
    sliced_dims = sliced_sequence.dimensions
    for dim, expected_dim in zip(sliced_dims, expected_dimensions):
        (dim == expected_dim).all()


@pytest.mark.parametrize("ndc, axis, expected_dimensions", 
        (
            ("ndcubesequence_4c_ln_lt_l", 0, (8 * u.pix, 3 * u.pix, 4 * u.pix)),
            ("ndcubesequence_4c_ln_lt_l_cax1", 1, (12 * u.pix, 2 * u.pix, 4 * u.pix)),
            ("ndcubesequence_4c_ln_lt_l", 2, (16 * u.pix, 2 * u.pix, 3 * u.pix))
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
            ("ndcubesequence_4c_ln_lt_l_cax1", (4 * u.pix, 2. * u.pix, 3. * u.pix, 4. * u.pix)),
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
    print(ndc.cube_like_dimensions, expected_dimensions)
    assert (ndc.cube_like_dimensions == expected_dimensions).all()


@pytest.mark.parametrize("test_input", [(seq_bad_common_axis)])
def test_cube_like_dimensions_error(test_input):
    with pytest.raises(TypeError):
        seq_bad_common_axis.cube_like_dimensions



@pytest.mark.parametrize(
    "test_input,expected",
    [(seq, {'pix': u.Quantity([0., 1., 2., 3., 4., 5., 6., 7.], unit=u.pix)}),
     (seq_time_common,
      {'time': np.array([datetime.datetime(2000, 1, 1, 0, 0), datetime.datetime(2000, 1, 1, 0, 1),
                         datetime.datetime(2000, 1, 1, 0, 2), datetime.datetime(2000, 1, 1, 0, 3),
                         datetime.datetime(2000, 1, 1, 0, 4), datetime.datetime(2000, 1, 1, 0, 5)],
                        dtype=object)})])
def test_common_axis_extra_coords(test_input, expected):
    output = test_input.common_axis_extra_coords
    assert output.keys() == expected.keys()
    for key in output.keys():
        try:
            assert output[key] == expected[key]
        except ValueError:
            assert (output[key] == expected[key]).all()


@pytest.mark.parametrize("test_input", [(seq_no_extra_coords)])
def test_no_common_axis_extra_coords(test_input):
    assert seq_no_extra_coords.sequence_axis_extra_coords is None


@pytest.mark.parametrize(
    "test_input,expected",
    [(seq,
      {'distance': u.Quantity(range(4), unit=u.cm),
       'time': np.array([datetime.datetime(2000, 1, 1) + datetime.timedelta(minutes=i)
                         for i in range(len(seq.data))])}),
     (seq2,
      {'distance': nan_extra_coord,
       'time': nan_time_extra_coord}),
     (seq3,
      {'distance': u.Quantity(range(4), unit=u.cm),
       'time': np.array([datetime.datetime(2000, 1, 1) + datetime.timedelta(minutes=i)
                         for i in range(len(seq.data))])})])
def test_sequence_axis_extra_coords(test_input, expected):
    output = test_input.sequence_axis_extra_coords
    assert output.keys() == expected.keys()
    for key in output.keys():
        if isinstance(output[key], u.Quantity):
            assert output[key].unit == expected[key].unit
            np.testing.assert_array_almost_equal(output[key].value, expected[key].value)
        else:
            # For non-Quantities, must check element by element due to
            # possible mixture of NaN and non-number elements on
            # arrays, e.g. datetime does not work with
            # np.testing.assert_array_almost_equal().
            for i, output_value in enumerate(output[key]):
                if isinstance(output_value, float):
                    # Check if output is nan, expected is no and vice versa.
                    if not isinstance(expected[key][i], float):
                        raise AssertionError("{} != {}".format(output_value, expected[key][i]))
                    elif np.logical_xor(np.isnan(output_value), np.isnan(expected[key][i])):
                        raise AssertionError("{0} != {1}", format(output_value, expected[key][i]))
                    # Else assert they are equal if they are both not NaN.
                    elif not np.isnan(output_value):
                        assert output_value == expected[key][i]
                # Else, is output is not a float, assert it equals expected.
                else:
                    assert output_value == expected[key][i]


@pytest.mark.parametrize("test_input", [(seq_no_extra_coords)])
def test_no_sequence_axis_extra_coords(test_input):
    assert seq_no_extra_coords.sequence_axis_extra_coords is None


@pytest.mark.parametrize("test_input", [(seq4)])
def test_sequence_axis_extra_coords_incompatible_unit_error(test_input):
    with pytest.raises(u.UnitConversionError):
        test_input.sequence_axis_extra_coords
