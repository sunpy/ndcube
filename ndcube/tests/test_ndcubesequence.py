# -*- coding: utf-8 -*-
from collections import namedtuple
import pytest
import datetime
import unittest

import sunpy.map
import numpy as np
import astropy.units as u

from ndcube import NDCube, NDCubeSequence
from ndcube.utils.wcs import WCS


# sample data for tests
# TODO: use a fixture reading from a test file. file TBD.
data = np.array([[[1, 2, 3, 4], [2, 4, 5, 3], [0, -1, 2, 3]],
                 [[2, 4, 5, 1], [10, 5, 2, 2], [10, 3, 3, 0]]])

data2 = np.array([[[11, 22, 33, 44], [22, 44, 55, 33], [0, -1, 22, 33]],
                  [[22, 44, 55, 11], [10, 55, 22, 22], [10, 33, 33, 0]]])

ht = {'CTYPE3': 'HPLT-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.5, 'CRPIX3': 0, 'CRVAL3': 0, 'NAXIS3': 2,
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

cube1 = NDCube(data, wt, missing_axes=[False, False, False, True], extra_coords=[
    ('pix', 0, u.Quantity(range(data.shape[0]), unit=u.pix)),
    ('distance', None, u.Quantity(0, unit=u.cm)),
    ('time', None, datetime.datetime(2000, 1, 1, 0, 0))])

cube2 = NDCube(data, wm, extra_coords=[
    ('pix', 0, u.Quantity(np.arange(1, data.shape[0]+1), unit=u.pix) +
     cube1.extra_coords['pix']['value'][-1]),
    ('distance', None, u.Quantity(1, unit=u.cm)),
    ('time', None, datetime.datetime(2000, 1, 1, 0, 1))])

cube3 = NDCube(data2, wt, missing_axes=[False, False, False, True], extra_coords=[
    ('pix', 0, u.Quantity(np.arange(1, data2.shape[0]+1), unit=u.pix) +
     cube2.extra_coords['pix']['value'][-1]),
    ('distance', None, u.Quantity(2, unit=u.cm)),
    ('time', None, datetime.datetime(2000, 1, 1, 0, 2))])

cube4 = NDCube(data2, wm, extra_coords=[
    ('pix', 0, u.Quantity(np.arange(1, data2.shape[0]+1), unit=u.pix) +
     cube3.extra_coords['pix']['value'][-1]),
    ('distance', None, u.Quantity(3, unit=u.cm)),
    ('time', None, datetime.datetime(2000, 1, 1, 0, 3))])

cube2_no_no = NDCube(data, wm, extra_coords=[
    ('pix', 0, u.Quantity(np.arange(1, data.shape[0]+1), unit=u.pix) +
     cube1.extra_coords['pix']['value'][-1]),
    ('time', None, datetime.datetime(2000, 1, 1, 0, 1))])

cube3_no_time = NDCube(data2, wt, missing_axes=[False, False, False, True], extra_coords=[
    ('pix', 0, u.Quantity(np.arange(1, data2.shape[0]+1), unit=u.pix) +
     cube2.extra_coords['pix']['value'][-1]),
    ('distance', None, u.Quantity(2, unit=u.cm))])

cube3_diff_compatible_unit = NDCube(
    data2, wt, missing_axes=[False, False, False, True], extra_coords=[
        ('pix', 0, u.Quantity(np.arange(data2.shape[0]), unit=u.pix) +
         cube2.extra_coords['pix']['value'][-1]),
        ('distance', None, u.Quantity(2, unit=u.cm).to('m')),
        ('time', None, datetime.datetime(2000, 1, 1, 0, 2))])

cube3_diff_incompatible_unit = NDCube(
    data2, wt, missing_axes=[False, False, False, True], extra_coords=[
        ('pix', 0, u.Quantity(np.arange(data2.shape[0]), unit=u.pix) +
         cube2.extra_coords['pix']['value'][-1]),
        ('distance', None, u.Quantity(2, unit=u.s)),
        ('time', None, datetime.datetime(2000, 1, 1, 0, 2))])

cube1_time_common = NDCube(
    data, wt, missing_axes=[False, False, False, True],
    extra_coords=[('time', 1, [datetime.datetime(2000, 1, 1) + datetime.timedelta(minutes=i)
                               for i in range(data.shape[1])])])

cube2_time_common = NDCube(data, wm, extra_coords=[
    ('time', 1,
     [cube1_time_common.extra_coords["time"]["value"][-1] + datetime.timedelta(minutes=i)
      for i in range(1, data.shape[1]+1)])])

cube1_no_extra_coords = NDCube(data, wt, missing_axes=[False, False, False, True])
cube3_no_extra_coords = NDCube(data2, wt, missing_axes=[False, False, False, True])

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
nan_time_extra_coord = np.array([datetime.datetime(2000, 1, 1)+datetime.timedelta(minutes=i)
                                 for i in range(len(seq.data))])
nan_time_extra_coord[2] = np.nan


@pytest.mark.parametrize("test_input,expected", [
    (seq[0], NDCube),
    (seq[1], NDCube),
    (seq[2], NDCube),
    (seq[3], NDCube),
    (seq[0:1], NDCubeSequence),
    (seq[1:3], NDCubeSequence),
    (seq[0:2], NDCubeSequence),
    (seq[slice(0, 2)], NDCubeSequence),
    (seq[slice(0, 3)], NDCubeSequence),
])
def test_slice_first_index_sequence(test_input, expected):
    assert isinstance(test_input, expected)


@pytest.mark.parametrize("test_input,expected", [
    (seq[1:3], 2*u.pix),
    (seq[0:2], 2*u.pix),
    (seq[0::], 4*u.pix),
    (seq[slice(0, 2)], 2*u.pix),
    (seq[slice(0, 3)], 3*u.pix),
])
def test_slice_first_index_sequence(test_input, expected):
    assert test_input.dimensions[0] == expected


@pytest.mark.parametrize("test_input,expected", [
    (seq.index_as_cube[0:5].dimensions, (3*u.pix, [2., 2., 1.]*u.pix, 3*u.pix, 4*u.pix)),
    (seq.index_as_cube[1:3].dimensions, (2*u.pix, 1*u.pix, 3*u.pix, 4*u.pix)),
    (seq.index_as_cube[0:6].dimensions, (3*u.pix, 2*u.pix, 3*u.pix, 4*u.pix)),
    (seq.index_as_cube[0::].dimensions, (4*u.pix, 2*u.pix, 3*u.pix, 4*u.pix)),
    (seq.index_as_cube[0:5, 0].dimensions, (3*u.pix, [2., 2., 1.]*u.pix, 4*u.pix)),
    (seq.index_as_cube[1:3, 0:2].dimensions, (2*u.pix, 1*u.pix, 2*u.pix, 4*u.pix)),
    (seq.index_as_cube[0:6, 0, 0:1].dimensions, (3*u.pix, 2*u.pix, 1*u.pix)),
    (seq.index_as_cube[0::, 0, 0].dimensions, (4*u.pix, 2*u.pix)),
])
def test_index_as_cube(test_input, expected):
    for i in range(len(test_input)):
        try:
            assert test_input[i] == expected[i]
        except ValueError:
            assert (test_input[i].value == expected[i].value).all()
            assert test_input[i].unit == expected[i].unit


@pytest.mark.parametrize("test_input,expected", [
    (seq1.explode_along_axis(0), (8*u.pix, 3*u.pix, 4*u.pix)),
    (seq1.explode_along_axis(1), (12*u.pix, 2*u.pix, 4*u.pix)),
    (seq1.explode_along_axis(2), (16*u.pix, 2*u.pix, 3*u.pix)),
])
def test_explode_along_axis(test_input, expected):
    assert test_input.dimensions == expected


def test_explode_along_axis_error():
    with pytest.raises(ValueError):
        seq.explode_along_axis(1)


@pytest.mark.parametrize(
    "test_input,expected",
    [(seq, (4*u.pix, 2.*u.pix, 3.*u.pix, 4.*u.pix))])
def test_dimensions(test_input, expected):
    unit_tester = unittest.TestCase()
    unit_tester.assertEqual(test_input.dimensions, expected)


@pytest.mark.parametrize(
    "test_input,expected",
    [(seq, u.Quantity([8., 3., 4], unit=u.pix))])
def test_cube_like_dimensions(test_input, expected):
    assert (seq.cube_like_dimensions == expected).all()


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
       'time': np.array([datetime.datetime(2000, 1, 1)+datetime.timedelta(minutes=i)
                        for i in range(len(seq.data))])}),
     (seq2,
      {'distance': nan_extra_coord,
       'time': nan_time_extra_coord}),
     (seq3,
      {'distance': u.Quantity(range(4), unit=u.cm),
       'time': np.array([datetime.datetime(2000, 1, 1)+datetime.timedelta(minutes=i)
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
                        raise AssertionError("{0} != {1}".format(output_value, expected[key][i]))
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
