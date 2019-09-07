import pytest
import unittest

import numpy as np
import astropy.units as u

from ndcube import utils

missing_axes_none = [False] * 3
missing_axes_0_2 = [True, False, True]
missing_axes_1 = [False, True, False]

axes_length = 3
extra_coords_dict = {"time": {"axis": 0, "value": u.Quantity(range(axes_length), unit=u.pix)},
                     "hello": {"axis": 1, "value": u.Quantity(range(axes_length), unit=u.pix)}}
extra_coords_input = [('time', 0, u.Quantity(range(axes_length), unit=u.pix)),
                      ('hello', 1, u.Quantity(range(axes_length), unit=u.pix))]
extra_coords_dict_wcs = {"time": {"wcs axis": 0,
                                  "value": u.Quantity(range(axes_length), unit=u.pix)},
                         "hello": {"wcs axis": 1,
                                   "value": u.Quantity(range(axes_length), unit=u.pix)}}


@pytest.mark.parametrize(
    "test_input,expected",
    [((None, missing_axes_none), None),
     ((0, missing_axes_none), 2),
     ((1, missing_axes_none), 1),
     ((0, missing_axes_0_2), 1),
     ((1, missing_axes_1), 0),
     ((-1, missing_axes_0_2), 1),
     ((-2, missing_axes_1), 2),
     ((-1, missing_axes_none), 0)])
def test_data_axis_to_wcs_axis(test_input, expected):
    assert utils.cube.data_axis_to_wcs_axis(*test_input) == expected


@pytest.mark.parametrize("test_input", [(-2, missing_axes_0_2), (1, missing_axes_0_2)])
def test_data_axis_to_wcs_axis_error(test_input):
    with pytest.raises(IndexError):
        utils.cube.data_axis_to_wcs_axis(*test_input)


@pytest.mark.parametrize(
    "test_input,expected",
    [((None, missing_axes_none), None),
     ((0, missing_axes_none), 2),
     ((1, missing_axes_none), 1),
     ((1, missing_axes_0_2), 0),
     ((0, missing_axes_1), 1),
     ((-1, missing_axes_0_2), None),
     ((-2, missing_axes_0_2), 0),
     ((-2, missing_axes_1), None),
     ((-3, missing_axes_1), 1),
     ((-1, missing_axes_none), 0)])
def test_wcs_axis_to_data_axis(test_input, expected):
    assert utils.cube.wcs_axis_to_data_axis(*test_input) == expected


@pytest.mark.parametrize("test_input", [(-10, missing_axes_0_2), (10, missing_axes_0_2)])
def test_wcs_axis_to_data_axis_error(test_input):
    with pytest.raises(IndexError):
        utils.cube.data_axis_to_wcs_axis(*test_input)


def test_select_order():
    lists = [['TIME', 'WAVE', 'HPLT-TAN',
              'HPLN-TAN'], ['WAVE', 'HPLT-TAN', 'UTC',
                            'HPLN-TAN'], ['HPLT-TAN', 'TIME', 'HPLN-TAN'],
             ['HPLT-TAN', 'DEC--TAN',
              'WAVE'], [], ['UTC', 'TIME', 'WAVE', 'HPLT-TAN']]

    results = [
        [0, 1, 2, 3],
        [2, 0, 1, 3],
        [1, 0, 2],  # Second order is initial order
        [2, 0, 1],
        [],
        [1, 0, 2, 3]
    ]

    for (l, r) in zip(lists, results):
        assert utils.cube.select_order(l) == r


@pytest.mark.parametrize("test_input", [
    ([('name', 0)], [False, False], (1, 2)),
    ([(0, 0, 0)], [False, False], (1, 2)),
    ([('name', '0', 0)], [False, False], (1, 2)),
    ([('name', 0, [0, 1])], [False, False], (1, 2))
])
def test_format_input_extra_coords_to_extra_coords_wcs_axis_value(test_input):
    with pytest.raises(ValueError):
        utils.cube._format_input_extra_coords_to_extra_coords_wcs_axis(*test_input)


@pytest.mark.parametrize("test_input,expected", [
    ((extra_coords_dict, missing_axes_none), extra_coords_input),

    ((extra_coords_dict_wcs, missing_axes_none),
     [('time', 2, u.Quantity(range(axes_length), unit=u.pix)),
      ('hello', 1, u.Quantity(range(axes_length), unit=u.pix))]),

    ((extra_coords_dict_wcs, missing_axes_1),
     [('time', 1, u.Quantity(range(axes_length), unit=u.pix)),
      ('hello', None, u.Quantity(range(axes_length), unit=u.pix))])
])
def test_convert_extra_coords_dict_to_input_format(test_input, expected):
    output = utils.cube.convert_extra_coords_dict_to_input_format(*test_input)
    if len(output) != len(expected):
        raise AssertionError(f"{output} != {expected}")
    for output_tuple in output:
        j = 0
        while j < len(expected):
            if output_tuple[0] == expected[j][0]:
                assert len(output_tuple) == len(expected[j])
                print(output_tuple)
                print(expected[j])
                for k, el in enumerate(output_tuple):
                    try:
                        assert el == expected[j][k]
                    except ValueError as err:
                        if err.args[0] == "The truth value of an array with more than" + \
                                " one element is ambiguous. Use a.any() or a.all()":
                            assert (el == expected[j][k]).all()
                        else:
                            raise err
                j = len(expected) + 1
            else:
                j += 1
        if j == len(expected):
            raise AssertionError(f"{output} != {expected}")


def test_convert_extra_coords_dict_to_input_format_error():
    with pytest.raises(KeyError):
        utils.cube.convert_extra_coords_dict_to_input_format(
            {"time": {"not axis": 0, "value": []}}, missing_axes_none)


@pytest.mark.parametrize("test_input, expected", [
    ((5, False), np.asarray([0, 1, 2, 3, 4])),
    ((6, True), np.asarray([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]))
])
def test_pixel_centers_or_edges(test_input, expected):
    output = utils.cube._pixel_centers_or_edges(*test_input)
    assert isinstance(output, np.ndarray)
    np.testing.assert_allclose(output, expected)


@pytest.mark.parametrize("test_input, expected", [
    ((5, False), 5),
    ((6, True), 7)
])
def test_get_dimension_for_pixel(test_input, expected):
    output = utils.cube._get_dimension_for_pixel(*test_input)
    assert output == expected
