import unittest

import astropy.units as u
import numpy as np
import pytest

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


@pytest.mark.parametrize("test_input", [
    ([('name', 0)], np.array([0, 1]), 2, (1, 2)),
    ([(0, 0, 0)], np.array([0, 1]), 2, (1, 2)),
    ([('name', '0', 0)], np.array([0, 1]), 2, (1, 2))
])
def test_format_input_extra_coords_to_extra_coords_wcs_axis_value(test_input):
    with pytest.raises(ValueError):
        utils.cube._format_input_extra_coords_to_extra_coords_wcs_axis(*test_input)


@pytest.mark.parametrize("test_input,expected", [
    ((extra_coords_dict, np.array([0, 1, 2]), 3), extra_coords_input),

    ((extra_coords_dict_wcs, np.array([0, 1, 2]), 3),
     [('time', 2, u.Quantity(range(axes_length), unit=u.pix)),
      ('hello', 1, u.Quantity(range(axes_length), unit=u.pix))]),

    ((extra_coords_dict_wcs, np.array([0, 2]), 3),
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
            {"time": {"not axis": 0, "value": []}}, [0, 1, 2], 3)

