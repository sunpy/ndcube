# -*- coding: utf-8 -*-
import pytest

from ndcube import utils

missing_axis_none = [False]*3
missing_axis_0_2 = [True, False, True]
missing_axis_1 = [False, True, False]

@pytest.mark.parametrize(
    "test_input,expected",
    [((None, missing_axis_none), None),
     ((0, missing_axis_none), 2),
     ((1, missing_axis_none), 1),
     ((0, missing_axis_0_2), 1),
     ((1, missing_axis_1), 0)])
def test_data_axis_to_wcs_axis(test_input, expected):
    assert utils.cube.data_axis_to_wcs_axis(*test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [((None, missing_axis_none), None),
     ((0, missing_axis_none), 2),
     ((1, missing_axis_none), 1),
     ((1, missing_axis_0_2), 0),
     ((0, missing_axis_1), 1)])
def test_wcs_axis_to_data_axis(test_input, expected):
    assert utils.cube.wcs_axis_to_data_axis(*test_input) == expected


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


@pytest.mark.parametrize(
    "test_input",
    [([('name', 0)], [False, False], (1, 2)),
      ([(0, 0, 0)], [False, False], (1, 2)),
      ([('name', '0', 0)], [False, False], (1, 2)),
      ([('name', 0, [0, 1])], [False, False], (1, 2))])
def test_format_input_extra_coords_to_extra_coords_wcs_axis_value(test_input):
    with pytest.raises(ValueError):
        utils.cube._format_input_extra_coords_to_extra_coords_wcs_axis(*test_input)
