# -*- coding: utf-8 -*-
import pytest

from ndcube import utils

@pytest.mark.parametrize(
    "test_input,expected",
    [({}, False),
     ([slice(1, 5), slice(-1, -5, -2)], True)])
def test_all_slice(test_input, expected):
    assert utils.wcs._all_slice(test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [({}, []),
     ((slice(1,2), slice(1,3), 2, slice(2,4), 8),
      [slice(1, 2, None), slice(1, 3, None), slice(2, 3, None),
       slice(2, 4, None), slice(8, 9, None)])])
def test_slice_list(test_input, expected):
    assert utils.wcs._slice_list(test_input) == expected
