import pytest
import unittest

import numpy as np

from ndcube import utils

# sample data for tests
tuple_item0 = (0, slice(0, 3))
tuple_item1 = (slice(0, 2), slice(0, 3), slice(None))
tuple_item2 = (slice(3, 1, -1), slice(0, 3), slice(None))
tuple_item3 = (slice(4, None, -2), slice(0, 3), slice(None))

n_cubes = 4


@pytest.mark.parametrize("test_input,expected", [
    ((1, n_cubes), [utils.sequence.SequenceItem(sequence_index=1, cube_item=slice(None))]),
    ((slice(None), 2), [utils.sequence.SequenceItem(sequence_index=0, cube_item=slice(None)),
                        utils.sequence.SequenceItem(sequence_index=1, cube_item=slice(None))]),
    ((slice(0, 2), 3), [utils.sequence.SequenceItem(sequence_index=0, cube_item=slice(None)),
                        utils.sequence.SequenceItem(sequence_index=1, cube_item=slice(None))]),
    ((slice(1, 4, 2), 5), [utils.sequence.SequenceItem(sequence_index=1, cube_item=slice(None)),
                           utils.sequence.SequenceItem(sequence_index=3, cube_item=slice(None))]),
    ((slice(3, 1, -1), 5), [utils.sequence.SequenceItem(sequence_index=3, cube_item=slice(None)),
                            utils.sequence.SequenceItem(sequence_index=2, cube_item=slice(None))]),
    ((tuple_item0, n_cubes),
     [utils.sequence.SequenceItem(sequence_index=0, cube_item=tuple_item0[1])]),
    ((tuple_item1, n_cubes),
     [utils.sequence.SequenceItem(sequence_index=0, cube_item=tuple_item1[1:]),
      utils.sequence.SequenceItem(sequence_index=1, cube_item=tuple_item1[1:])]),
    ((tuple_item2, n_cubes),
     [utils.sequence.SequenceItem(sequence_index=3, cube_item=tuple_item1[1:]),
      utils.sequence.SequenceItem(sequence_index=2, cube_item=tuple_item1[1:])]),
    ((tuple_item3, n_cubes),
     [utils.sequence.SequenceItem(sequence_index=4, cube_item=tuple_item1[1:]),
      utils.sequence.SequenceItem(sequence_index=2, cube_item=tuple_item1[1:]),
      utils.sequence.SequenceItem(sequence_index=0, cube_item=tuple_item1[1:])])
])
def test_convert_item_to_sequence_items(test_input, expected):
    unit_tester = unittest.TestCase()
    unit_tester.assertEqual(
        utils.sequence.convert_item_to_sequence_items(*test_input), expected)


def test_convert_item_to_sequence_items_error():
    with pytest.raises(TypeError):
        utils.sequence.convert_item_to_sequence_items('item')


@pytest.mark.parametrize("test_input,expected", [
    # Test int cube_like_items.
    ((0, 0, np.array([3])), [utils.sequence.SequenceItem(sequence_index=0, cube_item=0)]),
    ((5, 0, np.array([3, 3])), [utils.sequence.SequenceItem(sequence_index=1, cube_item=2)]),
    # Below test reveals function doesn't work with negative int indexing.
    # ((-1, 0, np.array([3, 3])), [utils.sequence.SequenceItem(sequence_index=1, cube_item=2)]),

    # Test slice cube_like_items.
    ((slice(0, 2), 0, np.array([3])),
     [utils.sequence.SequenceItem(sequence_index=0, cube_item=slice(0, 2, 1))]),
    ((slice(1, 4), 0, np.array([3, 3])),
     [utils.sequence.SequenceItem(sequence_index=0, cube_item=slice(1, 3, 1)),
      utils.sequence.SequenceItem(sequence_index=1, cube_item=slice(0, 1, 1))]),
    ((slice(1, 7, 2), 0, np.array([3, 5])),
     [utils.sequence.SequenceItem(sequence_index=0, cube_item=slice(1, 3, 2)),
      utils.sequence.SequenceItem(sequence_index=1, cube_item=slice(1, 4, 2))]),
    ((slice(1, 7, 3), 0, np.array([3, 5])),
     [utils.sequence.SequenceItem(sequence_index=0, cube_item=slice(1, 2, 3)),
      utils.sequence.SequenceItem(sequence_index=1, cube_item=slice(1, 4, 3))]),
    # Below test reveals function doesn't work with negative stepping.
    # ((slice(6, 1, -1), 0, np.array([3, 5])),
    # [utils.sequence.SequenceItem(sequence_index=1, cube_item=slice(3, 0, -1)),
    #  utils.sequence.SequenceItem(sequence_index=0, cube_item=slice(2, 1, -1))]),

    # Test tuple cube_like_items
    (((0, 0, slice(1, 10)), 0, np.array([3, 5])),
     [utils.sequence.SequenceItem(sequence_index=0, cube_item=(0, 0, slice(1, 10, None)))]),

    (((0, 0, slice(1, 10)), 1, np.array([3, 5])),
     [utils.sequence.SequenceItem(sequence_index=0, cube_item=(0, 0, slice(1, 10, None)))]),

    (((slice(2, 10), 0, slice(1, 10)), 0, np.array([3, 5, 5])),
     [utils.sequence.SequenceItem(sequence_index=0,
                                  cube_item=(slice(2, 3, 1), 0, slice(1, 10, None))),
      utils.sequence.SequenceItem(sequence_index=1,
                                  cube_item=(slice(0, 5, 1), 0, slice(1, 10, None))),
      utils.sequence.SequenceItem(sequence_index=2,
                                  cube_item=(slice(0, 2, 1), 0, slice(1, 10, None)))]),

    (((0, slice(2, 10), slice(1, 10)), 1, np.array([3, 5, 5])),
     [utils.sequence.SequenceItem(sequence_index=0,
                                  cube_item=(0, slice(2, 3, 1), slice(1, 10, None))),
      utils.sequence.SequenceItem(sequence_index=1,
                                  cube_item=(0, slice(0, 5, 1), slice(1, 10, None))),
      utils.sequence.SequenceItem(sequence_index=2,
                                  cube_item=(0, slice(0, 2, 1), slice(1, 10, None)))]),
])
def test_convert_cube_like_item_to_sequence_items(test_input, expected):
    unit_tester = unittest.TestCase()
    unit_tester.assertEqual(
        utils.sequence.convert_cube_like_item_to_sequence_items(*test_input), expected)


@pytest.mark.parametrize("test_input", [
    (0, 1, np.array(3)), (slice(None), 1, np.array(3)),
    ((0, 1), 2, np.array([3, 3, 3])), (('item', 2), 0, np.array([3, 3, 3]))
])
def test_convert_cube_like_item_to_sequence_items_value_error(test_input):
    with pytest.raises(ValueError):
        utils.sequence.convert_cube_like_item_to_sequence_items(*test_input)


def test_convert_cube_like_item_to_sequence_items_type_error():
    with pytest.raises(TypeError):
        utils.sequence.convert_cube_like_item_to_sequence_items('item', 1, np.array(3))


@pytest.mark.parametrize("test_input,expected", [
    ((5, np.array([8] * 4)), utils.sequence.SequenceSlice(0, 5)),
    ((8, np.array([8] * 4)), utils.sequence.SequenceSlice(1, 0)),
    ((20, np.array([8] * 4)), utils.sequence.SequenceSlice(2, 4)),
    ((50, np.array([8] * 4)), utils.sequence.SequenceSlice(3, 8)),
])
def test_convert_cube_like_index_to_sequence_slice(test_input, expected):
    assert utils.sequence._convert_cube_like_index_to_sequence_slice(
        *test_input) == expected


@pytest.mark.parametrize("test_input,expected",
                         [((slice(2, 5), np.array([8] * 4)),
                           [utils.sequence.SequenceSlice(0, slice(2, 5, 1))]),
                          ((slice(5, 15), np.array([8] * 4)), [
                              utils.sequence.SequenceSlice(0, slice(5, 8, 1)),
                              utils.sequence.SequenceSlice(1, slice(0, 7, 1))
                          ]), ((slice(5, 16), np.array([8] * 4)), [
                              utils.sequence.SequenceSlice(0, slice(5, 8, 1)),
                              utils.sequence.SequenceSlice(1, slice(0, 8, 1))
                          ]), ((slice(5, 23), np.array([8] * 4)), [
                              utils.sequence.SequenceSlice(0, slice(5, 8, 1)),
                              utils.sequence.SequenceSlice(1, slice(0, 8, 1)),
                              utils.sequence.SequenceSlice(2, slice(0, 7, 1))
                          ]), ((slice(5, 100), np.array([8] * 4)), [
                              utils.sequence.SequenceSlice(0, slice(5, 8, 1)),
                              utils.sequence.SequenceSlice(1, slice(0, 8, 1)),
                              utils.sequence.SequenceSlice(2, slice(0, 8, 1)),
                              utils.sequence.SequenceSlice(3, slice(0, 8, 1))
                          ])])
def test_convert_cube_like_slice_to_sequence_slices(test_input, expected):
    assert utils.sequence._convert_cube_like_slice_to_sequence_slices(*test_input) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [((slice(0, 10), 20), slice(0, 10, 1)),
     ((slice(0, 10, 2), 20), slice(0, 10, 2)),
     ((slice(None, 0, -1), 20), slice(20, 0, -1))])
def test_convert_slice_nones_to_ints(test_input, expected):
    assert utils.sequence.convert_slice_nones_to_ints(*test_input) == expected


@pytest.mark.parametrize("test_input,expected", [
    ((utils.sequence.SequenceSlice(0, 0), 1), utils.sequence.SequenceItem(0, (slice(None), 0)))
])
def test_convert_sequence_slice_to_sequence_item(test_input, expected):
    unit_tester = unittest.TestCase()
    unit_tester.assertEqual(
        utils.sequence._convert_sequence_slice_to_sequence_item(*test_input), expected)
