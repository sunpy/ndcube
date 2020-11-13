
import pytest

from ndcube import utils
from ndcube.utils.sequence import SequenceItem

# sample data for tests
tuple_item0 = (0, slice(0, 3))
tuple_item1 = (slice(0, 2), slice(0, 3), slice(None))
tuple_item2 = (slice(3, 1, -1), slice(0, 3), slice(None))
tuple_item3 = (slice(4, None, -2), slice(0, 3), slice(None))

n_cubes = 4


@pytest.mark.parametrize(
    "cube_like_index, common_axis, common_axis_lengths, expected_seq_idx, expected_common_idx",
    [(3, 1, [4, 4], 0, 3),
     (3, 1, [2, 2], 1, 1)]
)
def test_cube_like_index_to_sequence_and_common_axis_indices(
        cube_like_index, common_axis, common_axis_lengths, expected_seq_idx, expected_common_idx):
    sequence_index, common_axis_index = \
        utils.sequence.cube_like_index_to_sequence_and_common_axis_indices(
            cube_like_index, common_axis, common_axis_lengths)
    assert sequence_index == expected_seq_idx
    assert common_axis_index == expected_common_idx


@pytest.mark.parametrize(
    "item, common_axis, common_axis_lengths, n_cube_dims, expected_sequence_items", [
        ((slice(None), slice(4, 6)), 1, [3, 3], 4,
         [SequenceItem(sequence_index=1, cube_item=slice(1, 3))]),

        ((slice(None), slice(None)), 1, [3, 3, 3], 4,
         [SequenceItem(sequence_index=0, cube_item=slice(0, None)),
          SequenceItem(sequence_index=1, cube_item=slice(None)),
          SequenceItem(sequence_index=2, cube_item=slice(None, 9))])]
)
def test_cube_like_tuple_item_to_sequence_items(
        item, common_axis, common_axis_lengths, n_cube_dims, expected_sequence_items):
    pass


def test_cube_like_tuple_item_to_sequence_items_error1():
    with pytest.raises(TypeError):
        utils.sequence.cube_like_tuple_item_to_sequence_items(1, 1, [2, 2], 3)


def test_cube_like_tuple_item_to_sequence_items_error2():
    with pytest.raises(ValueError):
        utils.sequence.cube_like_tuple_item_to_sequence_items((1, 1), 3, [2, 2], 4)


def test_cube_like_tuple_item_to_sequence_items_error3():
    with pytest.raises(TypeError):
        utils.sequence.cube_like_tuple_item_to_sequence_items((1, 1), 1, [2, 2], 3)
