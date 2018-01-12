# -*- coding: utf-8 -*-
# Author: Ankit Baruah and Daniel Ryan <ryand5@tcd.ie>

"""Utilities for ndcube."""

from __future__ import absolute_import

from copy import deepcopy
from collections import namedtuple

import numpy as np
from astropy import units as u

from ndcube import wcs_util

# Define SequenceSlice named tuple of length 2. Its attributes are:
# sequence_index: an int giving the index of a cube within an NDCubeSequence.
# common_axis_item: slice of int index of to be to be applied to the common
# axis of the cube.
SequenceSlice = namedtuple("SequenceSlice", "sequence_index common_axis_item")
# Define SequenceItem named tuple of length 2. Its attributes are:
# sequence_index: an int giving the index of a cube within an NDCubeSequence.
# cube_item: item (int, slice, tuple) to be applied to cube identified
# by sequence_index attribute.
SequenceItem = namedtuple("SequenceItem", "sequence_index cube_item")


def select_order(axtypes):
    """
    Returns the indices of the correct axis priority for the given list of WCS
    CTYPEs. For example, given ['HPLN-TAN', 'TIME', 'WAVE'] it will return
    [1, 2, 0] because index 1 (time) has the highest priority, followed by
    wavelength and finally solar-x. When two or more celestial axes are in the
    list, order is preserved between them (i.e. only TIME, UTC and WAVE are
    moved)

    Parameters
    ----------
    axtypes: str list
        The list of CTYPEs to be modified.
    """
    order = [(0, t) if t in ['TIME', 'UTC'] else
             (1, t) if t == 'WAVE' else
             (2, t) if t == 'HPLT-TAN' else
             (axtypes.index(t) + 3, t) for t in axtypes]
    order.sort()
    result = [axtypes.index(s) for (_, s) in order]
    return result


def convert_item_to_sequence_items(item, n_cubes):
    """
    Converts NDCubeSequence __getitem__ item to list of SequenceSlice objects.

    Parameters
    ----------
    item: `int`, `slice`, or `tuple` of `int` and/or `slice`.
        An slice/index item compatible with input to NDCubeSequence.__getitem__.

    n_cubes: `int`
        Number of cubes in NDCubeSequence being sliced.

    Returns
    -------
    result: `list` of SequenceItem `namedtuple`.
        The slice/index items for each relevant NDCube within the NDCubeSequence
        which together represent the original input slice/index item.

    """
    cube_slice_default = slice(None)
    if isinstance(item, int):
        sequence_items = get_sequence_items_from_int_item(item, cube_slice_default)
    elif isinstance(item, slice):
        sequence_items = get_sequence_items_from_slice_item(item, cube_slice_default, n_cubes)
    elif isinstance(item, tuple):
        sequence_items = get_sequence_items_from_tuple_item(item, n_cubes)
    else:
        raise TypeError("Unrecognized slice type: {0}", item)
    return sequence_items


def get_sequence_items_from_int_item(int_item, cube_item):
    """
    Converts int index of an NDCubeSequence to list of SequenceSlices.

    Parameters
    ----------
    int_item: `int`
        index of NDCube within NDCubeSequence to be slices out.

    cube_item: `int`, `slice`, or `tuple`
        Item to be applied to selected NDCube.

    Returns
    -------
    result: `list` of SequenceItem `namedtuple`
        The slice/index items for each relevant NDCube within the NDCubeSequence
        which together represent the original input slice/index item.

    """
    return [SequenceItem(int_item, cube_item)]


def get_sequence_items_from_slice_item(slice_item, cube_item, n_cubes):
    """
    Converts slice item of an NDCubeSequence to list of SequenceSlices.

    Parameters
    ----------
    slice_item: `slice`
        Indicates which NDCubes within NDCubeSequence are to be slices out.

    cube_item: `int`, `slice`, or `tuple`
        Item to be applied to each selected NDCube.

    n_cubes: `int`
        Number of cubes in NDCubeSequence being sliced.

    Returns
    -------
    sequence_items: `list` of SequenceItem `namedtuple`.
        The slice/index items for each relevant NDCube within the NDCubeSequence
        which together represent the original input slice/index item.

    """
    # If there are None types in slice, replace with correct entries based on sign of step.
    if not slice_item.step:
        step = 1
    else:
        step = slice_item.step
    start = slice_item.start
    stop = slice_item.stop
    if step < 0:
        if not slice_item.start:
            start = n_cubes
        if not slice_item.stop:
            stop = 0
    else:
        if not slice_item.start:
            start = 0
        if not slice_item.stop:
            stop = n_cubes
    # Derive SequenceItems for each cube.
    sequence_items = [SequenceItem(i, cube_item) for i in range(start, stop, step)]
    return sequence_items


def get_sequence_items_from_tuple_item(tuple_item, n_cubes):
    """
    Converts NDCubeSequence slice item tuple to list of SequenceSlice objects.

    Parameters
    ----------
    tuple_item: `tuple` of `int` and/or `slice`.
        Index/slice for different dimensions of NDCubeSequence.  The first entry
        applies to the sequence axis while subsequent entries make up the slicing
        item to be applied to the NDCubes.

    n_cubes: `int`
        Number of cubes in NDCubeSequence being sliced.

    Returns
    -------
    sequence_items: `list` of SequenceItem `namedtuple`.
        The slice/index items for each relevant NDCube within the NDCubeSequence
        which together represent the original input slice/index item.

    """
    # Define slice to be applied to cubes.
    if len(tuple_item[1:]) == 1:
        cube_item = tuple_item[1]
    else:
        cube_item = tuple_item[1:]
    # Based on type of sequence index, define sequence slices.
    if isinstance(tuple_item[0], int):
        sequence_items = get_sequence_items_from_int_item(tuple_item[0], cube_item)
    elif isinstance(tuple_item[0], slice):
        sequence_items = get_sequence_items_from_slice_item(tuple_item[0], cube_item, n_cubes)
    else:
        raise TypeError("Unrecognized sequence slice type: {0}".format(tuple_item[0]))
    return sequence_items


def slice_sequence(cubesequence, sequence_items):
    """
    Slices an NDCubeSequence given a list of SequenceSlice objects.

    Parameters
    ----------
    cubesequence: `ndcube.NDCubeSequence`
        The cubesequence to slice.
    sequence_items: `list` of `SequenceItem`
        Slices to be applied to each relevant NDCube in the sequence.

    Returns
    -------
    result: `NDCubeSequence` or `NDCube`
        The sliced cube sequence.

    """
    result = deepcopy(cubesequence)
    if len(sequence_items) == 1:
        return result.data[sequence_items[0].sequence_index][sequence_items[0].cube_item]
    else:
        data = [result.data[sequence_item.sequence_index][sequence_item.cube_item]
                for sequence_item in sequence_items]
        result.data = data
        return result


def index_sequence_as_cube(cubesequence, item):
    """
    Enables NDCubeSequence to be indexed as if it were a single NDCube.

    This is only possible if cubesequence._common_axis is set,
    i.e. if the cubes are sequenced in order along one of the cube axes.
    For example, if cubesequence._common_axis is 1 where the first axis is
    time, and the cubes are sequenced chronologically such that the last
    time slice of one cube is directly followed in time by the first time
    slice of the next cube, then this function allows the NDCubeSequence to
    be indexed as though all cubes were combined into one ordered along
    the time axis.

    Parameters
    ----------
    cubesequence: `ndcube.NDCubeSequence`
        The cubesequence to get the item from
    item: `int`, `slice` or `tuple` of `int` and/or `slice`.
        The item to get from the cube.  If tuple length must be <= number
        of dimensions in single cube.

    Example
    -------
    >>> # Say we have three Cubes each cube has common_axis=1 is time and shape=(3,3,3)
    >>> data_list = [cubeA, cubeB, cubeC] # doctest: +SKIP
    >>> cs = NDCubeSequence(data_list, meta=None, common_axis=1) # doctest: +SKIP
    >>> # return zeroth time slice of cubeB in via normal CubeSequence indexing.
    >>> cs[1,:,0,:] # doctest: +SKIP
    >>> # Return same slice using this function
    >>> index_sequence_as_cube(cs, (slice(0, cubeB.shape[0]), 0, (slice(0, cubeB.shape[2])) # doctest: +SKIP

    """
    # Convert index_as_cube item to a list of regular NDCubeSequence
    # items of each relevant cube.
    sequence_items = convert_cube_like_item_to_sequence_items(cubesequence, item)
    # Use sequence items to slice NDCubeSequence.
    return slice_sequence(cubesequence, sequence_items)


def convert_cube_like_item_to_sequence_items(cubesequence, cube_like_item):
    """
    Converts an input item to NDCubeSequence.index_as_cube to a list od SequenceSlice objects.

    Parameters
    ----------
    cubesequence: `NDCubeSequence`
        NDCubeSequence being sliced/indexed.

    cube_like_item: `int`, `slice`, of `tuple` of `int and/or `slice`.
        Item compatible with input to NDCubeSequence.index_as_cube.

    Returns
    -------
    sequence_items: `list` of SequenceItem `namedtuple`.
        The slice/index items for each relevant NDCube within the NDCubeSequence
        which together represent the original input slice/index item.

    """
    # Determine length of each cube along common axis.
    cube_lengths = np.array([c.data.shape[cubesequence._common_axis]
                             for c in cubesequence.data])
    invalid_item_error_message = "Invalid index/slice input."
    # Case 1: Item is int and common axis is 0.
    if isinstance(cube_like_item, int):
        if cubesequence._common_axis != 0:
            raise ValueError("Input can only be indexed with an int if "
                             "CubeSequence's common axis is 0. common "
                             "axis = {0}".format(cubesequence._common_axis))
        else:
            # Derive list of SequenceSlice objects that describes the
            # cube_like_item in regular slicing notation.
            sequence_slices = [_convert_cube_like_index_to_sequence_slice(
                cube_like_item, cube_lengths)]
            all_axes_item = None
    # Case 2: Item is slice and common axis is 0.
    elif isinstance(cube_like_item, slice):
        if cubesequence._common_axis != 0:
            raise ValueError("Input can only be sliced with a single slice if "
                             "CubeSequence's common axis is 0. common "
                             "axis = {0}".format(cubesequence._common_axis))
        else:
            # Derive list of SequenceSlice objects that describes the
            # cube_like_item in regular slicing notation.
            # First ensure None types within slice are replaced with appropriate ints.
            sequence_slices = _convert_cube_like_slice_to_sequence_slices(
                cube_like_item, cube_lengths)
            all_axes_item = None
    # Case 3: Item is tuple.
    elif isinstance(cube_like_item, tuple):
        # Check item is long enough to include common axis.
        if len(cube_like_item) < cubesequence._common_axis:
            raise ValueError("Input item not long enough to include common axis."
                             "Must have length between "
                             "{0} and {1} inclusive.".format(
                                 cubesequence._common_axis, len(cubesequence[0].data.shape)))
        # Based on type of slice/index in the common axis position of
        # the cube_like_item, derive list of SequenceSlice objects that
        # describes the cube_like_item in regular slicing notation.
        if isinstance(cube_like_item[cubesequence._common_axis], int):
            sequence_index = _convert_cube_like_index_to_sequence_slice(
                cube_like_item[cubesequence._common_axis], cube_lengths)
            sequence_slices = get_sequence_items_from_int_item(
                sequence_index.sequence_index, sequence_index.common_axis_index)
        elif isinstance(cube_like_item[cubesequence._common_axis], slice):
            sequence_slices = _convert_cube_like_slice_to_sequence_slices(
                cube_like_item[cubesequence._common_axis], cube_lengths)
        else:
            raise ValueError(invalid_item_error_message)
        all_axes_item = cube_like_item
    # Convert the sequence slices, that only describe the slicing along
    # the sequence axis and common axis to sequence items which
    # additionally describe how the non-common cube axes should be sliced.
    sequence_items = [_convert_sequence_slice_to_sequence_item(
        sequence_slice, cubesequence._common_axis, cube_like_item=all_axes_item)
        for sequence_slice in sequence_slices]
    return sequence_items


def _convert_cube_like_index_to_sequence_slice(cube_like_index, cube_lengths):
    """
    Converts a cube-like index of an NDCubeSequence to indices along the sequence and common axes.

    Parameters
    ----------
    cube_like_index: `int`
        Cube-like index of NDCubeSequence

    cube_lengths: iterable of `int`
        Length of each cube along common axis.

    Returns
    -------
    sequence_slice: SequenceSlice `namedtuple`.
        First element gives index of cube along sequence axis.
        Second element each index along common axis of relevant cube.

    """
    # Derive cumulative lengths of cubes along common axis.
    cumul_cube_lengths = np.cumsum(cube_lengths)
    # If cube_like_index is within first cube in sequence, it is
    # simple to determine the sequence and comon axis indices.
    if cube_like_index < cumul_cube_lengths[0]:
        sequence_index = 0
        common_axis_index = cube_like_index
    # Else use more in-depth method.
    else:
        # Determine the index of the relevant cube within the sequence
        # from the cumulative common axis cube lengths.
        sequence_index = np.where(cumul_cube_lengths <= cube_like_index)[0][-1]
        if cube_like_index > cumul_cube_lengths[-1]-1:
            # If the cube is out of range then return the last common axis index.
            common_axis_index = cube_lengths[-1]-1
        else:
            # Else use simple equation to derive the relevant common axis index.
            common_axis_index = cube_like_index-cumul_cube_lengths[sequence_index]
        # sequence_index should be plus one as the sequence_index earlier is
        # previous index if it is not already the last cube index.
        if sequence_index < cumul_cube_lengths.size - 1:
            sequence_index += 1
    # Return sequence and cube indices.  Ensure they are int, rather
    # than np.int64 to avoid confusion in checking type elsewhere.
    return SequenceSlice(int(sequence_index), int(common_axis_index))


def _convert_cube_like_slice_to_sequence_slices(cube_like_slice, cube_lengths):
    """
    Converts common axis slice input to NDCubeSequence.index_as_cube to a list of sequence indices.

    Parameters
    ----------
    cube_like_slice: `slice`
        Slice along common axis in NDCubeSequence.index_as_cube item.

    cube_lengths: iterable of `int`
        Length of each cube along common axis.

    Returns
    -------
    sequence_slices: `list` of SequenceSlice `namedtuple`.
        List sequence slices (sequence axis, common axis) for each element
        along common axis represented by input cube_like_slice.

    """
    # Ensure any None attributes in input slice are filled with appropriate ints.
    cumul_cube_lengths = np.cumsum(cube_lengths)
    cube_like_slice = convert_slice_nones_to_ints(cube_like_slice, cumul_cube_lengths[-1])
    # Determine sequence indices of cubes included in cube-like slice.
    cube_like_indices = np.arange(cumul_cube_lengths[-1])[cube_like_slice]
    n_cubes = len(cube_like_indices)
    one_step_sequence_slices = np.empty(n_cubes, dtype=object)
    sequence_int_indices = np.zeros(n_cubes, dtype=int)
    for i in range(n_cubes):
        one_step_sequence_slices[i] = _convert_cube_like_index_to_sequence_slice(
            cube_like_indices[i], cube_lengths)
        sequence_int_indices[i] = one_step_sequence_slices[i].sequence_index
    unique_index = np.sort(np.unique(sequence_int_indices, return_index=True)[1])
    unique_sequence_indices = sequence_int_indices[unique_index]
    # Get cumulative cube lengths of selected cubes.
    unique_cumul_cube_lengths = cumul_cube_lengths[unique_sequence_indices]
    # Convert start and stop cube-like indices to sequence indices.
    first_sequence_index = _convert_cube_like_index_to_sequence_slice(cube_like_slice.start,
                                                                      cube_lengths)
    last_sequence_index = _convert_cube_like_index_to_sequence_slice(cube_like_slice.stop,
                                                                     cube_lengths)
    # Since the last index of any slice represents
    # 'up to but not including this element', if the last sequence index
    # is the first element of a new cube, elements from the last cube
    # will not appear in the sliced sequence.  Therefore for ease of
    # slicing, we can redefine the final sequence index as the penultimate
    # cube and its common axis index as beyond the range of the
    # penultimate cube's length along the common axis.
    if last_sequence_index.sequence_index > first_sequence_index.sequence_index and \
      last_sequence_index.common_axis_item == 0:
        last_sequence_index = SequenceSlice(
            last_sequence_index.sequence_index-1,
            cumul_cube_lengths[last_sequence_index.sequence_index-1])
    # Iterate through relevant cubes and determine slices for each.
    # Do last cube outside loop as its end index may not correspond to
    # the end of the cube's common axis.
    if not cube_like_slice.step:
        step = 1
    else:
        step = cube_like_slice.step
    sequence_slices = []
    common_axis_start_index = first_sequence_index.common_axis_item
    j = 0
    while j < len(unique_sequence_indices)-1:
        # Let i be the index along the sequence axis of the next relevant cube.
        i = unique_sequence_indices[j]
        # Determine last common axis index for this cube.
        common_axis_last_index = \
          cube_lengths[i] - ((cube_lengths[i]-common_axis_start_index) % step)
        # Generate SequenceSlice for this cube and append to list.
        sequence_slices.append(SequenceSlice(
            i, slice(common_axis_start_index, common_axis_last_index+1, step)))
        # Determine first common axis index for next cube.
        if cube_lengths[i] == common_axis_last_index:
            common_axis_start_index = step-1
        else:
            common_axis_start_index = \
              step - (((cube_lengths[i]-common_axis_last_index) % step) +
                      cumul_cube_lengths[unique_sequence_indices[j+1]-1] - cumul_cube_lengths[i])
        # Iterate counter.
        j += 1
    # Create slice for last cube manually.
    sequence_slices.append(SequenceSlice(
        unique_sequence_indices[j],
        slice(common_axis_start_index, last_sequence_index.common_axis_item, step)))
    return sequence_slices


def _convert_sequence_slice_to_sequence_item(sequence_slice, common_axis, cube_like_item=None):
    """
    Converts sequence/cube index to a SequenceSlice object.

    Parameters
    ----------
    sequence_slice: SequenceSlice `namedtuple`
        0th element gives index of cube along sequence axis.
        1st element each index along common axis of relevant cube.
        Must be same format as output from _convert_cube_like_index_to_sequence_indices.

    common_axis: `int`
        Common axis as defined in NDCubeSequence.

    cube_like_item: `None` or `tuple` of `slice` and/or `int` objects (Optional)
        The original item input to `NDCubeSequence.index_as_cube` including the
        slices/indices of non-common axes of cubes within sequence.  If None, a
        tuple of slice(None) objects is generated  long enough so that the last
        element in the tuple corresponds to the common axis and is set to the
        1st (0-based counting) the sequence_index input, above.  This tuple is
        then set to the cube_item attribute of the output `SequenceSlice` object.

    Returns
    -------
    sequence_item: SequenceSlice `namedtuple`.
        Describes sequence index of an NDCube within an NDCubeSequence and the
        slice/index item to be applied to the whole NDCube.

    """
    if not cube_like_item and common_axis == 0:
        sequence_item = SequenceItem(sequence_slice.sequence_index,
                                     sequence_slice.common_axis_item)
    else:
        # Create mutable version of cube_like_item.
        try:
            cube_item_list = list(cube_like_item)
        except TypeError as err:
            if err.args[0] == "'NoneType' object is not iterable":
                cube_item_list = []
            else:
                raise err
        # Make sure cube_like_item is long enough to include common axis
        while len(cube_item_list) < common_axis:
            cube_item_list.append(slice(None))
        # Create new sequence slice
        cube_item_list[common_axis] = sequence_slice.common_axis_item
        sequence_item = SequenceItem(sequence_slice.sequence_index, tuple(cube_item_list))
    return sequence_item


def convert_slice_nones_to_ints(slice_item, target_length):
    """
    Converts None types within a slice to the appropriate ints based on object to be sliced.

    Parameters
    ----------
    slice_item: `slice`
       Slice for which Nones should be converted.

    target_length: `int`
        Length of object to which slice will be applied.

    Returns
    -------
    new_slice: `slice`
        Slice with Nones replaced with ints.

    """
    if not slice_item.step:
        step = 1
    else:
        step = slice_item.step
    start = slice_item.start
    stop = slice_item.stop
    if step < 0:
        if not slice_item.start:
            start = int(target_length)
        if not slice_item.stop:
            stop = 0
    else:
        if not slice_item.start:
            start = 0
        if not slice_item.stop:
            stop = int(target_length)
    return slice(start, stop, step)


def assert_extra_coords_equal(test_input, extra_coords):
    assert test_input.keys() == extra_coords.keys()
    for key in list(test_input.keys()):
        assert test_input[key]['axis'] == extra_coords[key]['axis']
        assert (test_input[key]['value'] == extra_coords[key]['value']).all()

def assert_metas_equal(test_input, expected_output):
    assert test_input.keys() == expected_output.keys()
    for key in list(test_input.keys()):
        assert test_input[key] == expected_output[key]


def assert_cubes_equal(test_input, expected_cube):
    assert type(test_input) == type(expected_cube)
    assert np.all(test_input.mask == expected_cube.mask)
    wcs_util.assert_wcs_are_equal(test_input.wcs, expected_cube.wcs)
    assert test_input.missing_axis == expected_cube.missing_axis
    assert test_input.uncertainty.array.shape == expected_cube.uncertainty.array.shape
    assert test_input.dimensions[1] == expected_cube.dimensions[1]
    assert np.all(test_input.dimensions[0].value == expected_cube.dimensions[0].value)
    assert test_input.dimensions[0].unit == expected_cube.dimensions[0].unit
    assert_extra_coords_equal(test_input._extra_coords, expected_cube._extra_coords)


def assert_cubesequences_equal(test_input, expected_sequence):
    assert type(test_input) == type(expected_sequence)
    assert_metas_equal(test_input.meta, expected_sequence.meta)
    assert test_input._common_axis == expected_sequence._common_axis
    for i, cube in enumerate(test_input.data):
        assert_cubes_equal(cube, expected_sequence.data[i])
