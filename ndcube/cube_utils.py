# -*- coding: utf-8 -*-
# Author: Ankit Baruah and Daniel Ryan <ryand5@tcd.ie>

"""Utilities for ndcube."""

from __future__ import absolute_import

from copy import deepcopy

import numpy as np
from astropy import units as u

from ndcube import wcs_util


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


def convert_item_to_sequence_slices(item):
    """Converts NDCubeSequence slice item to list of SequenceSlice objects."""
    cube_slice_default = slice(None)
    if isinstance(item, int):
        sequence_slices = get_sequence_slices_from_int_item(item, cube_slice_default)
    elif isinstance(item, slice):
        sequence_slices = get_sequence_slices_from_slice_item(item, cube_slice_default)
    elif isinstance(item, tuple):
        sequence_slices = get_sequence_slices_from_tuple_item(item)
    else:
        raise TypeError("Unrecognized slice type: {0}", item)
    return sequence_slices


def get_sequence_slices_from_int_item(int_item, cube_slice):
    """
    Converts int index of an NDCubeSequence to list of SequenceSlices.

    Parameters
    ----------
    int_item: `int`
        index of NDCube within NDCubeSequence to be slices out.

    cube_slice: `int`, `slice`, or `tuple`
        Slice to be applied to selected NDCube.

    Returns
    -------
    result: `list` of `SequenceSlice`
        List of a length one containing a SequenceSlice object giving the
        index of the selected NDCube and the slice to be applied to that cube.

    """
    return [SequenceSlice(int_item, cube_slice)]


def get_sequence_slices_from_slice_item(slice_item, cube_slice):
    """
    Converts slice item of an NDCubeSequence to list of SequenceSlices.

    Parameters
    ----------
    slice_item: `slice`
        Indicates which NDCubes within NDCubeSequence are to be slices out.

    cube_slice: `int`, `slice`, or `tuple`
        Slice to be applied to each selected NDCube.

    Returns
    -------
    sequence_slices: `list` of `SequenceSlice`
        SequenceSlices for each relevant NDCube within NDCubeSequence.

    """
    try:
        sequence_slices = [SequenceSlice(i, cube_slice)
                           for i in range(slice_item.start, slice_item.stop, slice_item.step)]
    except TypeError:
        sequence_slices = [SequenceSlice(i, cube_slice)
                           for i in range(slice_item.start, slice_item.stop)]
    return sequence_slices


def get_sequence_slices_from_tuple_item(tuple_item):
    """
    Converts NDCubeSequence slice item tuple to list of SequenceSlice objects.

    Parameters
    ----------
    tuple_item: `tuple` of `int` and/or `slice`.
        Index/slice for different dimensions of NDCubeSequence.  The first entry
        applies to the sequence axis while subsequent entries make up the slicing
        item to be applied to the NDCubes.

    Returns
    -------
    sequence_slices: `list` of `SequenceSlice`
        SequenceSlices for each relevant NDCube within NDCubeSequence.

    """
    # Define slice to be applied to cubes.
    if len(tuple_item[1:]) == 1:
        cube_slice = tuple_item[1]
    else:
        cube_slice = tuple_item[1:]
    # Based on type of sequence index, define sequence slices.
    if isinstance(tuple_item[0], int):
        sequence_slices = get_sequence_slices_from_int_item(tuple_item[0], cube_slice)
    elif isinstance(tuple_item[0], slice):
        sequence_slices = get_sequence_slices_from_slice_item(tuple_item[0], cube_slice)
    else:
        raise TypeError("Unrecognized sequence slice type: {0}".format(tuple_item[0]))
    return sequence_slices

def slice_sequence(cubesequence, sequence_slices):
    """
    Slices an NDCubeSequence given a list of SequenceSlice objects.

    Parameters
    ----------
    cubesequence: `ndcube.NDCubeSequence`
        The cubesequence to slice.
    sequence_slices: `list` of `SequenceSlice`
        Slices to be applied to each relevant NDCube in the sequence.

    Returns
    -------
    result: `NDCubeSequence` or `NDCube`
        The sliced cube sequence.

    """
    result = deepcopy(cubesequence)
    if len(sequence_slices) == 1:
        return result.data[sequence_slices[0].sequence_index][sequence_slices[0].cube_slice]
    else:
        data = [result.data[sequence_slice.sequence_index][sequence_slice.cube_slice]
                for sequence_slice in sequence_slices]
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
    # Convert index_as_cube item to a regular NDCubeSequence item.
    sequence_slices = convert_index_as_cube_item_to_sequence_slices(cubesequence, item)
    # Use output to slice NDCubeSequence as normal.
    return slice_sequence(cubesequence, sequence_slices)


def _index_sequence_as_cube(cubesequence, item):
    """
    Enables CubeSequence to be indexed as a single Cube.

    This is only possible if cubesequence._common_axis is set,
    i.e. if the Cubes are sequence in order along one of the Cube axes.
    For example, if cubesequence._common_axis=1 where the first axis is
    time, and the Cubes are sequence chronologically such that the last
    time slice of one Cube is directly followed in time by the first time
    slice of the next Cube, then this function allows the CubeSequence to
    be indexed as though all Cubes were combined into one ordered along
    the time axis.

    Parameters
    ----------
    cubesequence: ndcube.CubeSequence object
        The cubesequence to get the item from
    item: int, slice object, or tuple of these
        The item to get from the cube.  If tuple length must be <= number
        of dimensions in single Cube.

    Example
    -------
    >>> # Say we have three Cubes each cube has common_axis=1 is time and shape=(3,3,3)
    >>> data_list = [cubeA, cubeB, cubeC] # doctest: +SKIP
    >>> cs = CubeSequence(data_list, meta=None, common_axis=1) # doctest: +SKIP
    >>> # return zeroth time slice of cubeB in via normal CubeSequence indexing.
    >>> cs[1,:,0,:] # doctest: +SKIP
    >>> # Return same slice using this function
    >>> index_sequence_as_cube(cs, (slice(0, cubeB.shape[0]), 0, (slice(0, cubeB.shape[2])) # doctest: +SKIP

    """
    # Determine starting slice of each cube along common axis.
    cumul_cube_lengths = np.cumsum(np.array([c.data.shape[cubesequence._common_axis]
                                             for c in cubesequence.data]))
    # Case 1: Item is int and common axis is 0.
    if isinstance(item, int):
        if cubesequence._common_axis != 0:
            raise ValueError("Input can only be indexed with an int if "
                             "CubeSequence's common axis is 0. common "
                             "axis = {0}".format(cubesequence._common_axis))
        else:
            sequence_index, cube_index = _convert_cube_like_index_to_sequence_indices(
                item, cumul_cube_lengths)
            item_list = [item]
    # Case 2: Item is slice and common axis is 0.
    elif isinstance(item, slice):
        if cubesequence._common_axis != 0:
            raise ValueError("Input can only be sliced with a single slice if "
                             "CubeSequence's common axis is 0. common "
                             "axis = {0}".format(cubesequence._common_axis))
        else:
            sequence_index, cube_index = _convert_cube_like_slice_to_sequence_slices(
                item, cumul_cube_lengths)
            item_list = [item]
    # Case 3: Item is tuple and common axis index is int.ance(item[cubesequence._common_axis], int):
        # Since item must be a tuple, convert to list to
        # make ensure it's mutable for next cases.
        item_list = list(item)
        # Check item is long enough to include common axis.
        if len(item_list) < cubesequence._common_axis:
            raise ValueError("Input item not long enough to include common axis."
                             "Must have length of of between "
                             "{0} and {1} inclusive.".format(
                                 cubesequence._common_axis, len(cubesequence[0].data.shape)))
        sequence_index, cube_index = _convert_cube_like_index_to_sequence_indices(
            item_list[cubesequence._common_axis], cumul_cube_lengths)
    # Case 4: Item is tuple and common axis index is slice.
    elif isinstance(item[cubesequence._common_axis], slice):
        # Since item must be a tuple, convert to list to
        # make ensure it's mutable for next cases.
        item_list = list(item)
        # Check item is long enough to include common axis.
        if len(item_list) < cubesequence._common_axis:
            raise ValueError("Input item not long enough to include common axis."
                             "Must have length of of between "
                             "{0} and {1} inclusive.".format(
                                 cubesequence._common_axis, len(cubesequence[0].data.shape)))
        sequence_index, cube_index = _convert_cube_like_slice_to_sequence_slices(
            item_list[cubesequence._common_axis], cumul_cube_lengths)
    else:
        raise ValueError("Invalid index/slice input.")
    # Replace common axis index/slice with corresponding
    # index/slice with cube.
    item_list[cubesequence._common_axis] = cube_index
    # Insert corresponding index/slice of required cube in sequence.
    item_list.insert(0, sequence_index)
    item_tuple = tuple(item_list)
    if item is None or (isinstance(item, tuple) and None in item):
        raise IndexError("None indices not supported")

    return get_cube_from_sequence(cubesequence, item_tuple)


def convert_index_as_cube_item_to_sequence_slices(cubesequence, item):
    """
    Converts slice item input to NDCubeSequence.index_as_cube to list of SequenceSlices.
    """
    # Determine starting slice of each cube along common axis.
    cumul_cube_lengths = np.cumsum(np.array([c.data.shape[cubesequence._common_axis]
                                             for c in cubesequence.data]))
    invalid_item_error_message = "Invalid index/slice input."
    # Case 1: Item is int and common axis is 0.
    if isinstance(item, int):
        if cubesequence._common_axis != 0:
            raise ValueError("Input can only be indexed with an int if "
                             "CubeSequence's common axis is 0. common "
                             "axis = {0}".format(cubesequence._common_axis))
        else:
            sequence_index, cube_index = _convert_cube_like_index_to_sequence_indices(
                item, cumul_cube_lengths)
            sequence_slices = get_sequence_slices_from_int_item(sequence_index, cube_index)
    # Case 2: Item is slice and common axis is 0.
    elif isinstance(item, slice):
        if cubesequence._common_axis != 0:
            raise ValueError("Input can only be sliced with a single slice if "
                             "CubeSequence's common axis is 0. common "
                             "axis = {0}".format(cubesequence._common_axis))
        else:
            sequence_slices = _convert_cube_like_slice_to_sequence_slices(
                item, cumul_cube_lengths)
    # Case 3: Item is tuple.
    elif isinstance(item, tuple):
        # Check item is long enough to include common axis.
        if len(item) < cubesequence._common_axis:
            raise ValueError("Input item not long enough to include common axis."
                             "Must have length between "
                             "{0} and {1} inclusive.".format(
                                 cubesequence._common_axis, len(cubesequence[0].data.shape)))
        # Use common axis index/slice to generate slices for sequence.
        if isinstance(item[cubesequence._common_axis], int):
            sequence_index, cube_index = _convert_cube_like_index_to_sequence_indices(
            item[cubesequence._common_axis], cumul_cube_lengths)
        elif isinstance(item[cubesequence._common_axis], slice):
            sequence_index, cube_index = _convert_cube_like_slice_to_sequence_slices(
            item[cubesequence._common_axis], cumul_cube_lengths)
        else:
            raise ValueError(invalid_item_error_message)
        sequence_item = [slice(None) for i in range(len(cubesequence.dimensions))]
        sequence_item[0] = sequence_index
        sequence_item[cubesequence._common_axis+1] = cube_index
        sequence_slices = get_sequence_slices_from_tuple_item(sequence_item)
    else:
        raise ValueError(invalid_item_error_message)
    return sequence_slices


def _convert_cube_like_index_to_sequence_indices(cube_like_index, cumul_cube_lengths):
    # so that it returns the correct sequence_index and cube_index as
    # np.where(cumul_cube_lengths <= cube_like_index) returns NULL.
    if cube_like_index < cumul_cube_lengths[0]:
        sequence_index = 0
        cube_index = cube_like_index
    else:
        sequence_index = np.where(cumul_cube_lengths <= cube_like_index)[0][-1]
        # if the cube is out of range then return the last index
        if cube_like_index > cumul_cube_lengths[-1] - 1:
            if len(cumul_cube_lengths) == 1:
                cube_index = cumul_cube_lengths[-1] - 1
            else:
                cube_index = cumul_cube_lengths[-1] - cumul_cube_lengths[-2] - 1
        else:
            cube_index = cube_like_index - cumul_cube_lengths[sequence_index]
        # sequence_index should be plus one as the sequence_index earlier is
        # previous index if it is not already the last cube index.
        if sequence_index < cumul_cube_lengths.size - 1:
            sequence_index += 1
    # Return sequence and cube indices.  Ensure they are int, rather
    # than np.int64 to avoid confusion in checking type elsewhere.
    return int(sequence_index), int(cube_index)


def _convert_cube_like_slice_to_sequence_indices(cube_like_slice, cumul_cube_lengths):
    if cube_like_slice.start is None:
        cube_like_slice_start = 0
    else:
        cube_like_slice_start = cube_like_slice.start
    if cube_like_slice.stop is None:
        cube_like_slice_stop = cumul_cube_lengths[-1]
    else:
        cube_like_slice_stop = cube_like_slice.stop
    sequence_start_index, cube_start_index = _convert_cube_like_index_to_sequence_indices(
        cube_like_slice_start, cumul_cube_lengths)
    sequence_stop_index, cube_stop_index = _convert_cube_like_index_to_sequence_indices(
            cube_like_slice_stop, cumul_cube_lengths)
    if cube_like_slice.stop is None:
        sequence_stop_index = len(cumul_cube_lengths)-1
    else:
        if not cube_like_slice.stop < cumul_cube_lengths[-1]:
            # as _convert_cube_like_index_to_sequence_indices function returns last
            # cube index so we need to increment it by one and set the cube_stop_index
            # as 0 as the function returns the last index of the cube.
            cube_stop_index = 0
            sequence_stop_index += 1
    # if the start and end sequence index are not equal implies slicing across cubes.
    if sequence_start_index != sequence_stop_index:
        # the first slice of cube_slice will be cube_start_index and the length of
        # that cube's end index
        # only storing those cube_slice that needs to be changed.
        # Like if sequence_slice is slice(0, 3) meaning - 0, 1, 2 cubes this means we will
        # store only 0th index slice and 2nd index slice in this list.
        cube_slices = [slice(cube_start_index, cumul_cube_lengths[
                             sequence_start_index], cube_like_slice.step)]

        # for cube over which slices occur appending them
        # for i in range(sequence_start_index+1, sequence_stop_index):
        #     cube_slice.append(slice(0, cumul_cube_lengths[i]-cumul_cube_lengths[i-1]))
        # if the stop index is 0 then slice(0, 0) is not taken. slice(0,3)
        # represent 0,1,2 not 0,1,2,3.
        if int(cube_stop_index) is not 0:
            cube_slices.append(slice(0, cube_stop_index, cube_like_slice.step))
            sequence_indices = range(sequence_start_index,
                                     sequence_stop_index+1)
        else:
            sequence_indices = range(sequence_start_index, sequence_stop_index, cube_like_slice.step)
    else:
        cube_slices = [slice(cube_start_index, cube_stop_index, cube_like_slice.step)]
        sequence_indices = range(sequence_start_index, sequence_stop_index+1, cube_like_slice.step)
    while len(cube_slices) < len(sequence_indices):
        # ATTENTION: What happens when step is not multiple of cube lengths?
        cube_slices.insert(1, slice(None, None, cube_like_slice.step))
    sequence_slices = []
    for i in range(len(sequence_indices)):
        sequence_slices.append(SequenceSlice(sequence_indices[i], cube_slices[i]))
    return sequence_slices


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


class SequenceSlice(object):
    """
    Holds index of an NDCube within NDCubeSequence and a slice item to be applied to cube.

    Used in slicing NDCubeSequences.

    Parameters
    ----------
    sequence_index: `int`
        index of NDCube within NDCubeSequence.data.

    cube_slice: `slice` or `tuple`
        Slice to be applied to NDCube at NDCubeSequence.data[sequence_index].

    """
    def __init__(self, sequence_index, cube_slice):
        if not isinstance(sequence_index, int):
            raise TypeError("sequence_index must be ant int.")
        self.sequence_index = sequence_index
        self.cube_slice = cube_slice

    def __repr__(self):
        return ("SequenceSlice\n-------------\nSequence Index: {0}\nCube Slice: {1}\n".format(
            self.sequence_index, self.cube_slice))
