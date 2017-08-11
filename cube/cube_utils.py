# -*- coding: utf-8 -*-
# Author: Mateo Inchaurrandieta <mateo.inchaurrandieta@gmail.com>
# pylint: disable=E1101, C0330
"""
Utilities used in the sunpy.cube.cube module. Moved here to prevent clutter and
aid readability.
"""

from __future__ import absolute_import
import numpy as np
from sunpycube import wcs_util
from astropy import units as u
from copy import deepcopy


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


def get_cube_from_sequence(cubesequence, item):
    """
    Handles CubeSequence's __getitem__ method for list of cubes.

    Parameters
    ----------
    cubesequence: sunpycube.CubeSequence object
        The cubesequence to get the item from
    item: int, slice object, or tuple of these
        The item to get from the cube
    """
    if isinstance(item, int):
        result = cubesequence.data[item]
    if isinstance(item, slice):
        data = cubesequence.data[item]
        result = cubesequence._new_instance(
            data, meta=cubesequence.meta, common_axis=cubesequence.common_axis)
    if isinstance(item, tuple):
        # if the 0th index is int.
        if isinstance(item[0], int):
            # to satisfy something like cubesequence[0,0] this should have data type
            # as cubesequence[0][0]
            if len(item[1::]) is 1:
                result = cubesequence.data[item[0]][item[1]]
            else:
                result = cubesequence.data[item[0]][item[1::]]
        # if the 0th index is slice.
        # used for the index_sequence_as_cube function. Slicing across cubes.
        # item represents (slice(start_cube_index, end_cube_index, None),
        # [slice_of_start_cube, slice_of_end_cube]) if end cube is not sliced then length is 1.
        if isinstance(item[0], slice):
            data = cubesequence.data[item[0]]
            # applying the slice in the start of cube.
            data[0] = data[0][item[1][0]]
            if len(item[1]) is 2:
                # applying the slice in the end of cube.
                data[-1] = data[-1][item[1][-1]]
            # applying the rest of the item in all the cubes.
            for i, cube in enumerate(data):
                if len(item[2::]) is 1:
                    data[i] = cube[item[2]]
                else:
                    data[i] = cube[item[2::]]
            result = cubesequence._new_instance(
                data, meta=cubesequence.meta, common_axis=cubesequence.common_axis)
    return result


def index_sequence_as_cube(cubesequence, item):
    """
    Enables CubeSequence to be indexed as a single Cube.

    This is only possible if cubesequence.common_axis is set,
    i.e. if the Cubes are sequence in order along one of the Cube axes.
    For example, if cubesequence.common_axis=1 where the first axis is
    time, and the Cubes are sequence chronologically such that the last
    time slice of one Cube is directly followed in time by the first time
    slice of the next Cube, then this function allows the CubeSequence to
    be indexed as though all Cubes were combined into one ordered along
    the time axis.

    Parameters
    ----------
    cubesequence: sunpycube.CubeSequence object
        The cubesequence to get the item from
    item: int, slice object, or tuple of these
        The item to get from the cube.  If tuple length must be <= number
        of dimensions in single Cube.

    Example
    -------
    >>> # Say we have three Cubes each cube has common_axis=1 is time and shape=(3,3,3)
    >>> data_list = [cubeA, cubeB, cubeC]
    >>> cs = CubeSequence(data_list, meta=None, common_axis=1)
    >>> # return zeroth time slice of cubeB in via normal CubeSequence indexing.
    >>> cs[1,:,0,:]
    >>> # Return same slice using this function
    >>> index_sequence_as_cube(cs, (slice(0, cubeB.shape[0]), 0, (slice(0, cubeB.shape[2]))

    """
    # Determine starting slice of each cube along common axis.
    cumul_cube_lengths = np.cumsum(np.array([c.data.shape[cubesequence.common_axis]
                                             for c in cubesequence.data]))
    # Case 1: Item is int and common axis is 0. Not yet supported.
    if isinstance(item, int):
        if cubesequence.common_axis != 0:
            raise ValueError("Input can only be indexed with an int if "
                             "CubeSequence's common axis is 0. common "
                             "axis = {0}".format(cubesequence.common_axis))
        else:
            sequence_index, cube_index = _convert_cube_like_index_to_sequence_indices(
                item, cumul_cube_lengths)
            item_list = [item]
    # Case 2: Item is slice and common axis is 0.
    elif isinstance(item, slice):
        if cubesequence.common_axis != 0:
            raise ValueError("Input can only be sliced with a single slice if "
                             "CubeSequence's common axis is 0. common "
                             "axis = {0}".format(cubesequence.common_axis))
        else:
            sequence_index, cube_index = _convert_cube_like_slice_to_sequence_slices(
                item, cumul_cube_lengths)
            item_list = [item]
    # Case 3: Item is tuple and common axis index is int.
    elif isinstance(item[cubesequence.common_axis], int):
        # Since item must be a tuple, convert to list to
        # make ensure it's mutable for next cases.
        item_list = list(item)
        # Check item is long enough to include common axis.
        if len(item_list) < cubesequence.common_axis:
            raise ValueError("Input item not long enough to include common axis."
                             "Must have length of of between "
                             "{0} and {1} inclusive.".format(
                                 cubesequence.common_axis, len(cubesequence[0].data.shape)))
        sequence_index, cube_index = _convert_cube_like_index_to_sequence_indices(
            item_list[cubesequence.common_axis], cumul_cube_lengths)
    # Case 4: Item is tuple and common axis index is slice.
    elif isinstance(item[cubesequence.common_axis], slice):
        # Since item must be a tuple, convert to list to
        # make ensure it's mutable for next cases.
        item_list = list(item)
        # Check item is long enough to include common axis.
        if len(item_list) < cubesequence.common_axis:
            raise ValueError("Input item not long enough to include common axis."
                             "Must have length of of between "
                             "{0} and {1} inclusive.".format(
                                 cubesequence.common_axis, len(cubesequence[0].data.shape)))
        sequence_index, cube_index = _convert_cube_like_slice_to_sequence_slices(
            item_list[cubesequence.common_axis], cumul_cube_lengths)
    else:
        raise ValueError("Invalid index/slice input.")
    # Replace common axis index/slice with corresponding
    # index/slice with cube.
    item_list[cubesequence.common_axis] = cube_index
    # Insert corresponding index/slice of required cube in sequence.
    item_list.insert(0, sequence_index)
    item_tuple = tuple(item_list)
    if item is None or (isinstance(item, tuple) and None in item):
        raise IndexError("None indices not supported")
    return get_cube_from_sequence(cubesequence, item)


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
            cube_index = cumul_cube_lengths[0] - 1
        else:
            cube_index = cube_like_index - cumul_cube_lengths[sequence_index]
        # sequence_index should be plus one as the sequence_index earlier is
        # previous index if it is not already the last cube index.
        if sequence_index < cumul_cube_lengths.size - 1:
            sequence_index += 1
    return sequence_index, cube_index


def _convert_cube_like_slice_to_sequence_slices(cube_like_slice, cumul_cube_lengths):
    if cube_like_slice.start is not None:
        sequence_start_index, cube_start_index = _convert_cube_like_index_to_sequence_indices(
            cube_like_slice.start, cumul_cube_lengths)
    else:
        sequence_start_index, cube_start_index = _convert_cube_like_index_to_sequence_indices(
            0, cumul_cube_lengths)
    if cube_like_slice.stop is not None:
        sequence_stop_index, cube_stop_index = _convert_cube_like_index_to_sequence_indices(
            cube_like_slice.stop, cumul_cube_lengths)
    else:
        sequence_stop_index, cube_stop_index = _convert_cube_like_index_to_sequence_indices(
            cumul_cube_lengths[-1], cumul_cube_lengths)
    if cube_like_slice.stop is not None:
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
        cube_slice = [slice(cube_start_index, cumul_cube_lengths[
                            sequence_start_index], cube_like_slice.step)]

        # for cube over which slices occur appending them
        # for i in range(sequence_start_index+1, sequence_stop_index):
        #     cube_slice.append(slice(0, cumul_cube_lengths[i]-cumul_cube_lengths[i-1]))
        # if the stop index is 0 then slice(0, 0) is not taken. slice(0,3)
        # represent 0,1,2 not 0,1,2,3.
        if int(cube_stop_index) is not 0:
            cube_slice.append(slice(0, cube_stop_index, cube_like_slice.step))
            sequence_slice = slice(sequence_start_index,
                                   sequence_stop_index+1, cube_like_slice.step)
        else:
            sequence_slice = slice(sequence_start_index, sequence_stop_index, cube_like_slice.step)
    else:
        cube_slice = slice(cube_start_index, cube_stop_index, cube_like_slice.step)
        sequence_slice = slice(sequence_start_index, sequence_stop_index+1, cube_like_slice.step)
    return sequence_slice, cube_slice
