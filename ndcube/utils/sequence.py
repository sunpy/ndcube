
"""
Utilities for ndcube sequence.
"""

from copy import deepcopy
from collections import namedtuple

import astropy.units as u
import numpy as np

__all__ = ['SequenceItem',
           'cube_like_index_to_sequence_and_common_axis_indices',
           'cube_like_tuple_item_to_sequence_items']


SequenceItem = namedtuple("SequenceItem", "sequence_index cube_item")
"""
Define SequenceItem named tuple of length 2. Its attributes are:
sequence_index: an int giving the index of a cube within an NDCubeSequence.
cube_item: item (int, slice, tuple) to be applied to cube identified
by sequence_index attribute.
"""


def _get_axis_extra_coord_names_and_units(cube_list, axis):
    """
    Retrieve all extra coord names and units assigned to a data axis along a
    sequence of cubes.

    Parameters
    ----------
    cube_list: `list` of `ndcube.NDCube`
       The sequence of cubes from which to extract the extra coords.

    axis: `int`
       Number of axis (in data/numpy ordering convention).

    Returns
    -------
    axis_coord_names: `ndarray` of `str`
        Names of all extra coordinates in sequence.

    axis_coord_units: `ndarray` of `astropy.unit.unit`
        Units of extra coordinates.
    """
    # Define empty lists to hold results.
    axis_coord_names = []
    axis_coord_units = []
    # Extract extra coordinate names and units (if extra coord is a
    # quantity) from each cube.
    for cube in cube_list:
        all_extra_coords = cube.extra_coords
        if all_extra_coords is not None:
            all_extra_coords_keys = list(all_extra_coords.keys())
            for coord_key in all_extra_coords_keys:
                if all_extra_coords[coord_key]["axis"] == axis:
                    axis_coord_names.append(coord_key)
                    if isinstance(all_extra_coords[coord_key]["value"], u.Quantity):
                        axis_coord_units.append(all_extra_coords[coord_key]["value"].unit)
                    else:
                        axis_coord_units.append(None)
    # Extra coords common between cubes will be repeated.  Get rid of
    # duplicate names and then only keep the units corresponding to
    # the first occurence of that name.
    if len(axis_coord_names) > 0:
        axis_coord_names, ind = np.unique(np.asarray(axis_coord_names), return_index=True)
        axis_coord_units = np.asarray(axis_coord_units)[ind]
    else:
        axis_coord_names = None
        axis_coord_units = None
    return axis_coord_names, axis_coord_units


def _get_int_axis_extra_coords(cube_list, axis_coord_names, axis_coord_units, axis):
    """
    Retrieve all extra coord names and units assigned to a data axis along a
    sequence of cubes.

    Parameters
    ----------
    cube_list: `list` of `ndcube.NDCube`
       The sequence of cubes from which to extract the extra coords.

    axis_coord_names: `ndarray` of `str`
        Names of all extra coordinates in sequence.

    axis_coord_units: `ndarray` of `astropy.unit.unit`
        Units of extra coordinates.

    axis: `int`
       Number of axis (in data/numpy ordering convention).

    Returns
    -------
    axis_extra_coords: `dict`
        Extra coords along given axis.
    """
    # Define empty dictionary which will hold the extra coord
    # values not assigned a cube data axis.
    axis_extra_coords = {}
    # Iterate through cubes and populate values of each extra coord
    # not assigned a cube data axis.
    cube_extra_coords = [cube.extra_coords for cube in cube_list]
    for i, coord_key in enumerate(axis_coord_names):
        coord_values = []
        for j, cube in enumerate(cube_list):
            # Construct list of coord values from each cube for given extra coord.
            try:
                if isinstance(cube_extra_coords[j][coord_key]["value"], u.Quantity):
                    cube_coord_values = \
                        cube_extra_coords[j][coord_key]["value"].to(axis_coord_units[i]).value
                else:
                    cube_coord_values = np.asarray(cube_extra_coords[j][coord_key]["value"])
                coord_values = coord_values + list(cube_coord_values)
            except KeyError:
                # If coordinate not in cube, set coordinate values to NaN.
                coord_values = coord_values + [np.nan] * cube.dimensions[axis].value
        # Enter sequence extra coord into dictionary
        if axis_coord_units[i]:
            axis_extra_coords[coord_key] = coord_values * axis_coord_units[i]
        else:
            axis_extra_coords[coord_key] = np.asarray(coord_values)
    return axis_extra_coords


def cube_like_index_to_sequence_and_common_axis_indices(cube_like_index, common_axis,
                                                        common_axis_lengths):
    """
    Converts a cube-like index for an NDCubeSequence to a sequence index and a common axis index.

    The sequence index is the index of the relevant NDCube in the sequence
    while the common axis index is the index within that cube along the common axis
    to which the input cube-like index corresponds.

    Parameters
    ----------
    cube_like_index: `int`

    common_axis_lengths: iterable of `int`
        The lengths of each cube in the sequence along the common axis.

    Returns
    -------
    sequence_index: `int`
        Index of the cube in the sequence in which the cube-like index can be found.

    common_axis_index: `int`
        The index along the cube's common axis to which the input cube-like index corresponds.
    """
    cumul_lengths = np.cumsum(common_axis_lengths)
    sequence_index = np.arange(len(cumul_lengths))[cumul_lengths > cube_like_index][0]
    if sequence_index == 0:
        common_axis_index = cube_like_index
    else:
        common_axis_index = cube_like_index - cumul_lengths[sequence_index - 1]
    return sequence_index, common_axis_index


def cube_like_tuple_item_to_sequence_items(item, common_axis, common_axis_lengths, n_cube_dims):
    """
    Convert a tuple for slicing an NDCubeSequence in the cube-like API to a list of SequenceItems.

    This requires the common_axis item to be a slice item.
    If it is an int, this function should not be used.

    Parameters
    ----------
    item: iterable of `int` or `slice`
        The slicing item.  The common axis entry must be a `slice`

    common_axis: `int`
        The index of the item corresponding to the common axis.

    common_axis_lengths: iterable of `int`
        The lengths of each cube in the sequence along the common axis.

    n_cube_dims: `int`
        The number of dimensions in the cubes in the sequence.

    Returns
    -------
    sequence_items: `list` of `SequenceItem`
        The sequence index and slicing item for each cube in the sequence to be included
        in the NDCubeSequence that would result by applying the input slicing item
        via the cube-like API.
    """
    if not hasattr(item, "__len__"):
        raise TypeError("item must be an iterable of slices and/or ints.")
    if len(item) <= common_axis:
        raise ValueError("item must be include an entry for the common axis, "
                         "i.e. length of item must be > common_axis.")
    if not isinstance(item[common_axis], slice):
        raise TypeError("This function should only be used when the common axis entry "
                        "of item is a slice object.")
    # Define default item for slicing the cubes
    default_cube_item = list(item)
    default_cube_item[common_axis] = slice(None)

    # Convert start and stop cube-like indices to sequence and cube indices.
    if item[common_axis].start is None:
        common_axis_start = 0
    else:
        common_axis_start = item[common_axis].start
    if item[common_axis].stop is None:
        common_axis_stop = sum(common_axis_lengths)
    else:
        common_axis_stop = item[common_axis].stop
    item[common_axis] = slice(common_axis_start, common_axis_stop)
    start_sequence_index, start_common_axis_index = \
        cube_like_index_to_sequence_and_common_axis_indices(
            item[common_axis].start, common_axis, common_axis_lengths)
    stop_sequence_index, stop_common_axis_index = \
        cube_like_index_to_sequence_and_common_axis_indices(
            item[common_axis].stop - 1, common_axis, common_axis_lengths)
    stop_common_axis_index += 1
    # In the two lines above, the stop index was decremented by one in the
    # calculation of the stop sequence axis to avoid ticking over to new NDCube if not needed.
    # Once the correct sequence index was found,
    # 1 was added back onto the corresponding common axis index.

    # Create iterable of SequenceItems to tell us how to slice each cube.
    n_cubes_after_first = stop_sequence_index - start_sequence_index
    # If only one cube included in slicing item, return iterable of single SequenceItem
    if n_cubes_after_first == 0:
        cube_item = deepcopy(default_cube_item)
        cube_item[common_axis] = slice(start_common_axis_index, stop_common_axis_index)
        sequence_items = [SequenceItem(start_sequence_index, tuple(cube_item))]
    else:
        if n_cubes_after_first > 1:  # Condition > 1 to exclude final cube. We'll add it later.
            sequence_items = [SequenceItem(i, tuple(default_cube_item))
                              for i in range(start_sequence_index + 1, stop_sequence_index)]
        else:
            sequence_items = []
        # Insert final cube if there is one after the first.
        if n_cubes_after_first > 0:
            final_cube_item = deepcopy(default_cube_item)
            final_cube_item[common_axis] = slice(0, stop_common_axis_index)
            sequence_items.append(SequenceItem(stop_sequence_index, final_cube_item))
        # Finally, add Sequence index for first cube.
        first_cube_item = deepcopy(default_cube_item)
        first_cube_item[common_axis] = slice(start_common_axis_index, None)
        sequence_items.insert(0, SequenceItem(start_sequence_index, first_cube_item))
    return sequence_items
