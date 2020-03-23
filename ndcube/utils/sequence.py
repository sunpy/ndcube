
"""
Utilities for ndcube sequence.
"""

from copy import deepcopy
from collections import namedtuple
from functools import singledispatch

import numpy as np
import astropy.units as u


__all__ = ['SequenceSlice', 'SequenceItem', 'slice_sequence', 'convert_item_to_sequence_items',
           'convert_cube_like_item_to_sequence_items', 'convert_slice_nones_to_ints']


SequenceSlice = namedtuple("SequenceSlice", "sequence_index common_axis_item")
"""
Define SequenceSlice named tuple of length 2. Its attributes are:
sequence_index: an int giving the index of a cube within an NDCubeSequence.
common_axis_item: slice of int index of to be to be applied to the common
axis of the cube.
"""
SequenceItem = namedtuple("SequenceItem", "sequence_index cube_item")
"""
Define SequenceItem named tuple of length 2. Its attributes are:
sequence_index: an int giving the index of a cube within an NDCubeSequence.
cube_item: item (int, slice, tuple) to be applied to cube identified
by sequence_index attribute.
"""


def slice_sequence(cubesequence, item):
    """
    Slice an NDCubeSequence given a slicing/index item.

    Parameters
    ----------
    cubesequence: `ndcube.NDCubeSequence`
        The cubesequence to slice.

    item: `int`, `slice`, or `tuple` of `int` and/or `slice`.
        An slice/index item compatible with input to NDCubeSequence.__getitem__.

    Returns
    -------
    result: `NDCubeSequence` or `NDCube`
        The sliced cube sequence.
    """
    if item is None or (isinstance(item, tuple) and None in item):
        raise IndexError("None indices not supported")
    # Convert item to list of SequenceSlices
    sequence_items = convert_item_to_sequence_items(item, len(cubesequence.data))
    return slice_sequence_by_sequence_items(cubesequence, sequence_items)


@singledispatch
def convert_item_to_sequence_items(item, n_cubes=None, cube_item=None):
    """
    Converts NDCubeSequence __getitem__ item to list of SequenceSlice objects.

    Parameters
    ----------
    item: `int`, `slice`, or `tuple` of `int` and/or `slice`.
        An slice/index item compatible with input to NDCubeSequence.__getitem__.

    n_cubes: `int`
        Number of cubes in NDCubeSequence being sliced.  Must be supplied, but
        not used if item type is `int` or `slice`.

    Returns
    -------
    result: `list` of SequenceItem `namedtuple`.
        The slice/index items for each relevant NDCube within the NDCubeSequence
        which together represent the original input slice/index item.
    """
    # If type if the first input of this function does not match the
    # type of first input of one of the below registered functions,
    # raise an error.  Otherwise one of the below registered functions
    # is executed.
    raise TypeError("Unrecognized slice type: {0}", item)


@convert_item_to_sequence_items.register(int)
def _get_sequence_items_from_int_item(int_item, n_cubes=None, cube_item=slice(None)):
    """
    Converts int index of an NDCubeSequence to list of SequenceSlices.

    Parameters
    ----------
    int_item: `int`
        index of NDCube within NDCubeSequence to be slices out.

    n_cubes: `None`
        Not used.  Exists in API to be consistent with API of convert_item_to_sequence_items()
        to which it this function is registered under single dispatch.

    cube_item: `int`, `slice`, or `tuple`
        Item to be applied to selected NDCube.

    Returns
    -------
    result: `list` of SequenceItem `namedtuple`
        The slice/index items for each relevant NDCube within the NDCubeSequence
        which together represent the original input slice/index item.
    """
    return [SequenceItem(int_item, cube_item)]


@convert_item_to_sequence_items.register(slice)
def _get_sequence_items_from_slice_item(slice_item, n_cubes, cube_item=slice(None)):
    """
    Converts slice item of an NDCubeSequence to list of SequenceSlices.

    Parameters
    ----------
    slice_item: `slice`
        Indicates which NDCubes within NDCubeSequence are to be slices out.

    n_cubes: `int`
        Number of cubes in NDCubeSequence being sliced.

    cube_item: `int`, `slice`, or `tuple`
        Item to be applied to each selected NDCube.

    Returns
    -------
    sequence_items: `list` of SequenceItem `namedtuple`.
        The slice/index items for each relevant NDCube within the NDCubeSequence
        which together represent the original input slice/index item.
    """
    # If there are None types in slice, replace with correct entries based on sign of step.
    no_none_slice = convert_slice_nones_to_ints(slice_item, n_cubes)
    # Derive SequenceItems for each cube.  Recall that
    # once convert_slice_nones_to_ints() has been applied, a None will
    # only be present to signify the beginning of the array when the
    # step is negative.  Therefore, if the stop parmeter of the above
    # slice object is None, set the stop condition of the below for
    # loop to -1.
    if no_none_slice.stop is None:
        stop = -1
    else:
        stop = no_none_slice.stop
    # If slice has interval length 1, make sequence index length 1 slice to
    # ensure dimension is not dropped in accordance with slicing convention.
    if abs(stop - no_none_slice.start) == 1:
        sequence_items = [SequenceItem(slice_item, cube_item)]
    else:
        sequence_items = [SequenceItem(i, cube_item)
                          for i in range(no_none_slice.start, stop, no_none_slice.step)]
    return sequence_items


@convert_item_to_sequence_items.register(tuple)
def _get_sequence_items_from_tuple_item(tuple_item, n_cubes, cube_item=None):
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

    cube_item: `None`
        Not used.  Exists in API to be consistent with API of convert_item_to_sequence_items()
        to which it this function is registered under single dispatch.

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
    sequence_items = convert_item_to_sequence_items(tuple_item[0], n_cubes=n_cubes,
                                                    cube_item=cube_item)
    return sequence_items


def slice_sequence_by_sequence_items(cubesequence, sequence_items):
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
        # If sequence item is interval length 1 slice, ensure an NDCubeSequence
        # is returned in accordance with slicing convention.
        # Due to code up to this point, if sequence item is a slice, it can only
        # be an interval length 1 slice.
        if isinstance(sequence_items[0].sequence_index, slice):
            result.data = [result.data[sequence_items[0].sequence_index.start][sequence_items[0].cube_item]]
        else:
            result = result.data[sequence_items[0].sequence_index][sequence_items[0].cube_item]
    else:
        data = [result.data[sequence_item.sequence_index][sequence_item.cube_item]
                for sequence_item in sequence_items]
        result.data = data
    return result


def _index_sequence_as_cube(cubesequence, item):
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
    >>> _index_sequence_as_cube(cs, (slice(0, cubeB.shape[0]), 0,
    ...                             (slice(0, cubeB.shape[2])) # doctest: +SKIP
    """
    # Convert index_as_cube item to a list of regular NDCubeSequence
    # items of each relevant cube.
    common_axis_cube_lengths = np.array([c.data.shape[cubesequence._common_axis]
                                         for c in cubesequence.data])
    sequence_items = convert_cube_like_item_to_sequence_items(item, cubesequence._common_axis,
                                                              common_axis_cube_lengths)
    # Use sequence items to slice NDCubeSequence.
    return slice_sequence_by_sequence_items(cubesequence, sequence_items)


def convert_cube_like_item_to_sequence_items(cube_like_item, common_axis, common_axis_cube_lengths):
    """
    Converts an input item to NDCubeSequence.index_as_cube to a list od
    SequenceSlice objects.

    Parameters
    ----------
    cube_like_item: `int`, `slice`, of `tuple` of `int and/or `slice`.
        Item compatible with input to NDCubeSequence.index_as_cube.

    common_axis: `int`
        Data axis of NDCubes common to NDCubeSequence

    common_axis_cube_lengths: `np.array`
        Length of each cube in sequence along the common axis.

    Returns
    -------
    sequence_items: `list` of SequenceItem `namedtuple`.
        The slice/index items for each relevant NDCube within the NDCubeSequence
        which together represent the original input slice/index item.
    """
    invalid_item_error_message = "Invalid index/slice input."
    # Case 1: Item is int and common axis is 0.
    if isinstance(cube_like_item, int):
        if common_axis != 0:
            raise ValueError("Input can only be indexed with an int if "
                             "CubeSequence's common axis is 0. common "
                             "axis = {}".format(common_axis))
        else:
            # Derive list of SequenceSlice objects that describes the
            # cube_like_item in regular slicing notation.
            sequence_slices = [_convert_cube_like_index_to_sequence_slice(
                cube_like_item, common_axis_cube_lengths)]
            all_axes_item = None
    # Case 2: Item is slice and common axis is 0.
    elif isinstance(cube_like_item, slice):
        if common_axis != 0:
            raise ValueError("Input can only be sliced with a single slice if "
                             "CubeSequence's common axis is 0. common "
                             "axis = {}".format(common_axis))
        else:
            # Derive list of SequenceSlice objects that describes the
            # cube_like_item in regular slicing notation.
            # First ensure None types within slice are replaced with appropriate ints.
            sequence_slices = _convert_cube_like_slice_to_sequence_slices(
                cube_like_item, common_axis_cube_lengths)
            all_axes_item = None
    # Case 3: Item is tuple.
    elif isinstance(cube_like_item, tuple):
        # Check item is long enough to include common axis.
        if len(cube_like_item) < common_axis + 1:
            raise ValueError("Input item not long enough to include common axis."
                             "Must have length > {}".format(common_axis))
        # Based on type of slice/index in the common axis position of
        # the cube_like_item, derive list of SequenceSlice objects that
        # describes the cube_like_item in regular slicing notation.
        if isinstance(cube_like_item[common_axis], int):
            sequence_slices = [_convert_cube_like_index_to_sequence_slice(
                cube_like_item[common_axis], common_axis_cube_lengths)]
        elif isinstance(cube_like_item[common_axis], slice):
            sequence_slices = _convert_cube_like_slice_to_sequence_slices(
                cube_like_item[common_axis], common_axis_cube_lengths)
        else:
            raise ValueError(invalid_item_error_message)
        all_axes_item = cube_like_item
    else:
        raise TypeError("Unrecognized item type.")
    # Convert the sequence slices, that only describe the slicing along
    # the sequence axis and common axis to sequence items which
    # additionally describe how the non-common cube axes should be sliced.
    sequence_items = [_convert_sequence_slice_to_sequence_item(
        sequence_slice, common_axis, cube_like_item=all_axes_item)
        for sequence_slice in sequence_slices]
    return sequence_items


def _convert_cube_like_index_to_sequence_slice(cube_like_index, common_axis_cube_lengths):
    """
    Converts a cube-like index of an NDCubeSequence to indices along the
    sequence and common axes.

    Parameters
    ----------
    cube_like_index: `int`
        Cube-like index of NDCubeSequence

    common_axis_cube_lengths: iterable of `int`
        Length of each cube along common axis.

    Returns
    -------
    sequence_slice: SequenceSlice `namedtuple`.
        First element gives index of cube along sequence axis.
        Second element each index along common axis of relevant cube.
    """
    # Derive cumulative lengths of cubes along common axis.
    cumul_common_axis_cube_lengths = np.cumsum(common_axis_cube_lengths)
    # If cube_like_index is within 0th cube in sequence, it is
    # simple to determine the sequence and common axis indices.
    try:
        index_in_0th_cube = cube_like_index < cumul_common_axis_cube_lengths[0]
    except TypeError as err:
        none_not_int_error_messages = [
            "'>' not supported between instances of 'int' and 'NoneType'",
            "unorderable types: int() > NoneType()"]
        if err.args[0] in none_not_int_error_messages:
            index_in_0th_cube = True
        else:
            raise err
    if index_in_0th_cube:
        sequence_index = 0
        common_axis_index = cube_like_index
    # Else use more in-depth method.
    else:
        # Determine the index of the relevant cube within the sequence
        # from the cumulative common axis cube lengths.
        sequence_index = np.where(cumul_common_axis_cube_lengths <= cube_like_index)[0][-1]
        if cube_like_index > cumul_common_axis_cube_lengths[-1] - 1:
            # If the cube is out of range then return the last common axis index.
            common_axis_index = common_axis_cube_lengths[-1]
        else:
            # Else use simple equation to derive the relevant common axis index.
            common_axis_index = cube_like_index - cumul_common_axis_cube_lengths[sequence_index]
        # sequence_index should be plus one as the sequence_index earlier is
        # previous index if it is not already the last cube index.
        if sequence_index < cumul_common_axis_cube_lengths.size - 1:
            sequence_index += 1
    # Return sequence and cube indices.  Ensure they are int, rather
    # than np.int64 to avoid confusion in checking type elsewhere.
    if common_axis_index is not None:
        common_axis_index = int(common_axis_index)
    return SequenceSlice(int(sequence_index), common_axis_index)


def _convert_cube_like_slice_to_sequence_slices(cube_like_slice, common_axis_cube_lengths):
    """
    Converts common axis slice input to NDCubeSequence.index_as_cube to a list
    of sequence indices.

    Parameters
    ----------
    cube_like_slice: `slice`
        Slice along common axis in NDCubeSequence.index_as_cube item.

    common_axis_cube_lengths: iterable of `int`
        Length of each cube along common axis.

    Returns
    -------
    sequence_slices: `list` of SequenceSlice `namedtuple`.
        List sequence slices (sequence axis, common axis) for each element
        along common axis represented by input cube_like_slice.
    """
    # Ensure any None attributes in input slice are filled with appropriate ints.
    cumul_common_axis_cube_lengths = np.cumsum(common_axis_cube_lengths)
    # Determine sequence indices of cubes included in cube-like slice.
    cube_like_indices = np.arange(cumul_common_axis_cube_lengths[-1])[cube_like_slice]
    n_cube_like_indices = len(cube_like_indices)
    one_step_sequence_slices = np.empty(n_cube_like_indices, dtype=object)
    # Define array of ints for all indices along common axis.
    # This is restricted to range of interest below.
    sequence_int_indices = np.zeros(n_cube_like_indices, dtype=int)
    for i in range(n_cube_like_indices):
        one_step_sequence_slices[i] = _convert_cube_like_index_to_sequence_slice(
            cube_like_indices[i], common_axis_cube_lengths)
        sequence_int_indices[i] = one_step_sequence_slices[i].sequence_index
    unique_index = np.sort(np.unique(sequence_int_indices, return_index=True)[1])
    unique_sequence_indices = sequence_int_indices[unique_index]
    # Convert start and stop cube-like indices to sequence indices.
    first_sequence_index = _convert_cube_like_index_to_sequence_slice(cube_like_slice.start,
                                                                      common_axis_cube_lengths)
    last_sequence_index = _convert_cube_like_index_to_sequence_slice(cube_like_slice.stop,
                                                                     common_axis_cube_lengths)
    # Since the last index of any slice represents
    # 'up to but not including this element', if the last sequence index
    # is the first element of a new cube, elements from the last cube
    # will not appear in the sliced sequence.  Therefore for ease of
    # slicing, we can redefine the final sequence index as the penultimate
    # cube and its common axis index as beyond the range of the
    # penultimate cube's length along the common axis.
    if (last_sequence_index.sequence_index > first_sequence_index.sequence_index and
            last_sequence_index.common_axis_item == 0):
        last_sequence_index = SequenceSlice(
            last_sequence_index.sequence_index - 1,
            common_axis_cube_lengths[last_sequence_index.sequence_index - 1])
    # Iterate through relevant cubes and determine slices for each.
    # Do last cube outside loop as its end index may not correspond to
    # the end of the cube's common axis.
    if cube_like_slice.step is None:
        step = 1
    else:
        step = cube_like_slice.step
    sequence_slices = []
    common_axis_start_index = first_sequence_index.common_axis_item
    j = 0
    while j < len(unique_sequence_indices) - 1:
        # Let i be the index along the sequence axis of the next relevant cube.
        i = unique_sequence_indices[j]
        # Determine last common axis index for this cube.
        common_axis_last_index = common_axis_cube_lengths[i] - (
            (common_axis_cube_lengths[i] - common_axis_start_index) % step)
        # Generate SequenceSlice for this cube and append to list.
        sequence_slices.append(SequenceSlice(
            i, slice(common_axis_start_index,
                     min(common_axis_last_index + 1, common_axis_cube_lengths[i]), step)))
        # Determine first common axis index for next cube.
        if common_axis_cube_lengths[i] == common_axis_last_index:
            common_axis_start_index = step - 1
        else:
            common_axis_start_index = \
                step - (((common_axis_cube_lengths[i] - common_axis_last_index) % step) +
                        cumul_common_axis_cube_lengths[unique_sequence_indices[j + 1] - 1] -
                        cumul_common_axis_cube_lengths[i])
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
        Must be same format as output from _convert_cube_like_index_to_sequence_slice.

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
    sequence_item: SequenceItem `namedtuple`.
        Describes sequence index of an NDCube within an NDCubeSequence and the
        slice/index item to be applied to the whole NDCube.
    """
    if cube_like_item is None and common_axis == 0:
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
        while len(cube_item_list) <= common_axis:
            cube_item_list.append(slice(None))
        # Create new sequence slice
        cube_item_list[common_axis] = sequence_slice.common_axis_item
        sequence_item = SequenceItem(sequence_slice.sequence_index, tuple(cube_item_list))
    return sequence_item


def convert_slice_nones_to_ints(slice_item, target_length):
    """
    Converts None types within a slice to the appropriate ints based on object
    to be sliced.

    The one case where a None is left in the slice object is when the step is negative and
    the stop parameter is None, since this scenario cannot be represented with an int stop
    parameter.

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
    if slice_item.step is None:
        step = 1
    else:
        step = slice_item.step
    start = slice_item.start
    stop = slice_item.stop
    if step < 0:
        if slice_item.start is None:
            start = int(target_length)
        stop = slice_item.stop
    else:
        if not slice_item.start:
            start = 0
        if not slice_item.stop:
            stop = int(target_length)
    return slice(start, stop, step)


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
