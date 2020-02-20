import numpy as np

def _sanitize_aligned_axes(data, aligned_axes, n_cubes):
    """
    Converts input aligned_axes to standard format.

    aligned_axes can be supplied by the user in a few ways:
    *. A tuple of tuples of ints, where each tuple corresponds to a cube
    in the collection, and each int designates the an aligned axis in numpy order.
    In this case, the axis represented by the 0th int in the 0th tuple is aligned
    with the 0th int in the 1st tuple and so on.
    *. A single tuple of ints if all aligned axes are in the same order.
    *. A single int if only one axis is aligned and if the aligned axis in each cube
    is in the same order.

    """
    aligned_axes_error_message = "aligned_axes must contain ints or " + \
            "a tuple of ints for each element in data."
    cube0_dims = data[0].dimensions
    # If user entered a single int or string, convert to length 1 tuple of int.
    if isinstance(aligned_axes, int):
        aligned_axes = tuple(aligned_axes)
    if not isinstance(aligned_axes, tuple):
        raise ValueError(aligned_axes_error_message)
    # Check type of each element.
    axes_all_ints = all([isinstance(axis, int) for axis in aligned_axes])
    axes_all_tuples = all([isinstance(axis, tuple) for axis in aligned_axes])
    # If all elements are int, duplicate tuple so there is one for each cube.
    if axes_all_ints is True:

        n_aligned_axes = len(aligned_axes)
        aligned_axes = tuple([aligned_axes for i in range(n_cubes)])

    # If all elements are tuple, ensure there is a tuple for each cube and
    # all elements of each sub-tuple are ints.
    elif axes_all_tuples is True:
        if len(aligned_axes) != n_cubes:
            raise ValueError(
                "aligned_axes must have a tuple for each element in data.")

        n_aligned_axes = len(aligned_axes[0])

        # Ensure all elements of sub-tuples are ints,
        # each tuple has the same number of aligned axes,
        # number of aligned axes are <= number of cube dimensions,
        # and the dimensions of the aligned axes in each cube are the same.
        subtuples_are_ints = [False] * n_cubes
        aligned_axes_same_lengths = [False] * n_cubes
        subtuple_types = [False] * n_aligned_axes
        if not all([len(axes) == n_aligned_axes for axes in aligned_axes]):
            raise ValueError("Each element in aligned_axes must have same length.")
        for i in range(n_cubes):
            # Check each cube has at least as many dimensions as there are aligned axes
            # and that all cubes have enough dimensions to accommodate aligned axes.
            n_cube_dims = len(data[i].dimensions)
            max_aligned_axis = max(aligned_axes[i])
            if n_cube_dims < max([max_aligned_axis, n_aligned_axes]):
                raise ValueError(
                    "Each cube in data must have at least as many axes as aligned axes " + \
                    "and aligned axis numbers must be less than number of cube axes.\n" + \
                    "Cube number: {0};\n".format(i) + \
                    "Number of cube dimensions: {0};\n".format(n_cube_dims) + \
                    "No. aligned axes: {0};\n".format(n_aligned_axes)  + \
                    "Highest aligned axis: {0}".format(max_aligned_axis))
            subtuple_types = [False] * n_aligned_axes
            cube_lengths_equal = [False] * n_aligned_axes
            for j, axis in enumerate(aligned_axes[i]):
                subtuple_types[j] = isinstance(axis, (int, np.integer))
                cube_lengths_equal[j] = data[i].dimensions[axis] == cube0_dims[axis]
            subtuples_are_ints[i] = all(subtuple_types)
            aligned_axes_same_lengths[i] = all(cube_lengths_equal)
        if not all(subtuples_are_ints):
            print([[type(axis) for axis in aligned_axis] for aligned_axis in aligned_axes], subtuple_types)
            raise ValueError(aligned_axes_error_message)
        if not all(aligned_axes_same_lengths):
            raise ValueError("Aligned cube/sequence axes must be of same length.")
    else:
        raise ValueError(aligned_axes_error_message)

    return aligned_axes, n_aligned_axes


def _update_aligned_axes(drop_aligned_axes_indices, aligned_axes):
    # Remove dropped axes from aligned_axes.  MUST BE A BETTER WAY TO DO THIS.
    if len(drop_aligned_axes_indices) > 0:
        new_aligned_axes = []
        for key in aligned_axes.keys():
            cube_aligned_axes = np.array(aligned_axes[key])
            for drop_axis_index in drop_aligned_axes_indices:
                drop_axis = cube_aligned_axes[drop_axis_index]
                cube_aligned_axes = np.delete(cube_aligned_axes, drop_axis_index)
                w = np.where(cube_aligned_axes > drop_axis)
                cube_aligned_axes[w] -= 1
                w = np.where(drop_aligned_axes_indices > drop_axis_index)
                drop_aligned_axes_indices[w] -= 1
            new_aligned_axes.append(tuple(cube_aligned_axes))
        new_aligned_axes = tuple(new_aligned_axes)
    else:
        new_aligned_axes = aligned_axes

    return new_aligned_axes


def slice_interval_is_1(item, axis_length):
    # Make start index numeric and positive.
    if item.start is None:
        start = 0
    else:
        start = item.start
    start = make_index_positive(start, axis_length)

    # Make stop index numeric and positive.
    if item.stop is None:
        stop = axis_length
    else:
        stop = item.stop
    stop = make_index_positive(stop, axis_length)

    # Make step numeric.
    if item.step is None:
        step = 1
    else:
        step = item.step

    # Check if slice interval is 1.
    if abs((stop - start) // step) == 1:
        result = True
    else:
        result = False

    return result

def make_index_positive(index, axis_length):
    if index < 0:
        pos_index = axis_length + index
    else:
        pos_index = index
    return pos_index

