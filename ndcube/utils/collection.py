import numbers

import numpy as np


def _sanitize_aligned_axes(keys, data, aligned_axes):
    if aligned_axes is None:
        return None
    # If aligned_axes set to "all", assume all axes are aligned in order.
    elif isinstance(aligned_axes, str) and aligned_axes.lower() == "all":
        # Check all cubes are of same shape
        cube0_dims = data[0].dimensions
        cubes_same_shape = all([all([d.dimensions[i] == dim for i, dim in enumerate(cube0_dims)])
                                for d in data])
        if cubes_same_shape is not True:
            raise ValueError(
                "All cubes in data not of same shape. Please set aligned_axes kwarg.")
        sanitized_axes = tuple([tuple(range(len(cube0_dims)))] * len(data))
    else:
        # Else, sanitize user-supplied aligned axes.
        sanitized_axes = _sanitize_user_aligned_axes(data, aligned_axes)

    return dict(zip(keys, sanitized_axes))


def _sanitize_user_aligned_axes(data, aligned_axes):
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
    aligned_axes_error_message = ("aligned_axes must contain ints or "
                                  "a tuple of ints for each element in data.")
    if isinstance(data[0].dimensions, tuple):
        cube0_dims = np.array(data[0].dimensions, dtype=object)[np.array(aligned_axes[0])]
    else:
        cube0_dims = data[0].dimensions[np.array(aligned_axes[0])]
    # If user entered a single int or string, convert to length 1 tuple of int.
    if isinstance(aligned_axes, int):
        aligned_axes = (aligned_axes,)
    if not isinstance(aligned_axes, tuple):
        raise ValueError(aligned_axes_error_message)
    # Check type of each element.
    axes_all_ints = all([isinstance(axis, int) for axis in aligned_axes])
    axes_all_tuples = all([isinstance(axis, tuple) for axis in aligned_axes])
    # If all elements are int, duplicate tuple so there is one for each cube.
    n_cubes = len(data)
    if axes_all_ints:
        n_aligned_axes = len(aligned_axes)
        aligned_axes = tuple([aligned_axes for i in range(n_cubes)])

    # If all elements are tuple, ensure there is a tuple for each cube and
    # all elements of each sub-tuple are ints.
    elif axes_all_tuples:
        if len(aligned_axes) != n_cubes:
            raise ValueError("aligned_axes must have a tuple for each element in data.")

        n_aligned_axes = len(aligned_axes[0])

        # Ensure all elements of sub-tuples are ints,
        # each tuple has the same number of aligned axes,
        # number of aligned axes are <= number of cube dimensions,
        # and the dimensions of the aligned axes in each cube are the same.
        subtuples_are_ints = [False] * n_cubes
        aligned_axes_same_lengths = [False] * n_cubes
        if not all([len(axes) == n_aligned_axes for axes in aligned_axes]):
            raise ValueError("Each element in aligned_axes must have same length.")
        for i in range(n_cubes):
            # Check each cube has at least as many dimensions as there are aligned axes
            # and that all cubes have enough dimensions to accommodate aligned axes.
            n_cube_dims = len(data[i].dimensions)
            max_aligned_axis = max(aligned_axes[i])
            if n_cube_dims < max([max_aligned_axis, n_aligned_axes]):
                raise ValueError(
                    "Each cube in data must have at least as many axes as aligned axes "
                    "and aligned axis indices must be less than number of cube axes.\n"
                    f"Cube number: {i};\n"
                    f"Number of cube dimensions: {n_cube_dims};\n"
                    f"No. aligned axes: {n_aligned_axes};\n"
                    f"Highest aligned axis: {max_aligned_axis}")
            subtuple_types = [False] * n_aligned_axes
            cube_lengths_equal = [False] * n_aligned_axes
            for j, axis in enumerate(aligned_axes[i]):
                subtuple_types[j] = isinstance(axis, numbers.Integral)
                cube_lengths_equal[j] = data[i].dimensions[axis] == cube0_dims[j]
            subtuples_are_ints[i] = all(subtuple_types)
            aligned_axes_same_lengths[i] = all(cube_lengths_equal)
        if not all(subtuples_are_ints):
            raise ValueError(aligned_axes_error_message)
        if not all(aligned_axes_same_lengths):
            raise ValueError("Aligned cube/sequence axes must be of same length.")
    else:
        raise ValueError(aligned_axes_error_message)

    # Ensure all aligned axes are of same length.
    check_dimensions = set([len(set([cube.dimensions[cube_aligned_axes[j]]
                                     for cube, cube_aligned_axes in zip(data, aligned_axes)]))
                            for j in range(n_aligned_axes)])
    if check_dimensions != {1}:
        raise ValueError("Aligned axes are not all of same length.")

    return aligned_axes


def _update_aligned_axes(drop_aligned_axes_indices, aligned_axes, first_key):
    # Remove dropped axes from aligned_axes.  MUST BE A BETTER WAY TO DO THIS.
    if len(drop_aligned_axes_indices) <= 0:
        new_aligned_axes = tuple(aligned_axes.values())
    elif len(drop_aligned_axes_indices) == len(aligned_axes[first_key]):
        new_aligned_axes = None
    else:
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

    return new_aligned_axes


def assert_aligned_axes_compatible(data_dimensions1, data_dimensions2, data_axes1, data_axes2):
    """
    Checks whether two sets of aligned axes are compatible.

    Parameters
    ----------
    data_dimensions1: sequence of ints
        The dimension lengths of data cube 1.

    data_dimensions2: sequence of ints
        The dimension lengths of data cube 2.

    data_axes1: `tuple` of `int`
        The aligned axes of data cube 1.

    data_axes2: `tuple` of `int`
        The aligned axes of data cube 2.

    """
    # Confirm same number of aligned axes.
    if len(data_axes1) != len(data_axes2):
        raise ValueError("Number of aligned axes must be equal: "
                         f"{len(data_axes1)} != {len(data_axes2)}")
    # Confirm dimension lengths of each aligned axis is the same.
    if not all(data_dimensions1[np.array(data_axes1)] == data_dimensions2[np.array(data_axes2)]):
        raise ValueError("All corresponding aligned axes between cubes must be of same length.")
