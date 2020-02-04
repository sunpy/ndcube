import collections.abc

import numpy as np

from ndcube import NDCube
from ndcube.utils.cube import convert_extra_coords_dict_to_input_format


class NDCollection(collections.abc.Sequence):
    def __init__(self, data, labels, aligned_axes=None, _sanitize_aligned_axes=True):
        """
        A class for holding and manipulating a collection of aligned NDCube or NDCubeSequences.

        Data cubes/sequences must be aligned, i.e. have the same WCS and be the same shape.

        Parameters
        ----------
        data: sequence of `~ndcube.NDCube` or `~ndcube.NDCubeSequence`
            The data cubes/sequences to held in the collection.

        labels: sequence of `str`
            Name of each cube/sequence. Each label must be unique and
            there must be one per element in the data input.

        aligned_axes: `tuple` of `int`, `tuple` of `tuple`s of `int`
            Axes of each cube/sequence that are aligned in numpy order.
            If elements are int, then the same axis numbers in all cubes/sequences are aligned.
            If elements are tuples of ints, then must be one tuple for every cube/sequence.
            Each element of each tuple gives the axes of each cube/sequence that are aligned.
            Default is all axes are aligned.

        Example
        -------
        Say the collection holds two NDCubes, each of 3 dimensions.
        aligned_axes = (1, 2)
        means that axis 1 (0-based counting) of cube0 is aligned with axis 1 of cube1,
        and axis 2 of cube0 is aligned with axis 2 of cube1.
        However, if
        aligned_axes = ((0, 1), (2, 1))
        then the first tuple corresponds to cube0 and the second with cube1.
        This is interpretted as axis 0 of cube0 is aligned with axis 2 of cube1 while
        axis 1 of cube0 is aligned with axis 1 of cube1.

        """
        # Check inputs
        # Ensure there are no duplicate labels
        if len(set(labels)) != len(labels):
            raise ValueError("Duplicate labels detected.")
        if len(labels) != len(data):
            raise ValueError("Data and labels inputs of different lengths.")

        n_cubes = len(data)
        self.data = dict(zip(labels, data))
        self.labels = tuple(labels)

        # If aligned_axes not set, assume all axes are aligned in order.
        if aligned_axes is None:
            # Check all cubes are of same shape
            cube0_dims = data[0].dimensions
            cubes_same_shape = all(
                [data[i].dimensions == cube0_dims for i in range(n_cubes)])
            if cubes_same_shape is not True:
                raise ValueError(
                    "All cubes in data not of same shape. Please set aligned_axes kwarg.")
            self.n_aligned_axes = len(cube0_dims)
            self.aligned_axes = dict([(labels[i], tuple(range(len(cube0_dims))))
                                      for i in range(n_cubes)])
        else:
            if _sanitize_aligned_axes is False:
                self.n_aligned_axes = len(aligned_axes[0])
                self.aligned_axes = dict(zip(labels, aligned_axes))
            else:
                aligned_axes, self.n_aligned_axes = _sanitize_aligned_axes(data, aligned_axes)
                self.aligned_axes = dict(zip(labels, aligned_axes))

    def __getitem__(self, item):
        if isinstance(item, str):
            item = [item]
        if isinstance(item, collections.abc.Sequence):
            item_strings = [isinstance(_item, str) for _item in item]
            item_is_strings = all(item_strings)
            # Ensure strings are not mixed with slices.
            if (not item_is_strings) and (not all(np.invert(item_strings))):
                raise TypeError("Cannot mix labels and non-labels when indexing instance.")
            new_data = [self.data[_item] for _item in item]
            new_labels = item
        else:
            # Define empty lists of slice items to be applied to each cube in collection.
            collection_items = [[slice(None)] * len(cube.dimensions) for cube in self.data]
            # Determine whether any axes are dropped by slicing.
            # If so, remove them from aligned_axes.
            drop_aligned_axes_indices = []
            # If item is an int, first aligned axis is dropped.
            if isinstance(item, int):
                drop_aligned_axes_indices = [0]
                # Insert item to each cube's slice item.
                for i, key in enumerate(self.labels):
                    collection_items[i][self.aligned_axes[key][0]] = item
            # If item is a slice such that only one element is in interval,
            # first aligned axis is dropped.
            elif isinstance(item, slice):
                step = 1 if item.step is None else step = item.step
                if abs((item.stop - item.start) // step) < 2:
                    drop_aligned_axes_indices = [0]
                # Insert item to each cube's slice item.
                for i, key in enumerate(self.labels):
                    collection_items[i][self.aligned_axes[key][0]] = item
            # If item is tuple, search sub-items for ints or 1-interval slices.dd
            elif isinstance(item, tuple):
                # If item is tuple, search sub-items for ints or 1-interval slices.
                if len(item) > self.n_aligned_axes:
                    raise IndexError("Too many indices")
                for i, axis_item in enumerate(item):
                    if isinstance(axis_item, int):
                        drop_aligned_axes_indices.append(i)
                    elif isinstance(axis_item, slice):
                        step = 1 if axis_item.step is None else step = axis_item.step
                        if abs((axis_item.stop - axis_item.start) // step) < 2):
                            drop_aligned_axes_indices.append(i)
                    else:
                        raise TypeError("Unsupported slicing type: {0}".format(axis_item))
                    # Enter slice item into correct index for slice tuple of each cube.
                    for j, key in enumerate(self.labels):
                        collection_items[j][self.aligned_axes[key][i]] = axis_item
            else:
                raise TypeError("Unsupported slicing type: {0}".format(axis_item))
            # Remove dropped axes from aligned_axes.  MUST BE A BETTER WAY TO DO THIS.
            if len(drop_aligned_axes_index) > 0:
                new_aligned_axes = []
                for label in self.labels:
                    new_aligned_axes.append(list(self.aligned_axes[label]))
                    for drop_axis in drop_aligned_axes_indices:
                        new_aligned_axes[-1].pop(drop_axis)
                    new_aligned_axes[-1] = tuple(new_aligned_axes[-1])
                new_aligned_axes = tuple(new_aligned_axes)
            else:
                new_aligned_axes = self.aligned_axes
            new_labels = self.labels
            new_data = [self.data[key][tuple(cube_item)]
                        for key, cube_item in zip(new_labels, collection_items)]
        return self.__class__(new_data, labels=new_labels, aligned_axes=new_aligned_axes)

    def __repr__(self):
        cube_types = type(self.data[self.labels[0]])
        n_cubes = len(self.labels)
        return ("""NDCollection
----------
Cube labels: {labels}
Number of Cubes: {n_cubes}
Cube Types: {cube_types}
Aligned dimensions: {aligned_dims}
Aligned world physical axis types: {align_axis_types}""".format(
    labels=self.labels, n_cubes=n_cubes, cube_types=cube_types,
    aligned_dims=self.aligned_dimensions,
    aligned_axis_types=self.aligned_world_physical_axis_types))

    @property
    def aligned_dimensions(self):
        return self.data[self.labels[0]].dimensions

    @property
    def aligned_world_axis_physical_types(self):
        return self.data[self.labels[0]].world_axis_physical_types


def _sanitize_aligned_axes(data, aligned_axes):
    aligned_axes_error_message = "aligned_axes must contain ints or " + \
            "a tuple of ints for each element in data."
    cube0_dims = data[0].dimensions
    # If user entered a single int, convert to length 1 tuple of int.
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
        if not all([len(axes) == self.n_aligned_axes for axes in aligned_axes]):
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
                    "No. aligned axes: {0};\n".format(self.n_aligned_axes)  + \
                    "Highest aligned axis: {0}".format(max_aligned_axis))
            for axis in aligned_axis[i]:
                subtuple_types = isinstance(axis, int)
                cube_lengths_equal = data[i].dimensions[axis] == cube0_dims[axis]
            subtuples_are_ints[i] = all(subtuple_types)
            aligned_axes_same_length[i] = all(cube_lengths_equal)
        if not all(subtuples_are_ints):
            raise ValueError(aligned_axes_error_message)
        if not all(aligned_axes_same_length):
            raise ValueError("Aligned cube/sequence axes must be of same length.")
    else:
        raise ValueError(aligned_axes_error_message)

    return aligned_axes, n_aligned_axes
