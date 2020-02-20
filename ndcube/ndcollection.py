import collections.abc

import numpy as np

from ndcube.utils.cube import convert_extra_coords_dict_to_input_format
import ndcube.utils.collection as collection_utils

__all__ = ["NDCollection"]

#class NDCollection(collections.abc.Sequence):
class NDCollection:
    def __init__(self, data, keys=None, aligned_axes=None, meta=None, dont_sanitize_aligned_axes=False):
        """
        A class for holding and manipulating a collection of aligned NDCube or NDCubeSequences.

        Data cubes/sequences must be aligned, i.e. have the same WCS and be the same shape.

        Parameters
        ----------
        data: sequence of `~ndcube.NDCube` or `~ndcube.NDCubeSequence`
            The data cubes/sequences to held in the collection.

        keys: sequence of `str`
            Name of each cube/sequence. Each label must be unique and
            there must be one per element in the data input.
            Default is ("0", "1",...)

        aligned_axes: `tuple` of `int`, `tuple` of `tuple`s of `int`
            Axes of each cube/sequence that are aligned in numpy order.
            If elements are int, then the same axis numbers in all cubes/sequences are aligned.
            If elements are tuples of ints, then must be one tuple for every cube/sequence.
            Each element of each tuple gives the axes of each cube/sequence that are aligned.
            Default is all axes are aligned.

        meta: `dict` (Optional)
            General metadata for the overall collection.

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
        # Ensure there are no duplicate keys
        if keys is None:
            keys = np.arange(len(data)).astype("str")
        elif len(set(keys)) != len(keys):
            raise ValueError("Duplicate keys detected.")
        if len(keys) != len(data):
            raise ValueError("Data and keys inputs of different lengths.")

        self.meta = meta

        n_cubes = len(data)
        self.data = dict(zip(keys, data))
        self.keys = tuple(keys)

        # If aligned_axes not set, assume all axes are aligned in order.
        if aligned_axes is None:
            # Check all cubes are of same shape
            cube0_dims = data[0].dimensions
            cubes_same_shape = all(
                [all(data[i].dimensions == cube0_dims) for i in range(n_cubes)])
            if cubes_same_shape is not True:
                raise ValueError(
                    "All cubes in data not of same shape. Please set aligned_axes kwarg.")
            self.n_aligned_axes = len(cube0_dims)
            self.aligned_axes = dict([(keys[i], tuple(range(len(cube0_dims))))
                                      for i in range(n_cubes)])
        else:
            if dont_sanitize_aligned_axes is True:
                self.n_aligned_axes = len(aligned_axes[0])
                self.aligned_axes = dict(zip(keys, aligned_axes))
            else:
                aligned_axes, self.n_aligned_axes = collection_utils._sanitize_aligned_axes(
                        data, aligned_axes, n_cubes)
                self.aligned_axes = dict(zip(keys, aligned_axes))

    def __getitem__(self, item):
        # There are two ways to slice:
        # by key or sequence of keys, i.e. slice out given cubes in the collection, or
        # by typical python numeric slicing API,
        # i.e. slice the each component cube along the aligned axes.

        # If item is single string, slicing is simple.
        if isinstance(item, str):
            return self.data[item]

        # If item is not a single string...
        else:
            # If item is a sequence, ensure strings and numeric items are not mixed.
            item_is_strings = False
            if isinstance(item, collections.abc.Sequence):
                item_strings = [isinstance(_item, str) for _item in item]
                item_is_strings = all(item_strings)
                # Ensure strings are not mixed with slices.
                if (not item_is_strings) and (not all(np.invert(item_strings))):
                    raise TypeError("Cannot mix keys and non-keys when indexing instance.")

            # If sequence is all strings, extract the cubes corresponding to the string keys.
            if item_is_strings:
                new_data = [self.data[_item] for _item in item]
                new_keys = item
                new_aligned_axes=tuple([self.aligned_axes[_item] for _item in item])

            # Else, the item is assumed to be a typical slicing item.
            # Slice each cube in collection using information in this item.
            else:
                # Derive item to be applied to each cube in collection and
                # whether any aligned axes are dropped by the slicing.
                collection_items, new_aligned_axes = self._generate_collection_getitems(item)
                # Apply those slice items to each cube in collection.
                new_data = [self.data[key][tuple(cube_item)]
                            for key, cube_item in zip(self.keys, collection_items)]
                # Since item is not strings, no cube in collection is dropped.
                # Therefore the collection keys remain unchanged.
                new_keys = self.keys

            return self.__class__(new_data, keys=new_keys, aligned_axes=new_aligned_axes,
                                  meta=self.meta, dont_sanitize_aligned_axes=True)

    def _generate_collection_getitems(self, item):
        # There are 3 supported cases of the slice item: int, slice, tuple of ints and/or slices.
        # Compile appropriate slice items for each cube in the collection and
        # and drop any aligned axes that are sliced out.

        # First, define empty lists of slice items to be applied to each cube in collection.
        collection_items = [[slice(None)] * len(self.data[key].dimensions) for key in self.keys]

        # Case 1: int
        # First aligned axis is dropped.
        if isinstance(item, int):
            drop_aligned_axes_indices = [0]
            # Insert item to each cube's slice item.
            for i, key in enumerate(self.keys):
                collection_items[i][self.aligned_axes[key][0]] = item

        # Case 2: slice
        # If only one element in interval of slice, first aligned axis is dropped.
        elif isinstance(item, slice):
            if collection_utils.slice_interval_is_1(item, int(self.aligned_dimensions.value[0])):
                drop_aligned_axes_indices = [0]
            # Insert item to each cube's slice item.
            for i, key in enumerate(self.keys):
                collection_items[i][self.aligned_axes[key][0]] = item

        # Case 3: tuple of ints/slices
        # Search sub-items within tuple for ints or 1-interval slices.
        elif isinstance(item, tuple):
            # Ensure item is not longer than number of aligned axes
            if len(item) > self.n_aligned_axes:
                raise IndexError("Too many indices")
            # If item is tuple, search sub-items for ints or 1-interval slices.
            drop_aligned_axes_indices = []
            for i, axis_item in enumerate(item):
                if isinstance(axis_item, int):
                    drop_aligned_axes_indices.append(i)
                elif isinstance(axis_item, slice):
                    slice_interval_is_1 = collection_utils.slice_interval_is_1(
                            axis_item, int(self.aligned_dimensions.value[i]))
                    if slice_interval_is_1:
                        drop_aligned_axes_indices.append(i)
                else:
                    raise TypeError("Unsupported slicing type: {0}".format(axis_item))
                # Enter slice item into correct index for slice tuple of each cube.
                for j, key in enumerate(self.keys):
                    collection_items[j][self.aligned_axes[key][i]] = axis_item

        else:
            raise TypeError("Unsupported slicing type: {0}".format(axis_item))

        # Use indices of dropped axes determine above to update aligned_axes
        # by removing any that have been dropped.
        drop_aligned_axes_indices = np.array(drop_aligned_axes_indices)
        new_aligned_axes = collection_utils._update_aligned_axes(drop_aligned_axes_indices,
                                                                 self.aligned_axes)

        return collection_items, new_aligned_axes


    def __repr__(self):
        cube_types = type(self.data[self.keys[0]])
        n_cubes = len(self.keys)
        return ("""NDCollection
----------
Cube keys: {keys}
Number of Cubes: {n_cubes}
Cube Types: {cube_types}
Aligned dimensions: {aligned_dims}
Aligned world physical axis types: {aligned_axis_types}""".format(
    keys=self.keys, n_cubes=n_cubes, cube_types=cube_types,
    aligned_dims=self.aligned_dimensions,
    aligned_axis_types=self.aligned_world_axis_physical_types))

    @property
    def aligned_dimensions(self):
        return self.data[self.keys[0]].dimensions

    @property
    def aligned_world_axis_physical_types(self):
        return self.data[self.keys[0]].world_axis_physical_types

