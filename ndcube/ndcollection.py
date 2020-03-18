import collections.abc
import copy

import numpy as np

from ndcube import NDCube, NDCubeSequence
from ndcube.utils.cube import convert_extra_coords_dict_to_input_format
import ndcube.utils.collection as collection_utils

__all__ = ["NDCollection"]

class NDCollection(dict):
    def __init__(self, data, keys, aligned_axes="all", meta=None, dont_sanitize_aligned_axes=False):
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

        aligned_axes: `tuple` of `int`, `tuple` of `tuple`s of `int`, or None, optional
            Axes of each cube/sequence that are aligned in numpy order.
            If elements are int, then the same axis numbers in all cubes/sequences are aligned.
            If elements are tuples of ints, then must be one tuple for every cube/sequence.
            Each element of each tuple gives the axes of each cube/sequence that are aligned.
            Default="All", i.e. all axes are aligned.

        meta: `dict`, optional
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

        self._first_key = keys[0]
        self._cube_types = type(data[0])

        # Enter data into object.
        super().__init__(zip(keys, data))
        self.meta = meta

        n_cubes = len(data)
        # If aligned_axes not set, assume all axes are aligned in order.
        if aligned_axes.lower() == "all":
            # Check all cubes are of same shape
            cube0_dims = data[0].dimensions
            cubes_same_shape = all([all(d.dimensions == cube0_dims) for d in data])
            if cubes_same_shape is not True:
                raise ValueError(
                    "All cubes in data not of same shape. Please set aligned_axes kwarg.")
            self.n_aligned_axes = len(cube0_dims)
            self.aligned_axes = dict([(k, tuple(range(len(cube0_dims)))) for k in keys])
        elif aligned_axes is None:
            self.n_aligned_axes = 0
            self.aligned_axes = None
        else:
            # Else, sanitize user-supplied aligned axes.
            if dont_sanitize_aligned_axes is True:
                self.n_aligned_axes = len(aligned_axes[0])
                self.aligned_axes = dict(zip(keys, aligned_axes))
            else:
                aligned_axes, self.n_aligned_axes = collection_utils._sanitize_aligned_axes(
                        data, aligned_axes, n_cubes)
                self.aligned_axes = dict(zip(keys, aligned_axes))

    def __repr__(self):
        return ("""NDCollection
------------
Cube keys: {keys}
Number of Cubes: {n_cubes}
Cube Types: {cube_types}
Aligned dimensions: {aligned_dims}
Aligned world physical axis types: {aligned_axis_types}""".format(
    keys=self.keys(), n_cubes=len(self), cube_types=self._cube_types,
    aligned_dims=self.aligned_dimensions,
    aligned_axis_types=self.aligned_world_axis_physical_types))

    @property
    def aligned_dimensions(self):
        return self[self._first_key].dimensions[np.array(self.aligned_axes[self._first_key])]

    @property
    def aligned_world_axis_physical_types(self):
        axis_types = np.array(self[self._first_key].world_axis_physical_types)
        return tuple(axis_types[np.array(self.aligned_axes[self._first_key])])

    def __getitem__(self, item):
        # There are two ways to slice:
        # by key or sequence of keys, i.e. slice out given cubes in the collection, or
        # by typical python numeric slicing API,
        # i.e. slice the each component cube along the aligned axes.

        # If item is single string, slicing is simple.
        if isinstance(item, str):
            return super().__getitem__(item)

        # If item is not a single string...
        else:
            # If item is a sequence, ensure strings and numeric items are not mixed.
            item_is_strings = False
            if isinstance(item, collections.abc.Sequence):
                item_strings = [isinstance(item_, str) for item_ in item]
                item_is_strings = all(item_strings)
                # Ensure strings are not mixed with slices.
                if (not item_is_strings) and (not all(np.invert(item_strings))):
                    raise TypeError("Cannot mix keys and non-keys when indexing instance.")

            # If sequence is all strings, extract the cubes corresponding to the string keys.
            if item_is_strings:
                new_data = [self[_item] for _item in item]
                new_keys = item
                new_aligned_axes = tuple([self.aligned_axes[item_] for item_ in item])

            # Else, the item is assumed to be a typical slicing item.
            # Slice each cube in collection using information in this item.
            # However, this can only be done if there are aligned axes.
            else:
                if self.aligned_axes is None:
                    raise IndexError("Cannot slice unless collection has aligned axes.")
                else:
                    # Derive item to be applied to each cube in collection and
                    # whether any aligned axes are dropped by the slicing.
                    collection_items, new_aligned_axes = self._generate_collection_getitems(item)
                    # Apply those slice items to each cube in collection.
                    new_data = [self[key][tuple(cube_item)]
                                for key, cube_item in zip(self, collection_items)]
                    # Since item is not strings, no cube in collection is dropped.
                    # Therefore the collection keys remain unchanged.
                    new_keys = list(self.keys())

            return self.__class__(new_data, keys=new_keys, aligned_axes=new_aligned_axes,
                                  meta=self.meta, dont_sanitize_aligned_axes=True)

    def _generate_collection_getitems(self, item):
        # There are 3 supported cases of the slice item: int, slice, tuple of ints and/or slices.
        # Compile appropriate slice items for each cube in the collection and
        # and drop any aligned axes that are sliced out.

        # First, define empty lists of slice items to be applied to each cube in collection.
        collection_items = [[slice(None)] * len(self[key].dimensions) for key in self]
        # Define empty list to hold aligned axes dropped by the slicing.
        drop_aligned_axes_indices = []

        # Case 1: int
        # First aligned axis is dropped.
        if isinstance(item, int):
            drop_aligned_axes_indices = [0]
            # Insert item to each cube's slice item.
            for i, key in enumerate(self):
                collection_items[i][self.aligned_axes[key][0]] = item

        # Case 2: slice
        elif isinstance(item, slice):
            # Insert item to each cube's slice item.
            for i, key in enumerate(self):
                collection_items[i][self.aligned_axes[key][0]] = item
            # Note that slice interval's of 1 result in an axis of length 1.
            # The axis is not dropped.

        # Case 3: tuple of ints/slices
        # Search sub-items within tuple for ints or 1-interval slices.
        elif isinstance(item, tuple):
            # Ensure item is not longer than number of aligned axes
            if len(item) > self.n_aligned_axes:
                raise IndexError("Too many indices")
            for i, axis_item in enumerate(item):
                if isinstance(axis_item, int):
                    drop_aligned_axes_indices.append(i)
                for j, key in enumerate(self):
                    collection_items[j][self.aligned_axes[key][i]] = axis_item

        else:
            raise TypeError(f"Unsupported slicing type: {axis_item}")

        # Use indices of dropped axes determine above to update aligned_axes
        # by removing any that have been dropped.
        drop_aligned_axes_indices = np.array(drop_aligned_axes_indices)
        new_aligned_axes = collection_utils._update_aligned_axes(
                drop_aligned_axes_indices, self.aligned_axes, self._first_key)

        return collection_items, new_aligned_axes

    def copy(self):
        return copy.deepcopy(self)

    def setdefault(self):
        raise NotImplementedError("NDCollection does not support setdefault.")

    def popitem(self):
        raise NotImplementedError("NDCollection does not support popitem.")

    def pop(self, key):
        """Removes the cube corresponding to the key from the collection and returns it."""
        # Extract desired cube from collection.
        popped_cube = super().pop(key)
        # Delete corresponding aligned axes
        popped_aligned_axes = self.aligned_axes.pop(key)
        # If first key removed, update.
        if key == self._first_key:
            self._first_key = list(self.keys())[0]

        return popped_cube

    def update(self, key, data, aligned_axes):
        """Updates existing cube within collection or adds new cube."""
        # Sanitize aligned axes.
        if isinstance(aligned_axes, int):
            aligned_axes = (aligned_axes,)
        sanitized_axes, n_sanitized_axes = collection_utils._sanitize_aligned_axes(
                [self[self._first_key], data],
                (self.aligned_axes[self._first_key], aligned_axes), 2)
        # Update collection
        super().update({key: data})
        self.aligned_axes.update({key: sanitized_axes[-1]})

    def __delitem__(self, key):
        super().__delitem__(key)
        self.aligned_axes.__delitem__(key)
        if key == self._first_key:
            self._first_key = list(self.keys())[0]
