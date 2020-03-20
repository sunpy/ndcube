import collections.abc
import copy
import textwrap

import numpy as np

from ndcube import NDCube, NDCubeSequence
from ndcube.utils.cube import convert_extra_coords_dict_to_input_format
import ndcube.utils.collection as collection_utils

__all__ = ["NDCollection"]


class NDCollection(dict):
    def __init__(self, key_data_pairs, aligned_axes=None, meta=None, **kwargs):
        """
        A class for holding and manipulating a collection of aligned NDCube or NDCubeSequences.

        Parameters
        ----------
        data: sequence of `tuple`s of (`str`, `~ndcube.NDCube` or `~ndcube.NDCubeSequence`)
            The names and data cubes/sequences to held in the collection.

        aligned_axes: `tuple` of `int`, `tuple` of `tuple`s of `int`, 'all', or None, optional
            Axes of each cube/sequence that are aligned in numpy order.
            If elements are int, then the same axis numbers in all cubes/sequences are aligned.
            If elements are tuples of ints, then must be one tuple for every cube/sequence.
            Each element of each tuple gives the axes of each cube/sequence that are aligned.
            If 'all', all axes are aligned in natural order, i.e. the 0th axes of all cubes
            are aligned, as are the 1st, and so on.
            Default=None

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
        # Unzip key_data_pairs
        keys, data = zip(*(key_data_pairs))

        # Sanitize inputs unless hidden kwarg indicates not to.
        sanitize_inputs = kwargs.get("sanitize_inputs", True)
        if sanitize_inputs is True:
            aligned_axes = _sanitize_aligned_axes(data, aligned_axes)

        # Enter data into object.
        super().__init__(key_data_pairs)
        self.meta = meta

        # Attach aligned axes to object
        if aligned_axes is None:
            self.n_aligned_axes = 0
            self.aligned_axes = aligned_axes
        else:
            self.n_aligned_axes = len(aligned_axes[0])
            self.aligned_axes = dict(zip(keys, aligned_axes))

    @property
    def _first_key(self):
        return list(self.keys())[0]

    def __repr__(self):
        return (textwrap.dedent("""
            NDCollection
            ------------
            Cube keys: {keys}
            Number of Cubes: {n_cubes}
            Aligned dimensions: {aligned_dims}
            Aligned world physical axis types: {aligned_axis_types}""".format(
                keys=self.keys(), n_cubes=len(self),
                aligned_dims=self.aligned_dimensions,
                aligned_axis_types=self.aligned_world_axis_physical_types)))

    @property
    def aligned_dimensions(self):
        if self.aligned_axes is None:
            return None
        else:
            return self[self._first_key].dimensions[np.array(self.aligned_axes[self._first_key])]

    @property
    def aligned_world_axis_physical_types(self):
        if self.aligned_axes is None:
            return None
        else:
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

            return self.__class__(list(zip(new_keys, new_data)), aligned_axes=new_aligned_axes,
                                  meta=self.meta, sanitize_inputs=False)

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
        return popped_cube

    def add_to_collection(self, key_data_pair, aligned_axes):
        """Updates existing cube within collection or adds new cube."""
        key, data = key_data_pair
        # Sanitize aligned axes.
        if isinstance(aligned_axes, str) and aligned_axes.lower() == "all":
            aligned_axes = tuple(range(len(data.dimensions)))
        elif isinstance(aligned_axes, int):
            aligned_axes = (aligned_axes,)
        if self.aligned_axes is None and aligned_axes is None:
            sanitize_axes = aligned_axes
        else:
            sanitized_axes = collection_utils._sanitize_user_aligned_axes(
                [self[self._first_key], data], (self.aligned_axes[self._first_key], aligned_axes))
        # Update collection
        super().update({key: data})
        self.aligned_axes.update({key: sanitized_axes[-1]})

    def update(self, collection):
        """Merges a new collection replacing cubes with common keys."""
        if not isinstance(collection, NDCollection):
            raise TypeError(f"collection must be an NDCollection. Type is {type(collection)}")
        for key in collection.keys():
            # Check aligned axes are compatible.
            collection_utils.assert_aligned_axes_compatible(
                    self[self._first_key].dimensions, collection[key].dimensions,
                    self.aligned_axes[self._first_key], collection.aligned_axes[key])
            # If key is common between collections delete original version.
            if key in self.keys():
                del self[key]
            # Add new data cube to collection.
            self.add_to_collection((key, collection[key]), collection.aligned_axes[key])

    def __delitem__(self, key):
        super().__delitem__(key)
        self.aligned_axes.__delitem__(key)


def _sanitize_aligned_axes(data, aligned_axes):
    # If aligned_axes set to "all", assume all axes are aligned in order.
    if isinstance(aligned_axes, str) and aligned_axes.lower() == "all":
        # Check all cubes are of same shape
        cube0_dims = data[0].dimensions
        cubes_same_shape = all([all([d.dimensions[i] == dim for i, dim in enumerate(cube0_dims)])
                                for d in data])
        if cubes_same_shape is not True:
            raise ValueError(
                "All cubes in data not of same shape. Please set aligned_axes kwarg.")
        sanitized_axes = tuple([tuple(range(len(cube0_dims)))] * len(data))
    elif aligned_axes is None:
        sanitized_axes = None
    else:
        # Else, sanitize user-supplied aligned axes.
        sanitized_axes = collection_utils._sanitize_user_aligned_axes(data, aligned_axes)

    return sanitized_axes
