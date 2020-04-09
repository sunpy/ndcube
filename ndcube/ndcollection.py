import collections.abc
import copy
import textwrap

import numpy as np

import ndcube.utils.collection as collection_utils
from ndcube.ndcube import NDCube
from ndcube.ndcube_sequence import NDCubeSequence
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
        ``aligned_axes = (1, 2)``
        means that axis 1 (0-based counting) of cube0 is aligned with axis 1 of cube1,
        and axis 2 of cube0 is aligned with axis 2 of cube1.
        However, if
        ``aligned_axes = ((0, 1), (2, 1))``
        then the first tuple corresponds to cube0 and the second with cube1.
        This is interpretted as axis 0 of cube0 is aligned with axis 2 of cube1 while
        axis 1 of cube0 is aligned with axis 1 of cube1.

        """
        # Enter data and metadata into object.
        super().__init__(key_data_pairs)
        self.meta = meta

        # Convert aligned axes to required format.
        sanitize_inputs = kwargs.pop("sanitize_inputs", True)
        if aligned_axes is not None:
            keys, data = zip(*key_data_pairs)
            # Sanitize aligned axes unless hidden kwarg indicates not to.
            if sanitize_inputs:
                aligned_axes = collection_utils._sanitize_aligned_axes(keys, data, aligned_axes)
            else:
                aligned_axes = dict(zip(keys, aligned_axes))
        if kwargs:
            raise TypeError(
                    f"__init__() got an unexpected keyword argument: '{list(kwargs.keys())[0]}'")
        # Attach aligned axes to object.
        self.aligned_axes = aligned_axes
        if self.aligned_axes is None:
            self.n_aligned_axes = 0
        else:
            self.n_aligned_axes = len(self.aligned_axes[keys[0]])

    @property
    def _first_key(self):
        return list(self.keys())[0]

    def __str__(self):
        return (textwrap.dedent(f"""\
            NDCollection
            ------------
            Cube keys: {tuple(self.keys())}
            Number of Cubes: {len(self)}
            Aligned dimensions: {self.aligned_dimensions}
            Aligned world physical axis types: {self.aligned_world_axis_physical_types}"""))

    def __repr__(self):
        return f"{object.__repr__(self)}\n{str(self)}"

    @property
    def aligned_dimensions(self):
        """
        Returns the lengths of all aligned axes.

        If there are no aligned axes, returns None.

        """
        if self.aligned_axes is not None:
            return self[self._first_key].dimensions[np.array(self.aligned_axes[self._first_key])]

    @property
    def aligned_world_axis_physical_types(self):
        """
        Returns the physical types of the aligned axes of an ND object in the collection.

        If there are no aligned axes, returns None.

        """
        if self.aligned_axes is not None:
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
        return self.__class__(self.items(), tuple(self.aligned_axes.values()),
                              meta=self.meta, sanitize_inputs=False)

    def setdefault(self):
        raise NotImplementedError("NDCollection does not support setdefault.")

    def popitem(self):
        raise NotImplementedError("NDCollection does not support popitem.")

    def pop(self, key):
        # Extract desired cube from collection.
        popped_cube = super().pop(key)
        # Delete corresponding aligned axes
        popped_aligned_axes = self.aligned_axes.pop(key)
        return popped_cube

    def update(self, *args):
        """
        Merges a new collection with current one replacing objects with common keys.

        Takes either a single input (`NDCollection`) or two inputs
        (sequence of key/value pair and aligned axes associated with each key/value pair.

        """
        # If two inputs, inputs must be key_data_pairs and aligned_axes.
        if len(args) == 2:
            key_data_pairs = args[0]
            new_keys, new_data = zip(*key_data_pairs)
            new_aligned_axes = collection_utils._sanitize_aligned_axes(new_keys, new_data, args[1])
        else:  # If one arg given, input must be NDCollection.
            collection = args[0]
            new_keys = list(collection.keys())
            new_data = list(collection.values())
            key_data_pairs = zip(new_keys, new_data)
            new_aligned_axes = collection.aligned_axes
        # Check aligned axes of new inputs are compatible with those in self.
        # As they've already been sanitized, only one set of aligned axes need be checked.
        collection_utils.assert_aligned_axes_compatible(
                self[self._first_key].dimensions, new_data[0].dimensions,
                self.aligned_axes[self._first_key], new_aligned_axes[new_keys[0]])
        # Update collection
        super().update(key_data_pairs)
        self.aligned_axes.update(new_aligned_axes)

    def __delitem__(self, key):
        super().__delitem__(key)
        self.aligned_axes.__delitem__(key)

    def __setitem__(self, key, value):
        raise NotImplementedError("NDCollection does not support __setitem__. "
                                  "Use NDCollection.update instead")
