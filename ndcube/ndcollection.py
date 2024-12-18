import copy
import numbers
import textwrap
import collections.abc

import numpy as np

import ndcube.utils.collection as collection_utils
from ndcube.utils.exceptions import warn_deprecated

__all__ = ["NDCollection"]


class NDCollection(dict):
    """
    Class for holding and manipulating an unordered collection of NDCubes or NDCubeSequences.

    Parameters
    ----------
    key_data_pairs: sequence of `tuple` of (`str`, `~ndcube.NDCube` or `~ndcube.NDCubeSequence`)
        The names and data cubes/sequences to held in the collection.

    aligned_axes: `tuple` of `int`, `tuple` of `tuple` of `int`, 'all', or None, optional
        Axes of each cube/sequence that are aligned in numpy order.
        If elements are int, then the same axis numbers in all cubes/sequences are aligned.
        If elements are tuples of ints, then there must be one tuple for every cube/sequence.
        Each element of each tuple gives the axes of each cube/sequence that are aligned.
        If 'all', all axes are aligned in natural order, i.e. the 0th axes of all cubes
        are aligned, as are the 1st, and so on.
        Default is `None`.

    meta: `dict`, optional
        General metadata for the overall collection.

    Example
    -------
    Say the collection holds two NDCubes, each of 3 dimensions.

    >>> aligned_axes = (1, 2)  # doctest:  +SKIP

    means that axis 1 (0-based counting) of cube0 is aligned with axis 1 of cube1,
    and axis 2 of cube0 is aligned with axis 2 of cube1.
    However, if

    >>> aligned_axes = ((0, 1), (2, 1))  # doctest: +SKIP

    then the first tuple corresponds to cube0 and the second with cube1.
    This is interpreted as axis 0 of cube0 is aligned with axis 2 of cube1 while
    axis 1 of cube0 is aligned with axis 1 of cube1.
    """

    def __init__(self, key_data_pairs, aligned_axes=None, meta=None, **kwargs):
        for key, _ in key_data_pairs:
            if isinstance(key, numbers.Number):
                warn_deprecated(
                    "Passing numerical keys to NDCollection is deprecated as they"
                    " lead to ambiguity when slicing the collection."
                )
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
                f"__init__() got an unexpected keyword argument: '{list(kwargs.keys())[0]}'"
            )
        # Attach aligned axes to object.
        self._aligned_axes = aligned_axes
        if self._aligned_axes is None:
            self.n_aligned_axes = 0
        else:
            self.n_aligned_axes = len(self.aligned_axes[keys[0]])

    @property
    def aligned_axes(self):
        """
        The axes of each array that are aligned in numpy order.
        """
        return self._aligned_axes

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
            Aligned physical types: {self.aligned_axis_physical_types}"""))

    def __repr__(self):
        return f"{object.__repr__(self)}\n{self!s}"

    @property
    def aligned_dimensions(self):
        """
        The lengths of all aligned axes.

        If there are no aligned axes, returns None.
        """
        if self.aligned_axes is not None:
            return np.asanyarray(self[self._first_key].shape, dtype=object)[
                np.array(self.aligned_axes[self._first_key])
            ]
        return None

    @property
    def aligned_axis_physical_types(self):
        """
        The physical types common to all members that are associated with each aligned axis.

        One tuple is returned for each axis as there can be more than one physical type
        associated with an aligned axis. If there are no physical types associated
        with an aligned that is common to all collection members, an empty tuple is
        returned for that axis.
        """
        if self.aligned_axes is None:
            return None
        # Get array axis physical types for each aligned axis for all members of collection.
        collection_types = [np.array(cube.array_axis_physical_types,
                                     dtype=object)[np.array(self.aligned_axes[name])]
                            for name, cube in self.items()]
        # Return physical types common to all members of collection for each axis.
        return [tuple(set.intersection(*[set(cube_types[i]) for cube_types in collection_types]))
                for i in range(self.n_aligned_axes)]

    def __getitem__(self, item):
        # There are two ways to slice:
        # by key or sequence of keys, i.e. slice out given cubes in the collection, or
        # by typical python numeric slicing API,
        # i.e. slice the each component cube along the aligned axes.

        # If item is single string, slicing is simple.
        if isinstance(item, str):
            return super().__getitem__(item)

        # If item is not a single string...
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
            new_meta = copy.deepcopy(self.meta)

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
            # Slice meta if sliceable
            if hasattr(self.meta, "__ndcube_can_slice__") and self.meta.__ndcube_can_slice__:
                # Convert negative indices to positive indices as they are not supported by NDMeta.slice
                sanitized_item = copy.deepcopy(item)
                aligned_shape = self.aligned_dimensions
                if isinstance(item, numbers.Integral):
                    if item < 0:
                        sanitized_item = int(self.aligned_dimensions[0] + item)
                elif isinstance(item, slice):
                    if (item.start is not None and item.start < 0) or (item.stop is not None and item.stop < 0):
                        new_start = aligned_shape[0] + item.start if item.start < 0 else item.start
                        new_stop = aligned_shape[0] + item.stop if item.stop < 0 else item.stop
                        sanitized_item = slice(new_start, new_stop)
                else:
                    sanitized_item = list(sanitized_item)
                    for i, ax_it in enumerate(item):
                        if isinstance(ax_it, numbers.Integral) and ax_it < 0:
                            sanitized_item[i] = aligned_shape[i] + ax_it
                        elif isinstance(ax_it, slice):
                            if (ax_it.start is not None and ax_it.start < 0) or (ax_it.stop is not None and ax_it.stop < 0):
                                new_start = aligned_shape[i] + ax_it.start if ax_it.start < 0 else ax_it.start
                                new_stop = aligned_shape[i] + ax_it.stop if ax_it.stop < 0 else ax_it.stop
                                sanitized_item[i] = slice(new_start, new_stop)
                    sanitized_item = tuple(sanitized_item)
                # Use sanitized item to slice meta.
                new_meta = self.meta.slice[sanitized_item]
            else:
                new_meta = copy.deepcopy(self.meta)

        return self.__class__(
            list(zip(new_keys, new_data)),
            aligned_axes=new_aligned_axes,
            meta=new_meta,
            sanitize_inputs=False
        )

    def _generate_collection_getitems(self, item):
        # There are 3 supported cases of the slice item: int, slice, tuple of ints and/or slices.
        # Compile appropriate slice items for each cube in the collection and
        # and drop any aligned axes that are sliced out.

        # First, define empty lists of slice items to be applied to each cube in collection.
        collection_items = [[slice(None)] * len(self[key].shape) for key in self]
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
        # Aligned axes is not a required parameter and may be None
        aligned_axes = None if self.aligned_axes is None else tuple(self.aligned_axes.values())
        return self.__class__(self.items(), aligned_axes, meta=self.meta, sanitize_inputs=False)

    def setdefault(self):
        """Not supported by `~ndcube.NDCollection`"""
        raise NotImplementedError("NDCollection does not support setdefault.")

    def popitem(self):
        """Not supported by `~ndcube.NDCollection`"""
        raise NotImplementedError("NDCollection does not support popitem.")

    def pop(self, key):
        """
        Remove a member from the `~ndcube.NDCollection` and return it.

        Parameters
        ----------
        key: `str`
            The name of the member to remove and return.
        """
        # Extract desired cube from collection.
        popped_cube = super().pop(key)
        # Aligned axes is not a required parameter and may be None
        if self.aligned_axes is not None:
            # Delete corresponding aligned axes
            self.aligned_axes.pop(key)
        return popped_cube

    def update(self, *args):
        """
        Merges a new collection with current one replacing objects with common keys.

        Takes either a single input (`~ndcube.NDCollection`) or two inputs
        (sequence of key/value pairs and aligned axes associated with each key/value pair.
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
        first_old_aligned_axes = self.aligned_axes[self._first_key] if self.aligned_axes is not None else None
        first_new_aligned_axes = new_aligned_axes[new_keys[0]] if new_aligned_axes is not None else None
        collection_utils.assert_aligned_axes_compatible(
            self[self._first_key].shape, new_data[0].shape,
            first_old_aligned_axes, first_new_aligned_axes
        )
        # Update collection
        super().update(key_data_pairs)
        if first_old_aligned_axes is not None:  # since the above assertion passed, if one aligned axes is not None, both are not None
            self.aligned_axes.update(new_aligned_axes)

    def __delitem__(self, key):
        super().__delitem__(key)
        self.aligned_axes.__delitem__(key)

    def __setitem__(self, key, value):
        raise NotImplementedError("NDCollection does not support __setitem__. "
                                  "Use NDCollection.update instead")
