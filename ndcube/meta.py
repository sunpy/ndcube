import collections.abc
import copy
import numbers

import numpy as np

__all__ = ["Meta"]


class Meta(dict):
    def __init__(self, header=None, comments=None, axes=None, data_shape=None):
        self.original_header = header

        # Sanitize metadata values and instantiate class.
        if header is None:
            header = {}
        else:
            header = dict(header)
        super().__init__(header.items())
        header_keys = header.keys()
        
        # Generate dictionary for comments.
        if comments is None:
            self._comments = dict(zip(header.keys(), [None] * len(header_keys)))
        else:
            comments = dict(comments)
            self._comments = dict([(key, comments.get(key)) for key in header])

        # Generate dictionary for axes.
        if axes is None:
            self._axes = dict(zip(header.keys(), [None] * len(header_keys)))
            self._data_shape = None
        else:
            # Verify data_shape is set if axes is set.
            if not (isinstance(data_shape, collections.abc.Iterable) and
                    all([isinstance(i, numbers.Integral) for i in data_shape])):
                raise TypeError("If axes is set, data_shape must be an iterable giving "
                                "the length of each axis of the assocated cube.")
            self._data_shape = np.asarray(data_shape)
            axes = dict(axes)
            self._axes = dict([(key, self._sanitize_axis_value(axes.get(key), header[key], key))
                               for key in header_keys])

    def _sanitize_axis_value(self, axis, value, key):
        if axis is None:
            return None
        if self.shape is None:
            raise TypeError("Meta instance does not have a shape so new metadata "
                            "cannot be assigned to an axis.")
        # Verify each entry in axes is an iterable of ints.
        if isinstance(axis, numbers.Integral):
            axis = (axis,)
        if not (isinstance(axis, collections.abc.Iterable) and
                all([isinstance(i, numbers.Integral) for i in axis])):
            raise TypeError("Values in axes must be an int or tuple of ints giving "
                            "the data axis/axes associated with the metadata.")
        axis = np.asarray(axis)

        # Confirm each axis-associated piece of metadata has the same shape
        # as its associated axes.
        shape_error_msg = (f"{key} must have shape {tuple(self.shape[axis])} "
                           f"as it is associated with axes {axis}")
        if len(axis) == 1:
            if not hasattr(value, "__len__"):
                raise TypeError(shape_error_msg)
            meta_shape = (len(value),)
        else:
            if not hasattr(value, "shape"):
                raise TypeError(shape_error_msg)
            meta_shape = value.shape
        if not all(meta_shape == self.shape[axis]):
            raise ValueError(shape_error_msg)

        return axis

    @property
    def comments(self):
        return self._comments

    @property
    def axes(self):
        return self._axes

    @property
    def shape(self):
        return self._data_shape

    def add(self, name, value, comment, axis, overwrite=False):
        """Need docstring!"""
        if name in self.keys() and overwrite is not True:
            raise KeyError(f"'{name}' already exists. "
                           "To update an existing metadata entry set overwrite=True.")
        if axis is not None:
            axis = self._sanitize_axis_value(axis, value, name)
        self._comments[name] = comment
        self._axes[name] = axis
        self.__setitem__(name, value)  # This must be done after updating self._axes otherwise it may error.

    def __del__(self, name):
        del self._comments[name]
        del self._axes[name]
        del self[name]

    def __setitem__(self, key, val):
        axis = self.axes[key]
        if axis is not None:
            recommendation = "We recommend using the 'add' method to set values."
            if len(axis) == 1:
                if not (hasattr(val, "__len__") and len(val) == self.shape[axis[0]]):
                    raise TypeError(f"{key} must have same length as associated axis, "
                                    f"i.e. axis {axis[0]}: {self.shape[axis[0]]}\n"
                                    f"{recommendation}")
            else:
                if not (hasattr(val, "shape") and all(val.shape == self.shape[axis])):
                    raise TypeError(f"{key} must have same shape as associated axes, "
                                    f"i.e axes {axis}: {self.shape[axis]}\n"
                                    f"{recommendation}")
        super().__setitem__(key, val)

    def __getitem__(self, item):
        # There are two ways to slice:
        # by key, or
        # by typical python numeric slicing API,
        # i.e. slice the each piece of metadata associated with an axes.

        # If item is single string, slicing is simple.
        if isinstance(item, str):
            return super().__getitem__(item)

        # Else, the item is assumed to be a typical slicing item.
        elif self.shape is None:
            raise TypeError("Meta object does not have a shape and so cannot be sliced.")

        else:
            new_meta = copy.deepcopy(self)
            # Convert item to array of ints and slices for consistent behaviour.
            if isinstance(item, (numbers.Integral, slice)):
                item = [item]
            item = np.array(list(item) + [slice(None)] * (len(self.shape) - len(item)),
                            dtype=object)

            # Edit data shape and calculate which axis will be dropped.
            dropped_axes = np.zeros(len(self.shape), dtype=bool)
            new_shape = new_meta.shape
            j = 0
            for i, axis_item in enumerate(item):
                if isinstance(axis_item, numbers.Integral):
                    new_shape = np.delete(new_shape, i)
                    dropped_axes[i] = True
                elif isinstance(axis_item, slice):
                    start = axis_item.start
                    if start is None:
                        start = 0
                    if start < 0:
                        start = self.shape[i] - start
                    stop = axis_item.stop
                    if stop is None:
                        stop = self.shape[i]
                        # Mustn't use new_shape here as indexing will be misaligned
                        # if an axis was deleted above.
                    if stop < 0:
                        stop = self.shape[i] - stop
                    new_shape[i - dropped_axes[:i].sum()] = stop - start
                else:
                    raise TypeError("Unrecognized slice type. "
                                    "Must be an int, slice and tuple of the same.")
            new_meta._data_shape = new_shape

            # Calculate the cumulative number of dropped axes.
            cumul_dropped_axes = np.cumsum(dropped_axes)

            # Slice all metadata associated with axes.
            for (key, value), axis in zip(self.items(), self.axes.values()):
                if axis is not None:
                    new_item = tuple(item[axis])
                    if len(new_item) == 1:
                        new_value = value[new_item[0]]
                    else:
                        new_value = value[new_item]
                    new_axis = np.array([-1 if isinstance(i, numbers.Integral) else a
                                         for i, a in zip(new_item, axis)])
                    new_axis -= cumul_dropped_axes[axis]
                    new_axis = new_axis[new_axis >= 0]
                    if len(new_axis) == 0:
                        new_axis = None
                    new_meta.add(key, new_value, self.comments[key], new_axis, overwrite=True)

            return new_meta
