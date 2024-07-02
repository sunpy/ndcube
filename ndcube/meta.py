import copy
import numbers
import collections.abc

import numpy as np

__all__ = ["NDMeta"]


class NDMeta(dict):
    """
    A sliceable object for storing metadata.

    Metadata can be linked to a data axis which causes it to be sliced when the
    standard Python numeric slicing API is applied to the object.
    Specific pieces of metadata can be obtain using the dict-like string slicing API.
    Metadata associated with an axis/axes must have the same length/shape as those axes.

    Parameters
    ----------
    meta: dict-like
        The names and values of metadata.

    comments: dict-like, optional
        Comments associated with any of the above pieces of metadata.

    axes: dict-like, optional
        The axis/axes associated with the metadata denoted by the keys.
        Metadata not included are considered not to be associated with any axis.
        Each axis value must be an iterable of `int`. An `int` itself is also
        acceptable if the metadata is associated with a single axis.
        The value of axis-assigned metadata in meta must be same length as
        number of associated axes (axis-aligned), or same shape as the associated
        data array's axes (grid-aligned).

    data_shape: iterator of `int`, optional
        The shape of the data with which this metadata is associated.
        Must be set if axes input is set.

    Notes
    -----
    **Axis-aware Metadata**
    There are two valid types of axis-aware metadata: axis-aligned and grid-aligned.
    Axis-aligned metadata gives one value per associated axis, while grid-aligned
    metadata gives a value for each data array element in the associated axes.
    Consequently, axis-aligned metadata has the same length as the number of
    associated axes, while grid-aligned metadata has the same shape as the associated
    axes. To avoid confusion, axis-aligned metadata that is only associated with one
    axis must be scalar or a string. Length-1 objects (excluding strings) are assumed
    to be grid-aligned and associated with a length-1 axis.

    **Slicing and Rebinning Axis-aware Metadata**
    Axis-aligned metadata is only considered valid if the associated axes are present.
    Therefore, axis-aligned metadata is only changed if an associated axis is dropped
    by an operation, e.g. slicing. In such a case, the value associated with the
    dropped axes is also dropped and hence lost.  If the axis of a 1-axis-aligned
    metadata value (scalar) is slicing away, the metadata key is entirely removed
    from the NDMeta object.

    Grid-aligned metadata is mirrors the data array, it is sliced following
    the same rules with one exception. If an axis is dropped by slicing, the metadata
    name is kept, but its value is set to the value at the row/column where the
    axis/axes was sliced away, and the metadata axis-awareness is removed. This is
    similar to how coordinate values are transferred to ``global_coords`` when their
    associated axes are sliced away.

    Note that because rebinning does not drop axes, axis-aligned metadata is unaltered
    by rebinning. By contrast, grid-aligned metadata must necessarily by affected by
    rebinning. However, how it is affected depends on the nature of the metadata and
    there is no generalized solution. Therefore, this class does not alter the shape
    or values of grid-aligned metadata during rebinning, but simply removes its
    axis-awareness.  If specific pieces of metadata have a known way to behave during
    rebinning, this can be handled by subclasses or mixins.
    """
    def __init__(self, meta=None, comments=None, axes=None, data_shape=None):
        self.__ndcube_can_slice__ = True
        self.__ndcube_can_rebin__ = True
        self.original_meta = meta

        if meta is None:
            meta = {}
        else:
            meta = dict(meta)
        super().__init__(meta.items())
        meta_keys = meta.keys()

        if comments is None:
            self._comments = dict()
        else:
            comments = dict(comments)
            if not set(comments.keys()).issubset(set(meta_keys)):
                raise ValueError(
                    "All comments must correspond to a value in meta under the same key.")
            self._comments = comments

        if data_shape is None:
            self._data_shape = data_shape
        else:
            self._data_shape = np.asarray(data_shape, dtype=int)

        if axes is None:
            self._axes = dict()
        else:
            if not (isinstance(data_shape, collections.abc.Iterable) and
                    all([isinstance(i, numbers.Integral) for i in data_shape])):
                raise TypeError("If axes is set, data_shape must be an iterable giving "
                                "the length of each axis of the associated cube.")
            axes = dict(axes)
            if not set(axes.keys()).issubset(set(meta_keys)):
                raise ValueError(
                    "All axes must correspond to a value in meta under the same key.")
            self._axes = dict([(key, self._sanitize_axis_value(axis, meta[key], key))
                               for key, axis in axes.items()])

    def _sanitize_axis_value(self, axis, value, key):
        axis_err_msg = ("Values in axes must be an integer or iterable of integers giving "
                        f"the data axis/axes associated with the metadata.  axis = {axis}.")
        if isinstance(axis, numbers.Integral):
            axis = (axis,)
        if len(axis) == 0:
            return ValueError(axis_err_msg)
        if self.shape is None:
            raise TypeError("NDMeta instance does not have a shape so new metadata "
                            "cannot be assigned to an axis.")
        # Verify each entry in axes is an iterable of ints or a scalar.
        if not (isinstance(axis, collections.abc.Iterable) and all([isinstance(i, numbers.Integral)
                                                                    for i in axis])):
            return ValueError(axis_err_msg)
        axis = np.asarray(axis)
        if _not_scalar(value):
            axis_shape = tuple(self.shape[axis])
            if not _is_grid_aligned(value, axis_shape) and not _is_axis_aligned(value, axis_shape):
                raise ValueError(
                    f"{key} must have shape {tuple(self.shape[axis])} "
                    f"as its associated axes {axis}, ",
                    f"or same length as number of associated axes ({len(axis)}). "
                    f"Has shape {value.shape if hasattr(value, 'shape') else len(value)}")
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

    def add(self, name, value, comment=None, axis=None, overwrite=False):
        """
        Add a new piece of metadata to instance.

        Parameters
        ----------
        name: `str`
            The name/label of the metadata.

        value: Any
            The value of the metadata. If axes input is not None, this must have the
            same length/shape as those axes as defined by ``self.shape``.

        comment: `str` or `None`
            Any comment associated with this metadata. Set to None if no comment desired.

        axis: `int`, iterable of `int`, or `None`
            The axis/axes with which the metadata is linked. If not associated with any
            axis, set this to None.

        overwrite: `bool`, optional
            If True, overwrites the entry of the name name if already present.
        """
        if name in self.keys() and overwrite is not True:
            raise KeyError(f"'{name}' already exists. "
                           "To update an existing metadata entry set overwrite=True.")
        if comment is not None:
            self._comments[name] = comment
        if axis is not None:
            axis = self._sanitize_axis_value(axis, value, name)
            self._axes[name] = axis
        elif name in self._axes:
            del self._axes[name]
        # This must be done after updating self._axes otherwise it may error.
        self.__setitem__(name, value)

    def remove(self, name):
        if name in self._comments:
            del self._comments[name]
        if name in self._axes:
            del self._axes[name]
        del self[name]

    def __setitem__(self, key, val):
        axis = self.axes.get(key, None)
        if axis is not None:
            if _not_scalar(val):
                axis_shape = tuple(self.shape[axis])
                if not _is_grid_aligned(val, axis_shape) and not _is_axis_aligned(val, axis_shape):
                    raise TypeError(
                        f"{key} is already associated with axis/axes {axis}. val must therefore "
                        f"must either have same length as number associated axes ({len(axis)}), "
                        f"or the same shape as associated data axes {tuple(self.shape[axis])}. "
                        f"val shape = {val.shape if hasattr(val, 'shape') else (len(val),)}\n"
                        "We recommend using the 'add' method to set values.")
        super().__setitem__(key, val)

    def __getitem__(self, item):
        # There are two ways to slice:
        # by key, or
        # by typical python numeric slicing API,
        # i.e. slice the each piece of metadata associated with an axes.

        if isinstance(item, str):
            return super().__getitem__(item)

        elif self.shape is None:
            raise TypeError("NDMeta object does not have a shape and so cannot be sliced.")

        else:
            new_meta = copy.deepcopy(self)
            if isinstance(item, (numbers.Integral, slice)):
                item = [item]
            naxes = len(self.shape)
            item = np.array(list(item) + [slice(None)] * (naxes - len(item)),
                            dtype=object)

            # Edit data shape and calculate which axis will be dropped.
            dropped_axes = np.zeros(naxes, dtype=bool)
            new_shape = new_meta.shape
            for i, axis_item in enumerate(item):
                if isinstance(axis_item, numbers.Integral):
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
                    if stop < 0:
                        stop = self.shape[i] - stop
                    new_shape[i] = stop - start
                else:
                    raise TypeError("Unrecognized slice type. "
                                    "Must be an int, slice and tuple of the same.")
            kept_axes = np.invert(dropped_axes)
            new_meta._data_shape = new_shape[kept_axes]

            # Slice all metadata associated with axes.
            for key, value in self.items():
                axis = self.axes.get(key, None)
                drop_key = False
                if axis is not None:
                    # Calculate new axis indices.
                    new_axis = np.asarray(list(
                        set(axis).intersection(set(np.arange(naxes)[kept_axes]))
                        ))
                    if len(new_axis) == 0:
                        new_axis = None
                    else:
                        cumul_dropped_axes = np.cumsum(dropped_axes)[new_axis]
                        new_axis -= cumul_dropped_axes

                    # Calculate sliced metadata values.
                    axis_shape = tuple(self.shape[axis])
                    if _is_scalar(value):
                        new_value = value
                        # If scalar metadata's axes have been dropped, mark metadata to be dropped.
                        if new_axis is None:
                            drop_key = True
                    else:
                        value_is_axis_aligned = _is_axis_aligned(value, axis_shape)
                        if value_is_axis_aligned:
                            new_item = kept_axes[axis]
                        else:
                            new_item = tuple(item[axis])
                        # Slice metadata value.
                        try:
                            new_value = value[new_item]
                        except:
                            # If value cannot be sliced by fancy slicing, convert it
                            # it to an array, slice it, and then if necessary, convert
                            # it back to its original type.
                            new_value = (np.asanyarray(value)[new_item])
                            if hasattr(new_value, "__len__"):
                                new_value = type(value)(new_value)
                        # If axis-aligned metadata sliced down to length 1, convert to scalar.
                        if value_is_axis_aligned and len(new_value) == 1:
                            new_value = new_value[0]
                    # Overwrite metadata value with newly sliced version.
                    if drop_key:
                        new_meta.remove(key)
                    else:
                        new_meta.add(key, new_value, self.comments.get(key, None), new_axis,
                                     overwrite=True)

            return new_meta

    def rebin(self, rebinned_axes, new_shape):
        """
        Adjusts axis-aware metadata to stay consistent with a rebinned `~ndcube.NDCube`.

        This is done by simply removing the axis-awareness of metadata associated with
        rebinned axes. The metadata itself is not changed or removed. This operation
        does not remove axis-awareness from metadata only associated with non-rebinned
        axes, i.e. axes whose corresponding entries in ``bin_shape`` are 1.

        Parameters
        ----------
        rebinned_axes: `set` of `int`
            Set of array indices of axes that are rebinned.
        new_shape: `tuple` of `int`
            The new shape of the rebinned data.
        """
        # Sanitize input.
        data_shape = self.shape
        if not isinstance(rebinned_axes, set):
            raise TypeError(
                f"rebinned_axes must be a set. type of rebinned_axes is {type(rebinned_axes)}")
        if not all([isinstance(dim, numbers.Integral) for dim in rebinned_axes]):
            raise ValueError("All elements of rebinned_axes must be ints.")
        list_axes = list(rebinned_axes)
        if min(list_axes) < 0 or max(list_axes) >= len(data_shape):
            raise ValueError(
                f"Elements in rebinned_axes must be in range 0--{len(data_shape)-1} inclusive.")
        if len(new_shape) != len(data_shape):
            raise ValueError(f"new_shape must be a tuple of same length as data shape: "
                             f"{len(new_shape)} != {len(self.shape)}")
        if not all([isinstance(dim, numbers.Integral) for dim in new_shape]):
            raise TypeError("bin_shape must contain only integer types.")
        # Remove axis-awareness from grid-aligned metadata associated with rebinned axes.
        new_meta = copy.deepcopy(self)
        null_set = set()
        for name, axes in self.axes.items():
            if (_is_grid_aligned(self[name], tuple(self.shape[axes]))
                and set(axes).intersection(rebinned_axes) != null_set):
                del new_meta._axes[name]
        # Update data shape.
        new_meta._data_shape = np.asarray(new_shape).astype(int)
        return new_meta


def _not_scalar(value):
    return (
        (
         hasattr(value, "shape")
         or hasattr(value, "__len__")
        )
        and not
        (
         isinstance(value, str)
        ))


def _is_scalar(value):
    return not _not_scalar(value)


def _is_grid_aligned(value, axis_shape):
    if _is_scalar(value):
        return False
    value_shape = value.shape if hasattr(value, "shape") else (len(value),)
    if value_shape != axis_shape:
        return False
    return True


def _is_axis_aligned(value, axis_shape):
    len_value = len(value) if _not_scalar(value) else 1
    return not _is_grid_aligned(value, axis_shape) and len_value == len(axis_shape)
