import abc
import copy
import numbers
import collections.abc
from types import MappingProxyType

import numpy as np

__all__ = ["NDMeta", "NDMetaABC"]


class NDMetaABC(collections.abc.Mapping):
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

    key_comments: dict-like, optional
        Comments associated with any of the above pieces of metadata.

    axes: dict-like, optional
        The axis/axes associated with the metadata denoted by the keys.
        Metadata not included are considered not to be associated with any axis.
        Each axis value must be an iterable of `int`. An `int` itself is also
        acceptable if the metadata is associated with a single axis.
        The value of axis-assigned metadata in meta must be same length as
        number of associated axes (axis-aligned), or same shape as the associated
        data array's axes (grid-aligned).

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

    __ndcube_can_slice__: bool
    __ndcube_can_rebin__: bool

    @property
    @abc.abstractmethod
    def axes(self):
        """
        Mapping from metadata keys to axes with which they are associated.

        Metadata not associated with any axes need not be represented here.
        """

    @property
    @abc.abstractmethod
    def key_comments(self):
        """
        Mapping from metadata keys to associated comments.

        Metadata without a comment need not be represented here.
        """

    @property
    @abc.abstractmethod
    def data_shape(self):
        """
        The shape of the data with which the metadata is associated.
        """

    @abc.abstractmethod
    def add(self, name, value, key_comment=None, axes=None, overwrite=False):
        """
        Add a new piece of metadata to instance.

        Parameters
        ----------
        name: `str`
            The name/label of the metadata.

        value: Any
            The value of the metadata. If axes input is not None, this must have the
            same length/shape as those axes as defined by ``self.data_shape``.

        key_comment: `str` or `None`
            Any comment associated with this metadata. Set to None if no comment desired.

        axes: `int`, iterable of `int`, or `None`
            The axis/axes with which the metadata is linked. If not associated with any
            axis, set this to None.

        overwrite: `bool`, optional
            If True, overwrites the entry of the name name if already present.
        """

    @property
    @abc.abstractmethod
    def slice(self):
        """
        A helper class which, when sliced, returns a new NDMeta with axis- and grid-aligned metadata sliced.

        Example
        -------
        >>> sliced_meta = meta.slice[0:3, :, 2] # doctest: +SKIP
        """

    @abc.abstractmethod
    def rebin(self, rebinned_axes, new_shape):
        """
        Adjusts grid-aware metadata to stay consistent with rebinned data.
        """


class NDMeta(dict, NDMetaABC):
    # Docstring in ABC
    __ndcube_can_slice__ = True
    __ndcube_can_rebin__ = True

    def __init__(self, meta=None, key_comments=None, axes=None, data_shape=None):
        self._original_meta = meta
        if data_shape is None:
            self._data_shape = np.array([], dtype=int)
        else:
            self._data_shape = np.asarray(data_shape).astype(int)

        if meta is None:
            meta = {}
        super().__init__(meta.items())
        meta_keys = meta.keys()

        if key_comments is None:
            self._key_comments = {}
        else:
            if not set(key_comments.keys()).issubset(set(meta_keys)):
                raise ValueError(
                    "All comments must correspond to a value in meta under the same key."
                )
            self._key_comments = key_comments

        if axes is None:
            self._axes = {}
        else:
            axes = dict(axes)
            if not set(axes.keys()).issubset(set(meta_keys)):
                raise ValueError(
                    "All axes must correspond to a value in meta under the same key.")
            self._axes = {key:self._sanitize_axis_value(axis, meta[key], key)
                          for key, axis in axes.items()}

    def _sanitize_axis_value(self, axis, value, key):
        axis_err_msg = ("Values in axes must be an integer or iterable of integers giving "
                        f"the data axis/axes associated with the metadata. axis = {axis}.")
        if isinstance(axis, numbers.Integral):
            axis = (axis,)
        if len(axis) == 0:
            return ValueError(axis_err_msg)
        # Verify each entry in axes is an iterable of ints or a scalar.
        if not (isinstance(axis, collections.abc.Iterable)
                and all(isinstance(i, numbers.Integral) for i in axis)):
            return ValueError(axis_err_msg)
        # If metadata's axis/axes include axis beyond current data shape, extend it.
        data_shape = self.data_shape
        if max(axis) >= len(data_shape):
            data_shape = np.concatenate((data_shape,
                                         np.zeros(max(axis) + 1 - len(data_shape), dtype=int)))
        # Check whether metadata is compatible with data shape based on shapes
        # of metadata already present.
        axis = np.asarray(axis)
        if _not_scalar(value):
            axis_shape = data_shape[axis]
            if not _is_axis_aligned(value, axis_shape):
                # If metadata corresponds to previously unconstrained axis, update data_shape.
                idx0 = axis_shape == 0
                if idx0.any():
                    axis_shape[idx0] = np.array(_get_metadata_shape(value))[idx0]
                    data_shape[axis] = axis_shape
                # Confirm metadata is compatible with data shape.
                if not _is_grid_aligned(value, axis_shape):
                    raise ValueError(
                        f"{key} must have same shape {tuple(data_shape[axis])} "
                        f"as its associated axes {axis}, ",
                        f"or same length as number of associated axes ({len(axis)}). "
                        f"Has shape {value.shape if hasattr(value, 'shape') else len(value)}")
        elif len(axis) != 1:
            raise ValueError("Scalar and str metadata can only be assigned to one axis. "
                             f"key = {key}; value = {value}; axes = {axis}")
        self._data_shape = data_shape
        return axis

    @property
    def key_comments(self):
        return self._key_comments

    @property
    def axes(self):
        return self._axes

    @property
    def data_shape(self):
        return self._data_shape

    @data_shape.setter
    def data_shape(self, new_shape):
        """
        Set data shape to new shape.

        Must agree with shapes of any axes already associated with metadata

        Parameters
        ----------
        new_shape: array-like
            The new shape of the data. Elements must of of type `int`.
        """
        new_shape = np.round(new_shape).astype(int)
        if (new_shape < 0).any():
            raise ValueError("new_shape cannot include negative numbers.")
        # Confirm input shape agrees with shapes of pre-existing metadata.
        old_shape = self.data_shape
        if len(new_shape) != len(old_shape) and len(self._axes) > 0:
            n_meta_axes = max([ax.max() for ax in self._axes.values()]) + 1
            old_shape = np.zeros(n_meta_axes, dtype=int)
            for key, ax in self._axes.items():
                old_shape[ax] = np.asarray(self[key].shape)
        # Axes of length 0 are deemed to be of unknown length, and so do not have to match.
        idx, = np.where(old_shape > 0)
        if len(idx) > 0 and (old_shape[idx] != new_shape[idx]).any():
            raise ValueError("new_shape not compatible with pre-existing metadata. "
                             f"old shape = {old_shape}, new_shape = {new_shape}")
        self._data_shape = new_shape

    def add(self, name, value, key_comment=None, axes=None, overwrite=False):
        # Docstring in ABC.
        if name in self.keys() and overwrite is not True:
            raise KeyError(f"'{name}' already exists. "
                           "To update an existing metadata entry set overwrite=True.")
        if key_comment is not None:
            self._key_comments[name] = key_comment
        if axes is not None:
            axes = self._sanitize_axis_value(axes, value, name)
            self._axes[name] = axes
            # Adjust data shape if not already set.
            axis_shape = self._data_shape[np.asarray(axes)]
            if _is_grid_aligned(value, axis_shape) and (self._data_shape[self._axes[name]] == 0).any():
                value_shape = np.asarray(value.shape)
                data_shape = self._data_shape
                # If new value represents axes not yet represented in Meta object,
                # add zero-length axes in their place to be filled in.
                if len(value_shape) > len(data_shape):
                    data_shape = np.concatenate(
                        (data_shape, np.zeros(len(value_shape) - len(data_shape), dtype=int)))
                idx_value, = np.where(data_shape[axes] == 0)
                data_shape[axes[idx_value]] = value_shape[idx_value]
                self._data_shape = data_shape
        elif name in self._axes:
            del self._axes[name]
        # This must be done after updating self._axes otherwise it may error.
        self.__setitem__(name, value)

    def __delitem__(self, name):
        if name in self._key_comments:
            del self._key_comments[name]
        if name in self._axes:
            del self._axes[name]
        super().__delitem__(name)

    def __setitem__(self, key, val):
        axis = self.axes.get(key, None)
        if axis is not None:
            if _not_scalar(val):
                axis_shape = tuple(self.data_shape[axis])
                if not _is_grid_aligned(val, axis_shape) and not _is_axis_aligned(val, axis_shape):
                    raise TypeError(
                        f"{key} is already associated with axis/axes {axis}. val must therefore "
                        f"must either have same length as number associated axes ({len(axis)}), "
                        f"or the same shape as associated data axes {tuple(self.data_shape[axis])}. "
                        f"val shape = {val.shape if hasattr(val, 'shape') else (len(val),)}\n"
                        "We recommend using the 'add' method to set values.")
        super().__setitem__(key, val)

    @property
    def original_meta(self):
        return MappingProxyType(self._original_meta)

    @property
    def slice(self):
        # Docstring in ABC.
        return _NDMetaSlicer(self)

    def rebin(self, bin_shape):
        """
        Adjusts axis-aware metadata to stay consistent with a rebinned `~ndcube.NDCube`.

        This is done by simply removing the axis-awareness of metadata associated with
        rebinned axes. The metadata itself is not changed or removed. This operation
        does not remove axis-awareness from metadata only associated with non-rebinned
        axes, i.e. axes whose corresponding entries in ``bin_shape`` are 1.

        Parameters
        ----------
        bin_shape : array-like
            The number of pixels in a bin in each dimension.
        """
        # Sanitize input
        bin_shape = np.round(bin_shape).astype(int)
        data_shape = self.data_shape
        bin_shape = bin_shape[:len(data_shape)]  # Drop info on axes not defined by NDMeta.
        if (np.mod(data_shape, bin_shape) != 0).any():
            raise ValueError("bin_shape must be integer factors of their associated axes.")
        # Remove axis-awareness from grid-aligned metadata associated with rebinned axes.
        rebinned_axes = set(np.where(bin_shape != 1)[0])
        new_meta = copy.deepcopy(self)
        null_set = set()
        for name, axes in self.axes.items():
            if (_is_grid_aligned(self[name], data_shape[axes])
                and set(axes).intersection(rebinned_axes) != null_set):
                del new_meta._axes[name]
        # Update data shape.
        new_meta._data_shape = new_meta._data_shape // bin_shape
        return new_meta


class _NDMetaSlicer:
    """
    Helper class to slice an NDMeta instance using a slicing item.

    Parameters
    ----------
    meta: `NDMetaABC`
        The metadata object to slice.
    """
    def __init__(self, meta):
        self.meta = meta

    def __getitem__(self, item):
        data_shape = self.meta.data_shape
        if len(data_shape) == 0:
            raise TypeError("NDMeta object does not have a shape and so cannot be sliced.")

        new_meta = copy.deepcopy(self.meta)
        naxes = len(data_shape)
        if isinstance(item, (numbers.Integral, slice)):
            item = [item]
        if len(item) < naxes:
            item = np.array(list(item) + [slice(None)] * (naxes - len(item)), dtype=object)
        elif len(item) > naxes:
            # If item applies to more axes than have been defined in NDMeta,
            # ignore items applying to those additional axes.
            item = np.array(item[:naxes])
        else:
            item = np.asarray(item)
        # Replace non-int item elements corresponding to length-0 axes
        # with slice(None) so data shape is not altered.
        idx = [not isinstance(i, numbers.Integral) and s == 0 for i, s in zip(item, data_shape)]
        idx = np.arange(len(idx))[idx]
        item[idx] = np.array([slice(None)] * len(idx))

        # Edit data shape and calculate which axis will be dropped.
        dropped_axes = np.zeros(naxes, dtype=bool)
        new_shape = new_meta.data_shape
        for i, axis_item in enumerate(item):
            if isinstance(axis_item, numbers.Integral):
                dropped_axes[i] = True
            elif isinstance(axis_item, slice):
                start = axis_item.start
                if start is None:
                    start = 0
                if start < 0:
                    start = data_shape[i] - start
                stop = axis_item.stop
                if stop is None:
                    stop = data_shape[i]
                if stop < 0:
                    stop = data_shape[i] - stop
                new_shape[i] = stop - start
            else:
                raise TypeError("Unrecognized slice type. "
                                "Must be an int, slice and tuple of the same.")
        kept_axes = np.invert(dropped_axes)
        new_meta._data_shape = new_shape[kept_axes]

        # Slice all metadata associated with axes.
        for key, value in self.meta.items():
            if (axis := self.meta.axes.get(key, None)) is None:
                continue
            drop_key = False
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
            axis_shape = tuple(self.meta.data_shape[axis])
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
                except Exception:  # noqa: BLE001
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
                del new_meta[key]
            else:
                new_meta.add(key, new_value, self.meta.key_comments.get(key, None), new_axis,
                             overwrite=True)

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


def _get_metadata_shape(value):
    return value.shape if hasattr(value, "shape") else (len(value),)

def _is_grid_aligned(value, axis_shape):
    if _is_scalar(value):
        return False
    value_shape = _get_metadata_shape(value)
    if value_shape != tuple(axis_shape):
        return False
    return True


def _is_axis_aligned(value, axis_shape):
    len_value = len(value) if _not_scalar(value) else 1
    return not _is_grid_aligned(value, axis_shape) and len_value == len(axis_shape)
