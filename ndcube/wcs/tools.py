from numbers import Integral

import numpy as np

from astropy.wcs import WCS
from astropy.wcs.wcsapi import SlicedLowLevelWCS
from astropy.wcs.wcsapi.wrappers.base import BaseWCSWrapper

from ndcube.wcs.wrappers import ResampledLowLevelWCS

__all__ = ["unwrap_wcs_to_fitswcs"]


def unwrap_wcs_to_fitswcs(wcs):
    """
    Create FITS-WCS equivalent to (nested) WCS wrapper object.

    Underlying WCS must be FITS-WCS.
    No axes are dropped from original FITS-WCS, even if sliced by an integer.
    Instead, integer-sliced axes is sliced to length-1 and marked True in the
    ``dropped_data_axes`` output.
    Currently supported wrapper classes include `astropy.wcs.wcsapi.SlicedLowLevelWCS`
    and `ndcube.wcs.wrappers.ResampledLowLevelWCS`.

    Parameters
    ----------
    wcs: `~astropy.wcs.wcsapi.BaseWCSWrapper`
        The WCS Wrapper object.
        Base level WCS implementation must be FITS-WCS.

    Returns
    -------
    fitswcs: `astropy.wcs.WCS`
        The equivalent FITS-WCS object.
    dropped_data_axes: 1-D `numpy.ndarray`
        Denotes which axes must have been dropped from the data array by slicing wrappers.
        Axes are in array/numpy order, reversed compared to WCS.
    """
    # If wcs is already a FITS-WCS, return it.
    low_level_wrapper = wcs.low_level_wcs if hasattr(wcs, "low_level_wcs") else wcs
    if isinstance(low_level_wrapper, WCS):
        return low_level_wrapper, np.zeros(low_level_wrapper.naxis, dtype=bool)
    # Determine chain of wrappers down to the FITS-WCS.
    wrapper_chain = []
    while isinstance(low_level_wrapper, BaseWCSWrapper):
        wrapper_chain.append(low_level_wrapper)
        low_level_wrapper = low_level_wrapper._wcs
        if hasattr(low_level_wrapper, "low_level_wcs"):
            low_level_wrapper = low_level_wrapper.low_level_wcs
    if not isinstance(low_level_wrapper, WCS):
        raise TypeError(f"Base-level WCS must be type {type(WCS)}. Found: {type(low_level_wrapper)}")
    fitswcs = low_level_wrapper
    dropped_data_axes = np.zeros(fitswcs.naxis, dtype=bool)
    # Unwrap each wrapper in reverse order and edit fitswcs.
    for low_level_wrapper in wrapper_chain[::-1]:
        if isinstance(low_level_wrapper, SlicedLowLevelWCS):
            slice_items = np.array([slice(None)] * fitswcs.naxis)
            slice_items[dropped_data_axes == False] = low_level_wrapper._slices_array  # numpy order  # NOQA: E712
            fitswcs, dda = _slice_fitswcs(fitswcs, slice_items, numpy_order=True)
            dropped_data_axes[dda] = True
        elif isinstance(low_level_wrapper, ResampledLowLevelWCS):
            factor = np.ones(fitswcs.naxis)
            offset = np.zeros(fitswcs.naxis)
            kept_wcs_axes = dropped_data_axes[::-1] == False  # WCS-order  # NOQA: E712
            factor[kept_wcs_axes] = low_level_wrapper._factor
            offset[kept_wcs_axes] = low_level_wrapper._offset
            fitswcs = _resample_fitswcs(fitswcs, factor, offset)
        else:
            raise TypeError("Unrecognized/unsupported WCS Wrapper type: {type(low_level_wrapper)}")
    return fitswcs, dropped_data_axes


def _slice_fitswcs(fitswcs, slice_items, numpy_order=True, shape=None):
    """
    Slice a FITS-WCS.

    If an `int` is given in ``slice_items``, the corresponding axis is not dropped.
    But the new 0th pixel will correspond the index given by the `int` in the
    original WCS.

    Parameters
    ----------
    fitswcs: `astropy.wcs.WCS`
        The FITS-WCS object to be sliced.
    slice_items: iterable of `slice` objects or `int`
        The slices to by applied to each axis. If an `int` is provided, the axis
        is sliced to length-1, but not dropped. However, its corresponding entry
        in the ``dropped_data_axes`` output is marked True.
    numpy_order: `bool`
        If True, slices in ``slice_items`` are in array/numpy order, which is
        reversed compared to the WCS order.
    shape: sequence of `int`, optional
        The length of each axis.  Only used if negative indices are supplied
        in ``slice_items``.  If not supplied, set to ``fitswcs._naxis``.
        Order defined by numpy_order kwarg.

    Returns
    -------
    sliced_wcs: `astropy.wcs.WCS`
        The sliced FITS-WCS.
    dropped_data_axes: 1-D `numpy.ndarray`
        Denotes which axes must have been dropped from the data array by slicing wrappers.
        Order of axes (numpy or WCS) is dictated by ``numpy_order`` kwarg.
    """
    def negative_index_error_msg(x): return (
        f"Negative indexing not supported as {x}th axis length is 0 in "
        "underlying FITS-WCS. Supply axes lengths via shape kwarg.")
    naxis = fitswcs.naxis
    dropped_data_axes = np.zeros(naxis, dtype=bool)
    # Sanitize inputs
    if shape is None:
        shape = fitswcs._naxis
        if numpy_order:
            shape = shape[::-1]
    else:
        if len(shape) != naxis:
            raise ValueError("shape kwarg must be same length as number of pixel axes "
                             f"in FITS-WCS, i.e. {naxis}")
        if not all(isinstance(s, Integral) for s in shape):
            raise TypeError("All elements of ``shape`` must be integers. "
                            f"shapes types = {[type(s) for s in shape]}")
    slice_items = list(slice_items)
    for i, (item, len_axis) in enumerate(zip(slice_items, shape)):
        if isinstance(item, Integral):
            # Mark axis corresponding to int item as dropped from data array.
            dropped_data_axes[i] = True
            # Convert negative indices to positive equivalent.
            if item < 0:
                if len_axis == 0:
                    raise ValueError(negative_index_error_msg(i))
                item = len_axis + item
            # Convert int item to slice so a FITS-WCS is returned after slicing.
            slice_items[i] = slice(item, item + 1)
        elif isinstance(item, slice):
            # Convert negative indices inside slice item to positive equivalent.
            start_neg = item.start is not None and item.start < 0
            stop_neg = item.stop is not None and item.stop < 0
            if start_neg or stop_neg:
                if len_axis == 0:
                    raise ValueError(negative_index_error_msg(i))
                start = len_axis + item.start if start_neg else item.start
                stop = len_axis + item.stop if stop_neg else item.stop
                slice_items[i] = slice(start, stop, item.step)
        else:
            raise TypeError("All slice_items must be a slice or an int. "
                            f"type(slice_items[{i}]) = {type(slice_items[i])}")
    # Slice WCS
    sliced_wcs = fitswcs.slice(slice_items, numpy_order=numpy_order)
    return sliced_wcs, dropped_data_axes


def _resample_fitswcs(fitswcs, factor, offset=0):
    """
    Resample the plate scale of a FITS-WCS by a given factor.

    ``factor`` and ``offset`` inputs are in pixel order.

    Parameters
    ----------
    fitswcs: `astropy.wcs.WCS`
        The FITS-WCS object to be resampled.
    factor: 1-D array-like or scalar
        The factor by which the FITS-WCS is resampled.
        Must be same length as number of axes in ``fitswcs``.
        If scalar, the same factor is applied to all axes.
        Factors must be given in WCS-order (opposite to data axes order).
    offset: 1-D array-like or scalar
        The location on the initial pixel grid which corresponds to zero on the
        resampled pixel grid. If scalar, the same offset is applied to all axes.
        Offsets must be given in WCS-order (opposite to data axes order).

    Returns
    -------
    resampled_wcs: `astropy.wcs.WCS`
        The resampled FITS-WCS.
    """
    # Sanitize inputs.
    factor = np.asarray(factor)
    if len(factor) != fitswcs.naxis:
        raise ValueError(f"Length of factor must equal number of dimensions {fitswcs.naxis}.")
    offset = np.asarray(offset)
    if len(offset) != fitswcs.naxis:
        raise ValueError(f"Length of offset must equal number of dimensions {fitswcs.naxis}.")
    # Scale plate scale and shift by offset.
    fitswcs.wcs.cdelt *= factor
    fitswcs.wcs.crpix = (fitswcs.wcs.crpix + offset) / factor
    fitswcs._naxis = list(np.round(np.array(fitswcs._naxis) / factor).astype(int))
    return fitswcs
