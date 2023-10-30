from numbers import Integral

import numpy as np

from astropy.wcs import WCS
from astropy.wcs.wcsapi import SlicedLowLevelWCS
from astropy.wcs.wcsapi.wrappers.base import BaseWCSWrapper

from ndcube.wcs.wrappers import ResampledLowLevelWCS

__all__ = ["unwrap_wcs_to_fitswcs"]

def unwrap_wcs_to_fitswcs(wcs):
    """Create FITS-WCS equivalent to (nested) WCS wrapper object.

    Underlying WCS must be FITS-WCS.
    No axes are dropped from original FITS-WCS, even if sliced by an integer.
    Instead, integer-sliced axes is sliced to length-1 and marked True in the
    ``dropped_data_axes`` ouput.
    Currently supported wrapper classes include `astropy.wcs.wcsapi.SlicedLowLevelWCS`
    and `ndcube.wcs.wrappers.ResampledLowLevelWCS`.

    Parameters
    ----------
    wcs: `BaseWCSWrapper`
        The WCS Wrapper object. Base level WCS implementation must be FITS-WCS.

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
        raise TypeError(f"Base-level WCS must be type {type(WCS)}. Found: {type(low_level_wcs)}")
    fitswcs = low_level_wrapper
    dropped_data_axes = np.zeros(fitswcs.naxis, dtype=bool)
    # Unwrap each wrapper in reverse order and edit fitswcs.
    for low_level_wrapper in wrapper_chain[::-1]:
        if isinstance(low_level_wrapper, SlicedLowLevelWCS):
            slice_items = np.array([slice(None)] * fitswcs.naxis)
            slice_items[dropped_data_axes == False] = low_level_wrapper._slices_array
            fitswcs, dda = _slice_fitswcs(fitswcs, slice_items, numpy_order=True)
            dropped_data_axes[dda] = True
        elif isinstance(low_level_wrapper, ResampledLowLevelWCS):
            factor = np.ones(fitswcs.naxis, dtype=int)
            offset = np.zeros(fitswcs.naxis, dtype=int)
            kept_data_axes = dropped_data_axes == False
            factor[kept_data_axes] = low_level_wrapper._factor
            offset[kept_data_axes] = low_level_wrapper._offset
            fitswcs = _resample_fitswcs(fitswcs, factor, offset)
        else:
            raise TypeError("Unrecognized/unsupported WCS Wrapper type: {type(low_level_wrapper)}")
    return fitswcs, dropped_data_axes


def _slice_fitswcs(fitswcs, slice_items, numpy_order=True):
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

    Returns
    -------
    sliced_wcs: `astropy.wcs.WCS`
        The sliced FITS-WCS.
    dropped_data_axes: 1-D `numpy.ndarray`
        Denotes which axes must have been dropped from the data array by slicing wrappers.
        Order of axes (numpy or WCS) is dictated by ``numpy_order`` kwarg.
    """
    naxis = fitswcs.naxis
    dropped_data_axes = np.zeros(naxis, dtype=bool)
    # Sanitize inputs
    slice_items = list(slice_items)
    for i, item in enumerate(slice_items):
        # Determine length of axis.
        len_axis = fitswcs._naxis[naxis - 1 - i] if numpy_order else fitswcs._naxis[i]
        if isinstance(item, Integral):
            # Mark axis corresponding to int item as dropped from data array.
            dropped_data_axes[i] = True
            # Convert negative indices to positive equivalent.
            if item < 0:
                item = len_axis + item
            slice_items[i] = slice(item, item + 1)
        elif isinstance(item, slice):
            # Convert negative indices inside slice item to positive equivalent.
            start = (len_axis + item.start if (item.start is not None and item.start < 0)
                     else item.start)
            stop = len_axis + item.stop if (item.stop is not None and item.stop < 0) else item.stop
            slice_items[i] = slice(start, stop, item.step)
        else:
            raise TypeError("All slice_items must be a slice or an int. "
                            f"type(slice_items[{i}]) = {type(slice_items[i])}")
    # Slice WCS
    sliced_wcs = fitswcs.slice(slice_items, numpy_order=numpy_order)
    return sliced_wcs, dropped_data_axes


def _resample_fitswcs(fitswcs, factor, offset=0):
    """Resample the plate scale of a FITS-WCS by a given factor.

    ``factor`` and ``offset`` inputs are in pixel order.

    Parameters
    ----------
    fitswcs: `astropy.wcs.WCS`
        The FITS-WCS object to be resampled.
    factor: 1-D array-like or scalar
        The factor by which the FITS-WCS is resampled.
        Must be same length as number of axes in ``fitswcs``.
        If scalar, the same factor is applied to all axes.
    offset: 1-D array-like or scalar
        The location on the intial pixel grid which corresponds to zero on the
        resampled pixel grid. If scalar, the same offset is applied to all axes.

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
    fitswcs.wcs.crpix -= offset
    return fitswcs
