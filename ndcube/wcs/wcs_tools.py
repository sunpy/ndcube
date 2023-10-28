from numbers import Integral

import numpy as np

from astropy.wcs import WCS
from astropy.wcs.wcsapi import SlicedLowLevelWCS
from astropy.wcs.wcsapi.wrappers.base import BaseWCSWrapper
from ndcube.wcs.wrappers import ResampledLowLevelWCS

__all__ = ["unwrap_wcs_wrappers_to_fitswcs"]


def unwrap_wcs_wrappers_to_fitswcs(wcs):
    """Create FITS-WCS equivalent to WCS wrapper object.

    Underlying WCS must be FITS-WCS.

    Parameters
    ----------
    wcs: `BaseWCSWrapper`
        The WCS Wrapper object. Base level WCS implementation must be FITS-WCS.
        Current supported wrapper classes include `astropy.wcs.wcsapi.SlicedLowLevelWCS`
        and `ndcube.wcs.wrappers.ResampledLowLevelWCS`.

    Returns
    -------
    fitswcs: `astropy.wcs.WCS`
        The equivalent FITS-WCS object.
    """
    low_level_wrapper = wcs.low_level_wcs
    if isinstance(low_level_wrapper, WCS):
        return low_level_wrapper
    # Determine chain of wrappers down to the FITS-WCS.
    wrapper_chain = [type(low_level_wrapper)]
    while hasattr(low_level_wrapper, "_wcs"):
        low_level_wrapper = low_level_wrapper._wcs.low_level_wcs
        wrapper_chain.append(type(low_level_wrapper))
    if not isinstance(low_level_wrapper, WCS):
        raise TypeError(f"Base-level WCS must be type {type(WCS)}. Found: {type(low_level_wcs)}")
    fitswcs = low_level_wrapper
    for low_level_wrapper in wrapper_chain[::-1]:
        if isinstance(low_level_wrapper, ResampledLowLevelWCS):
            fitswcs = resample_fitswcs(fitswcs, low_level_wrapper.factor, low_level_wrapper.offset)
        elif isinstance(low_level_wrapper, SlicedLowLevelWCS):
            fitswcs = slice_fitswcs(fitswcs, low_level_wrapper._slices_pixel)
        else:
            raise TypeError("Unrecognized/unsupported WCS Wrapper type: {type(low_level_wrapper)}")
    return fitswcs


def slice_fitswcs(fitswcs, slice_items):
    """
    Slice a FITS-WCS.

    ``slice_items`` must be in pixel order.
    If an `int` is given in ``slice_items``, the corresponding axis is not dropped.
    But the new 0th pixel will correspond the index given by the `int` in the
    original WCS.

    Parameters
    ----------
    fitswcs: `astropy.wcs.WCS`
        The FITS-WCS object to be sliced.
    slice_items: `tuple` of `slice` objects or `ìnt`
        The slices to by applied to each axis.

    Returns
    -------
    sliced_wcs: `astropy.wcs.WCS`
        The sliced FITS-WCS.
    redundant_dims: 1-D `numpy.ndarray`
        Dimensions in the WCS for which the corresponding dimensions in the data array
        must have been dropped if same slice items were applied to it.
    """
    # Sanitize inputs.
    if not hasattr(slice_items, "__len__"):
        raise TypeError("slice_items must be iterable")
    if len(slice_items) != fitswcs.naxis:
        raise ValueError(f"slice_items must be same length as number of WCS axes: {fitswcs.naxis}")
    if not all(isinstance(item, (slice, Integral)) for item in slice_items):
    # Determine which axes of the original WCS have been dropped by the slicing.
    redundant_dims = np.array([isinstance(slice_item, Integral) for slice_item in slice_items])
    # Determine starting index of slice interval and identify redundant dims
    # that have must have been dropped by if same slicing was applied to data array.
    slice_starts = np.zeros(len(slice_items), dtype=int)
    redundant_dims = np.zeros(len(slice_items), dtype=bool)
    for i, item in enumerate(slice_items):
        if isinstance(item, slice):
            slice_starts[i] = 0 if item.start is None else item.start
        elif isinstance(item, Integral):
            slice_starts[i] = item
            redundant_dims[i] = True
        else:
             raise TypeError("All slice items must be a slice object or an integer.")
    # Edit CRPIX
    fitwcs.wcs.crpix -= slice_starts  # WARNING: Does not handle or catch negative indexing.
    return fitswcs, redundant_dims


def resample_fitswcs(fitswcs, factor, offset=0):
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
    if np.isscalar(factor):
        factor = [factor] * fitswcs.naxis
    factor = np.array(factor)
    if len(factor) != fitswcs.naxis:
        raise ValueError(f"Length of factor must equal number of dimensions {fitswcs.naxis}.")
    if np.isscalar(offset):
        offset = [offset] * fitswcs.naxis
    offset = np.array(offset)
    if len(offset) != fitswcs.naxis:
        raise ValueError(f"Length of offset must equal number of dimensions {fitswcs.naxis}.")
    # Scale plate scale and shift by offset.
    fitswcs.wcs.cdelt *= factor
    fitswcs.wcs.crpix += offset
    return fitswcs
