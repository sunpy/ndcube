import inspect
from functools import wraps
from itertools import chain

import astropy.nddata
import numpy as np
from astropy.wcs.wcsapi import BaseHighLevelWCS, HighLevelWCSWrapper, SlicedLowLevelWCS

from ndcube.utils import wcs as wcs_utils

__all__ = ["sanitize_wcs", "sanitize_crop_inputs", "get_crop_item_from_points",
           "propagate_rebin_uncertainties"]


def sanitize_wcs(func):
    """
    A wrapper for NDCube methods to sanitise the wcs argument.

    This decorator is only designed to be used on methods of NDCube.

    It will find the wcs argument, keyword or positional and if it is `None`, set
    it to ``self.wcs``.
    It will then verify that the WCS has a matching number of pixel dimensions
    to the dimensionality of the array. It will finally verify that the object
    passed is a HighLevelWCS object, or an ExtraCoords object.
    """
    # This needs to be here to prevent a circular import
    from ndcube.extra_coords.extra_coords import ExtraCoords

    @wraps(func)
    def wcs_wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        params = sig.bind(*args, **kwargs)
        wcs = params.arguments.get('wcs', None)
        self = params.arguments['self']

        if wcs is None:
            wcs = self.wcs

        if not isinstance(wcs, ExtraCoords):
            if not wcs.pixel_n_dim == self.data.ndim:
                raise ValueError("The supplied WCS must have the same number of "
                                 "pixel dimensions as the NDCube object. "
                                 "If you specified `cube.extra_coords.wcs` "
                                 "please just pass `cube.extra_coords`.")

        if not isinstance(wcs, (BaseHighLevelWCS, ExtraCoords)):
            raise TypeError("wcs argument must be a High Level WCS or an ExtraCoords object.")

        params.arguments['wcs'] = wcs

        return func(*params.args, **params.kwargs)

    return wcs_wrapper


def sanitize_crop_inputs(points, wcs):
    """Sanitize inputs to NDCube crop methods.

    First arg returned signifies whether the inputs imply that cropping
    should be performed or not.
    """
    points = list(points)
    n_points = len(points)
    n_coords = [None] * n_points
    values_are_none = [False] * n_points
    for i, point in enumerate(points):
        # Ensure each point is a list
        if isinstance(point, (tuple, list)):
            points[i] = list(point)
        else:
            points[i] = [point]
        # Record number of objects in each point.
        # Later we will ensure all points have same number of objects.
        n_coords[i] = len(points[i])
        # Confirm whether point contains at least one None entry.
        if all([coord is None for coord in points[i]]):
            values_are_none[i] = True
    # If no points contain a coord, i.e. if all entries in all points are None,
    # set no-op flag to True and exit.
    if all(values_are_none):
        return True, points, wcs
    # Not not all points are of same length, error.
    if len(set(n_coords)) != 1:
        raise ValueError("All points must have same number of coordinate objects."
                         f"Number of objects in each point: {n_coords}")
    # Import must be here to avoid circular import.
    from ndcube.extra_coords.extra_coords import ExtraCoords
    if isinstance(wcs, ExtraCoords):
        # Determine how many dummy axes are needed
        n_dummy_axes = len(wcs._cube_array_axes_without_extra_coords)
        if n_dummy_axes > 0:
            points = [point + [None] * n_dummy_axes for point in points]
        # Convert extra coords to WCS describing whole cube.
        wcs = wcs.cube_wcs
    # Ensure WCS is low level.
    if isinstance(wcs, BaseHighLevelWCS):
        wcs = wcs.low_level_wcs
    return False, points, wcs


def get_crop_item_from_points(points, wcs, crop_by_values):
    """
    Find slice item that crops to minimum cube in array-space containing specified world points.

    Parameters
    ----------
    points : iterable of iterables
        Each iterable represents a point in real world space.
        Each element in a point gives the real world coordinate value of the point
        in high-level coordinate objects or quantities.
        (Must be consistently high or low level within and across points.)
        Objects must be in the order required by
        wcs.world_to_array_index/world_to_array_index_values.

    wcs : `~astropy.wcs.wcsapi.BaseHighLevelWCS`, `~astropy.wcs.wcsapi.BaseLowLevelWCS`
        The WCS to use to convert the world coordinates to array indices.

    crop_by_values : `bool`
        Denotes whether cropping is done using high-level objects or "values",
        i.e. low-level objects.

    Returns
    -------
    item : `tuple` of `slice`
        The slice item for each axis of the cube which, when applied to the cube,
        will return the minimum cube in array-index-space that contains all the
        input world points.
    """
    # Define a list of lists to hold the array indices of the points
    # where each inner list gives the index of all points for that array axis.
    combined_points_array_idx = [[]] * wcs.pixel_n_dim
    # For each point compute the corresponding array indices.
    for point in points:
        # Get the arrays axes associated with each element in point.
        if crop_by_values:
            point_inputs_array_axes = []
            for i in range(wcs.world_n_dim):
                pix_axes = np.array(
                    wcs_utils.world_axis_to_pixel_axes(i, wcs.axis_correlation_matrix))
                point_inputs_array_axes.append(tuple(
                    wcs_utils.convert_between_array_and_pixel_axes(pix_axes, wcs.pixel_n_dim)))
            point_inputs_array_axes = tuple(point_inputs_array_axes)
        else:
            point_inputs_array_axes = wcs_utils.array_indices_for_world_objects(
                HighLevelWCSWrapper(wcs))
        # Get indices of array axes which correspond to only None inputs in point
        # as well as those that correspond to a coord.
        point_indices_with_inputs = []
        array_axes_with_input = []
        for i, coord in enumerate(point):
            if coord is not None:
                point_indices_with_inputs.append(i)
                array_axes_with_input.append(point_inputs_array_axes[i])
        array_axes_with_input = set(chain.from_iterable(array_axes_with_input))
        array_axes_without_input = set(range(wcs.pixel_n_dim)) - array_axes_with_input
        # Slice out the axes that do not correspond to a coord
        # from the WCS and the input point.
        wcs_slice = np.array([slice(None)] * wcs.pixel_n_dim)
        if len(array_axes_without_input):
            wcs_slice[np.array(list(array_axes_without_input))] = 0
        sliced_wcs = SlicedLowLevelWCS(wcs, slices=tuple(wcs_slice))
        sliced_point = np.array(point, dtype=object)[np.array(point_indices_with_inputs)]
        # Derive the array indices of the input point and place each index
        # in the list corresponding to its axis.
        if crop_by_values:
            point_array_indices = sliced_wcs.world_to_array_index_values(*sliced_point)
            # If returned value is a 0-d array, convert to a length-1 tuple.
            if isinstance(point_array_indices, np.ndarray) and point_array_indices.ndim == 0:
                point_array_indices = (point_array_indices.item(),)
            else:
                # Convert from scalar arrays to scalars
                point_array_indices = tuple(a.item() for a in point_array_indices)
        else:
            point_array_indices = HighLevelWCSWrapper(sliced_wcs).world_to_array_index(
                *sliced_point)
            # If returned value is a 0-d array, convert to a length-1 tuple.
            if isinstance(point_array_indices, np.ndarray) and point_array_indices.ndim == 0:
                point_array_indices = (point_array_indices.item(),)
        for axis, index in zip(array_axes_with_input, point_array_indices):
            combined_points_array_idx[axis] = combined_points_array_idx[axis] + [index]
    # Define slice item with which to slice cube.
    item = []
    result_is_scalar = True
    for axis_indices in combined_points_array_idx:
        if axis_indices == []:
            result_is_scalar = False
            item.append(slice(None))
        else:
            min_idx = min(axis_indices)
            max_idx = max(axis_indices) + 1
            if max_idx - min_idx == 1:
                item.append(min_idx)
            else:
                item.append(slice(min_idx, max_idx))
                result_is_scalar = False
    # If item will result in a scalar cube, raise an error as this is not currently supported.
    if result_is_scalar:
        raise ValueError("Input points causes cube to be cropped to a single pixel. "
                         "This is not supported.")
    return tuple(item)


def propagate_rebin_uncertainties(uncertainty, data, mask, operation, operation_ignores_mask=False,
                                  propagation_operation=None, correlation=0, **kwargs):
    """
    Default algorithm for uncertainty propagation in :meth:`~ndcube.NDCube.rebin`.

    First dimension of uncertainty, data and mask inputs represent the pixels
    in the bin being aggregated by the rebin process while the latter dimensions
    must have the same shape as the rebinned data. The operation input is the
    function used to aggregate elements in the first dimension, e.g. `numpy.sum`.

    Parameters
    ----------
    uncertainty: `astropy.nddata.NDUncertainty`
        Cannot be instance of `astropy.nddata.UnknownUncertainty`.
        The uncertainties associated with the data. The first dimension represents
        pixels in each bin being aggregated while trailing dimensions must have
        the same shape as the rebinned data.
    data: array-like or `None`
        The data associated with the above uncertainties.
        Must have same shape as above.
    mask: array-like of `bool` or `None`
        Indicates whether any uncertainty elements should be ignored in propagation.
        If True, corresponding uncertainty element is ignored. If False, it is used.
        Must have same shape as above.
    operation: function
        The function used to aggregate the data for which the uncertainties are being
        propagated here.
    operation_ignores_mask: `bool`
        Determines whether masked values are used or excluded from calculation.
        Default is False causing masked data and uncertainty to be excluded.
    propagation_operation: function
        The operation which defines how the uncertainties are propagated.
        This can differ from operation, e.g. if operation is sum, then
        propagation_operation should be add.
    correlation: `int`
        Passed to `astropy.nddata.NDUncertainty.propagate`. See that method's docstring.
        Default=0.

    Returns
    -------
    new_uncertainty: same type as uncertainty input.
        The propagated uncertainty. Same shape as input uncertainty without its
        first dimension.
    """
    flat_axis = 0
    operation_is_mean = True if operation in {np.mean, np.nanmean} else False
    operation_is_nantype = True if operation in {np.nansum, np.nanmean, np.nanprod} else False
    # If propagation_operation kwarg not set manually, try to set it based on operation kwarg.
    if not propagation_operation:
        if operation in {np.sum, np.nansum, np.mean, np.nanmean}:
            propagation_operation = np.add
        elif operation in {np.product, np.prod, np.nanprod}:
            propagation_operation = np.multiply
        else:
            raise ValueError("propagation_operation not recognized.")
    # Build mask if not provided.
    new_uncertainty = uncertainty[0]  # Define uncertainty for initial iteration step.
    if operation_ignores_mask or mask is None:
        mask = False
    if mask is False:
        if operation_is_nantype:
            nan_mask = np.isnan(data)
            if nan_mask.any():
                mask = nan_mask
                idx = np.logical_not(mask)
                mask1 = mask[1:]
        else:
            # If there is no mask and operation is not nan-type, build generator
            # so non-mask can still be iterated.
            n_pix_per_bin = data.shape[flat_axis]
            new_shape = data.shape[1:]
            mask1 = (False for i in range(1, n_pix_per_bin))
    else:
        # Mask uncertainties corresponding to nan data if operation is nantype.
        if operation_is_nantype:
            mask[np.isnan(data)] = True
        # Set masked uncertainties in first mask to 0
        # as they shouldn't count towards final uncertainty.
        mask1 = mask[1:]
        idx = np.logical_not(mask)
        uncertainty.array[mask] = 0
        new_uncertainty.array[mask[0]] = 0
    # Propagate uncertainties.
    # Note uncertainty must be associated with a parent nddata for some propagations.
    cumul_data = data[0]
    if mask is not False and operation_ignores_mask is False:
        cumul_data[idx[0]] = 0
    parent_nddata = astropy.nddata.NDData(cumul_data, uncertainty=new_uncertainty)
    new_uncertainty.parent_nddata = parent_nddata
    for j, mask_slice in enumerate(mask1):
        i = j + 1
        cumul_data = operation(data[:i+1]) if mask is False else operation(data[:i+1][idx[:i+1]])
        data_slice = astropy.nddata.NDData(data=data[i], mask=mask_slice,
                                           uncertainty=uncertainty[i])
        new_uncertainty = new_uncertainty.propagate(propagation_operation, data_slice,
                                                    cumul_data, correlation)
        parent_nddata = astropy.nddata.NDData(cumul_data, uncertainty=new_uncertainty)
        new_uncertainty.parent_nddata = parent_nddata
    # If aggregation operation is mean, uncertainties must be divided by
    # number of unmasked pixels in each bin.
    if operation_is_mean and propagation_operation is np.add:
        if mask is False:
            new_uncertainty.array /= n_pix_per_bin
        else:
            unmasked_per_bin = np.logical_not(mask).astype(int).sum(axis=flat_axis)
            new_uncertainty.array /= np.clip(unmasked_per_bin, 1, None)
    return new_uncertainty
