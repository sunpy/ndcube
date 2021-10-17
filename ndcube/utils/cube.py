import inspect
from functools import wraps
from itertools import chain

import numpy as np
from astropy.wcs.wcsapi import BaseHighLevelWCS, HighLevelWCSWrapper, SlicedLowLevelWCS

from ndcube.utils import wcs as wcs_utils

__all__ = ["sanitize_wcs", "sanitize_crop_inputs", "get_crop_item_from_points"]


def sanitize_wcs(func):
    """
    A wrapper for NDCube methods to sanitise the wcs argument.

    This decorator is only designed to be used on methods of NDCube.

    It will find the wcs argument, keyword or positional and if it is None, set
    it to `self.wcs`.
    It will then verify that the WCS has a matching number of pixel dimensions
    to the dimensionality of the array. It will finally verify that the object
    passed is a HighLevelWCS object, or an ExtraCoords object.
    """
    # This needs to be here to prevent a circular import
    from ndcube.extra_coords import ExtraCoords

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
    from ndcube.extra_coords import ExtraCoords
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
        (Must be consistenly high or low level within and across points.)
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
        for i, axis in zip(point_array_indices, array_axes_with_input):
            combined_points_array_idx[axis] = combined_points_array_idx[axis] + [i]
    # Define slice item with which to slice cube.
    item = tuple([slice(None) if axis_indices == []
                  else slice(min(axis_indices), max(axis_indices) + 1)
                  for axis_indices in combined_points_array_idx])
    return item
