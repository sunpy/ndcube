import inspect
from functools import wraps

import astropy.units as u
import numpy as np
from astropy.wcs.wcsapi import BaseHighLevelWCS, HighLevelWCSWrapper

from ndcube.utils import misc as misc_utils
from ndcube.utils.wcs_high_level_conversion import high_level_objects_to_values

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


def get_crop_item(lower_corner, upper_corner, wcs, data_shape):
    # Sanitize inputs.
    no_op, lower_corner, upper_corner, wcs = sanitize_crop_inputs(lower_corner, upper_corner, wcs)
    # Quit out early if we are no-op
    if no_op:
        return tuple([slice(None)] * len(data_shape))

    lower_corner, upper_corner = fill_in_crop_nones(lower_corner, upper_corner,
                                                    wcs, data_shape, False)

    if isinstance(wcs, BaseHighLevelWCS):
        wcs = wcs.low_level_wcs

    lower_corner_values = high_level_objects_to_values(*lower_corner, low_level_wcs=wcs)
    upper_corner_values = high_level_objects_to_values(*upper_corner, low_level_wcs=wcs)
    lower_corner_values = [u.Quantity(v, unit=u.Unit(unit), copy=False)
                           for v, unit in zip(lower_corner_values, wcs.world_axis_units)]
    upper_corner_values = [u.Quantity(v, unit=u.Unit(unit), copy=False)
                           for v, unit in zip(upper_corner_values, wcs.world_axis_units)]

    return get_crop_item_from_points(*points, wcs=wcs, data_shape=data_shape)


def get_crop_by_values_item(*points, wcs, data_shape, units=None):
    # Sanitize inputs.
    no_op, points, wcs = sanitize_crop_inputs(points, wcs)
    # Quit out early if we are no-op
    if no_op:
        return tuple([slice(None)] * len(data_shape))

    n_coords = len(points[0])
    if units is None:
        units = [None] * n_coords
    elif len(units) != n_coords:
        raise ValueError("units must be None or have same length as corner inputs.")

    # Convert float inputs to quantities using units.
    types_with_units = (u.Quantity, type(None))
    for i, point in enumerate(points):
        for j, (value, unit) in enumerate(zip(point, units)):
            value_is_float = not isinstance(value, types_with_units)
            if value_is_float:
                if unit is None:
                    raise TypeError(
                        "If an element of a point is not a Quantity or None, "
                        "the corresponding unit must be a valid astropy Unit or unit string."
                        f"index: {i}; lower type: {type(lower)}; "
                        f"upper type: {type(upper)}; unit: {unit}")
                points[i][j] = u.Quantity(value, unit=unit)

    #points = fill_in_crop_nones(points, wcs, data_shape, True)

    # Convert coordinates to units used by WCS as WCS.world_to_array_index
    # does not handle quantities.
    points = [misc_utils.convert_quantities_to_units(point, wcs.world_axis_units)
              for point in points]

    return get_crop_item_from_points(*points, wcs=wcs, data_shape=data_shape)


def sanitize_crop_inputs(points, wcs):
    """Sanitize inputs to NDCube crop methods.

    First arg returned signifies whether the inputs imply that cropping
    should be performed or not.
    """
    points = list(points)
    n_points = len(points)
    n_coords = [None] * n_points
    values_are_none = [None] * n_points
    for i, point in enumerate(points):
        # Ensure each point is a list
        if isinstance(point, (tuple, list)):
            points[i] = list(point)
        else:
            points[i] = [point]
        # Record number of objects in each point.
        # Later we will ensure all points have same number of objects.
        n_coords[i] = len(points[i])
        # Check which values in point are None.
        # Later we will use this to determine whether no operation is needed.
        # (Boolean arrays are compiled here in a list and then stacked later because
        # at thisstage we have not confirmed all points are of same length.
        # If they aren't then generating a stacked array here will cause an error
        # with a less helpful traceback.)
        values_are_none[i] = np.array([value is None for value in points[i]])
    # Not not all points are of same length, error.
    if len(set(n_coords)) != 1:
        raise ValueError("All points must have same number of coordinate objects."
                         f"Number of objects in each point: {n_coords}")
    # Quit out early if we are no-op
    if np.stack(values_are_none).all():
        return True, lower_corner, upper_corner, wcs
    # Import must be here to avoid circular import.
    from ndcube.extra_coords import ExtraCoords
    if isinstance(wcs, ExtraCoords):
        # Add None inputs to upper and lower corners for new dummy axes.
        n_dummy_axes = len(wcs._cube_array_axes_without_extra_coords)
        points = [point + [None] * n_dummy_axes for point in points]
        # Convert extra coords to WCS describing whole cube.
        wcs = wcs.cube_wcs
    return False, points, wcs


def fill_in_crop_nones(lower_corner, upper_corner, wcs, data_shape, crop_by_values):
    """
    Replace any instance of None in the inputs with the bounds for that axis.
    """
    # Determine which algorithm to use to convert array indices to world coords.
    if crop_by_values:
        if isinstance(wcs, BaseHighLevelWCS):
            array_index_to_world = wcs.low_level_wcs.array_index_to_world_values
        else:
            array_index_to_world = wcs.array_index_to_world_values
    else:
        if isinstance(wcs, BaseHighLevelWCS):
            array_index_to_world = wcs.array_index_to_world
        else:
            print(type(wcs))
            array_index_to_world = HighLevelWCSWrapper(wcs).array_index_to_world

    # If user did not provide all intervals,
    # calculate missing intervals based on whole cube range along those axes.
    lower_nones = np.array([lower is None for lower in lower_corner])
    upper_nones = np.array([upper is None for upper in upper_corner])
    if lower_nones.any() or upper_nones.any():
        # Calculate real world coords for first and last index for all axes.
        array_intervals = [[0, np.round(d - 1).astype(int)] for d in data_shape]
        intervals = array_index_to_world(*array_intervals)
        # Overwrite None corner values with world coords of first or last index.
        iterable = zip(lower_nones, upper_nones, intervals)
        for i, (lower_is_none, upper_is_none, interval) in enumerate(iterable):
            if lower_is_none:
                lower_corner[i] = interval[0]
            if upper_is_none:
                upper_corner[i] = interval[-1]

    return lower_corner, upper_corner


def get_crop_item_from_points(*world_points_values, wcs, data_shape):
    """
    Find slice item that crops to minimum cube in array-space containing specified world points.

    Parameters
    ----------
    world_points_values
        The world coordinates in wcsapi "values" form (i.e. arrays /
        floats), for however many world points should be contained in the
        output cube. Each argument should be a tuple with number of
        coordinates equal to the number of world axes.

    wcs : `~astropy.wcs.wcsapi.BaseHighLevelWCS`, `~astropy.wcs.wcsapi.BaseLowLevelWCS`
        The WCS to use to convert the world coordinates to array indices.

    data_shape : `tuple` of `int`
        The shape of the cube in array order.

    Returns
    -------
    item : `tuple` of `slice`
        The slice item for each axis of the cube which, when applied to the cube,
        will return the minimum cube in array-index-space that contains all the
        input world points.
    """
    if isinstance(wcs, BaseHighLevelWCS):
        wcs = wcs.low_level_wcs

    # Convert all points to array indices.
    point_indices = []
    for point in world_points_values:
        indices = wcs.world_to_array_index_values(*point)

        if not isinstance(indices, tuple):
            indices = (indices,)

        point_indices.append(indices)

    point_indices = np.array(point_indices)
    lower = np.min(point_indices, axis=0)
    upper = np.max(point_indices, axis=0) + 1

    # Wrap the limits to the size of the array
    lower = [int(np.clip(index, 0, data_shape[i])) for i, index in enumerate(lower)]
    upper = [int(np.clip(index, 0, data_shape[i])) for i, index in enumerate(upper)]

    return tuple(slice(start, stop) for start, stop in zip(lower, upper))
