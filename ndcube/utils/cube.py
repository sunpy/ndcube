import inspect
from functools import wraps

import astropy.units as u
import numpy as np
from astropy.wcs.wcsapi import BaseHighLevelWCS, HighLevelWCSWrapper

from ndcube.utils import misc as misc_utils
from ndcube.utils.wcs_high_level_conversion import high_level_objects_to_values


def sanitise_wcs(func):
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

    points = bounding_box_to_corners(lower_corner_values, upper_corner_values,
                                     wcs.axis_correlation_matrix)
    return get_crop_item_from_points(*points, wcs=wcs, data_shape=data_shape)


def get_crop_by_values_item(lower_corner, upper_corner, wcs, data_shape, units=None):
    # Sanitize inputs.
    no_op, lower_corner, upper_corner, wcs = sanitize_crop_inputs(lower_corner, upper_corner, wcs)
    # Quit out early if we are no-op
    if no_op:
        return tuple([slice(None)] * len(data_shape))

    n_coords = len(lower_corner)
    if units is None:
        units = [None] * n_coords
    elif len(units) != n_coords:
        raise ValueError("units must be None or have same length as corner inputs.")

    # Convert float inputs to quantities using units.
    types_with_units = (u.Quantity, type(None))
    for i, (lower, upper, unit) in enumerate(zip(lower_corner, upper_corner, units)):
        lower_is_float = not isinstance(lower, types_with_units)
        upper_is_float = not isinstance(upper, types_with_units)
        if unit is None and (lower_is_float or upper_is_float):
            raise TypeError("If corner value is not a Quantity or None, "
                            "unit must be a valid astropy Unit or unit string."
                            f"index: {i}; lower type: {type(lower)}; "
                            f"upper type: {type(upper)}; unit: {unit}")
        if lower_is_float:
            lower_corner[i] = u.Quantity(lower, unit=unit)
        if upper_is_float:
            upper_corner[i] = u.Quantity(upper, unit=unit)
        # Convert each corner value to the same unit.
        if lower_corner[i] is not None and upper_corner[i] is not None:
            upper_corner[i] = upper_corner[i].to(lower_corner[i].unit)

    lower_corner, upper_corner = fill_in_crop_nones(lower_corner, upper_corner,
                                                    wcs, data_shape, True)

    # Convert coordinates to units used by WCS as WCS.world_to_array_index
    # does not handle quantities.
    lower_corner = misc_utils.convert_quantities_to_units(lower_corner, wcs.world_axis_units)
    upper_corner = misc_utils.convert_quantities_to_units(upper_corner, wcs.world_axis_units)

    points = bounding_box_to_corners(lower_corner, upper_corner, wcs.axis_correlation_matrix)
    return get_crop_item_from_points(*points, wcs=wcs, data_shape=data_shape)


def sanitize_crop_inputs(lower_corner, upper_corner, wcs):
    """Sanitize inputs to NDCube crop methods.

    First arg returned signifies whether the inputs imply that cropping
    should be performed or not.
    """
    lower_corner, upper_corner = sanitize_corners(lower_corner, upper_corner)

    # Quit out early if we are no-op
    lower_nones = np.array([lower is None for lower in lower_corner])
    upper_nones = np.array([upper is None for upper in upper_corner])
    if (lower_nones & upper_nones).all():
        return True, lower_corner, upper_corner, wcs

    # Import must be here to avoid circular import.
    from ndcube.extra_coords import ExtraCoords
    if isinstance(wcs, ExtraCoords):
        # Add None inputs to upper and lower corners for new dummy axes.
        n_dummy_axes = len(wcs._cube_array_axes_without_extra_coords)
        lower_corner += [None] * n_dummy_axes
        upper_corner += [None] * n_dummy_axes
        # Convert extra coords to WCS describing whole cube.
        wcs = wcs.cube_wcs

    return False, lower_corner, upper_corner, wcs


def sanitize_corners(*corners):
    """Sanitize corner inputs to NDCube crop methods."""
    corners = [list(corner) if isinstance(corner, (tuple, list)) else [corner]
               for corner in corners]
    n_coords = [len(corner) for corner in corners]
    if len(set(n_coords)) != 1:
        raise ValueError("All corner inputs must have same number of coordinate objects. "
                         f"Lengths of corner objects: {n_coords}")
    return corners


def fill_in_crop_nones(lower_corner, upper_corner, wcs, data_shape, crop_by_values):
    """
    Replace any instance of None in the inputs with the bounds for that axis.
    """
    lower_nones = np.array([lower is None for lower in lower_corner])
    upper_nones = np.array([upper is None for upper in upper_corner])

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


def bounding_box_to_corners(lower_corner_values, upper_corner_values, axis_correlation_matrix):
    """
    Convert two corners of a bounding box to the points of all corners.
    """
    # Calculate which world axes share multiple pixel axes.
    world_n_dep_dim = axis_correlation_matrix.sum(axis=1)
    # If all world coordinates are independent, bounding box is simple.
    max_dep_dim = world_n_dep_dim.max()
    if max_dep_dim < 2:
        return (tuple(lower_corner_values), tuple(upper_corner_values))

    # Otherwise we need calculate the corners more carefully.
    # This must be done based on correlation matrix as not all
    # world axes are independent.
    # Start by generating array of world corner values assuming
    # all coordinates are 1D. Strip units and add them back on later.
    corner_units = [None] * len(lower_corner_values)
    for i, (lcv, ucv) in enumerate(zip(lower_corner_values, upper_corner_values)):
        if not isinstance(ucv, type(lcv)):
            raise TypeError("Corresponding entries in lower and upper corner values "
                            "must be of same type.")
        if isinstance(lcv, u.Quantity):
            corner_units[i] = lcv.unit
            lower_corner_values[i] = lower_corner_values[i].value
            upper_corner_values[i] = ucv.to_value(corner_units[i])
    lower_corner_values = np.asarray(lower_corner_values)
    upper_corner_values = np.asarray(upper_corner_values)
    corners = np.stack([lower_corner_values, upper_corner_values])

    # Next, calculate the sets of pixel axes upon which each world axis depends.
    world_n_dim, pixel_n_dim = axis_correlation_matrix.shape
    world_axes = np.arange(world_n_dim)
    dep_pix_axes = np.array([set(np.arange(pixel_n_dim)[axis_correlation_matrix[j]])
                             for j in world_axes], dtype=object)

    # Iterate through number of shared pixel axes, i, and create additional corners.
    # Iterate from highest to lowest number of shared axes to reduce duplication.
    # Do not include world axes sharing only 1 pixel axis, as we have already
    # captured those in our initial definition of corners, above.
    for i in range(max_dep_dim, 1, -1):
        # Extract the world axes whose number of shared pixel axes is i.
        world_axes_idx = np.where(world_n_dep_dim == i)[0]
        world_axes_i = world_axes[world_axes_idx]
        # Determine which specific pixel axes each world axis shares.
        dep_pix_axes_i = dep_pix_axes[world_axes_idx]
        # Iterate through world axes with same shared pixel axes
        # and create required corners.
        while len(world_axes_i) > 0:
            dpa = dep_pix_axes_i[0]
            # To avoid duplication in future iterations of the top-level loop,
            # remove world axes from top-level list that share the same (or a subset of)
            # pixel axes as the current world axis.
            # Iterate backwards to delete last items first so as not to change index
            # of items yet to be deleted.
            n = len(dep_pix_axes) - 1  # last element index for converting iterator to index.
            for j, axes in enumerate(dep_pix_axes[::-1]):
                if axes.issubset(dpa):
                    world_axes = np.delete(world_axes, n-j)
                    world_n_dep_dim = np.delete(world_n_dep_dim, n-j)
                    dep_pix_axes = np.delete(dep_pix_axes, n-j)
            # Collect the world axes which share the same set of pixel axes.
            # To avoid duplication in future iterations of this loop,
            # remove those world axes from current-level list.
            # Iterate backwards to delete last items first so as not to change index
            # of items yet to be deleted.
            dep_world_axes_i = []
            n = len(dep_pix_axes_i) - 1  # last element index for converting iterator to index.
            for j, axes in enumerate(dep_pix_axes_i[::-1]):
                if axes == dpa:
                    dep_world_axes_i.append(world_axes_i[n-j])
                    world_axes_i = np.delete(world_axes_i, n-j)
                    dep_pix_axes_i = np.delete(dep_pix_axes_i, n-j)
            # We started this function by defining the corners for 1D axes.
            # This means that to fully describe the corners, the number of world axes
            # we need is the number of shared pixel axes minus 1.
            # If there are more, discard the excess.
            ii = i - 1
            dep_world_axes_i = np.array(dep_world_axes_i)[:ii]
            # For i number of shared pixel coords, we need 2^i corners.
            # Add the required number of corners by duplicating current one
            # then edit them so they contain the correct world values.
            # First double the number of corners.
            corners = np.repeat(corners[np.newaxis], 2, axis=0)
            # Then multiple the number of corners of ii (i-1) since we
            # started off by defining two corners.
            corners = np.repeat(corners[np.newaxis], ii, axis=0)
            # Fill in the correct world values for the new corners.
            for j in range(ii):
                corners[j, 0, :, dep_world_axes_i] = lower_corner_values[dep_world_axes_i]
                corners[j, 1, :, dep_world_axes_i] = upper_corner_values[dep_world_axes_i]
            # Reshape corners array so it will be predictable for next iteration.
            corners = corners.reshape((np.prod(corners.shape[:3]), corners.shape[-1]))

    # Reattach units if were present in input.
    corners = tuple(tuple(corner * corner_unit if corner_unit else corner
                          for corner, corner_unit in zip(corners[i], corner_units))
                    for i in range(corners.shape[0]))
    return corners


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
