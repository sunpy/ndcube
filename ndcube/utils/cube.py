
"""
Utilities for ndcube.
"""
import numbers

import numpy as np
from astropy.units import Quantity

import ndcube.utils.wcs as wcs_utils

__all__ = [
    'convert_extra_coords_dict_to_input_format',
    'wcs_axis_to_data_ape14']

# Deprecated in favor of utils.wcs.convert_between_array_and_pixel_axes.
# Can only remove after extra_coords refactor.
def wcs_axis_to_data_ape14(wcs_axis, pixel_keep, naxes, old_order=False):
    """Converts a wcs axis number to data axis number taking care of the missing axes"""

    # old_order tells us whether wcs_axis is an axis of before
    # slicing or after slicing
    # old_order=True tells us that wcs_axis is an axis before slicing

    # Make sure that wcs_axis is a scalar item
    if wcs_axis is not None:
        if not isinstance(wcs_axis, numbers.Integral):
            raise ValueError(f"The wcs_axis parameter accepts \
                numpy.int64 or np.int32 datatype, got this {type(wcs_axis)}")

    # Make sure _pixel_keep is numpy array
    if not isinstance(pixel_keep, np.ndarray):
        raise TypeError(f"The pixel_keep parameter should be np.ndarray, got this {type(pixel_keep)}.")

    # Sanitize the wcs_axis
    if wcs_axis is None:
        return None
    else:
        if wcs_axis < 0:
            wcs_axis += naxes
        if wcs_axis > naxes - 1 or wcs_axis < 0:
            raise IndexError(
                "WCS axis out of range.  Number WCS axes = {0} and the value requested is {1}".format(
                    naxes, wcs_axis))

    if not old_order:
        return naxes - 1 - wcs_axis
    else:
        # Try to convert the wcs_axis to its corresponding data axis if present
        # If not present, return None

        # pixel_keep is the old order of all wcs axes
        # Get the old order of all data axes
        old_data_order = naxes - 1 - pixel_keep

        # Get a mapping of the old order and new order of all data axes
        new_wcs_order = np.unique(pixel_keep, return_inverse=True)[1]

        # Mapping of the order of new wcs_axes
        new_data_order = new_wcs_order[::-1]

        # First we check if the wcs axis whose data_axis we want to calculate
        # is present in the old_wcs_order

        index = np.where(wcs_axis == pixel_keep)[0]
        if index.size != 0:
            index = index.item()
        else:
            index = None
        if index is None:
            # As we have performed the check for bound,
            # so the wcs_axis must have been missing if
            # index is None
            return None

        # Return the corresponding data_axis for the wcs_axis
        return new_data_order[index]


def _format_input_extra_coords_to_extra_coords_wcs_axis(extra_coords, pixel_keep, naxes,
                                                        data_shape):
    extra_coords_wcs_axis = {}
    coord_format_error = ("Coord must have three properties supplied, "
                          "name (str), axis (int), values (Quantity or array-like)."
                          " Input coord: {0}")
    coord_0_format_error = ("1st element of extra coordinate tuple must be a "
                            "string giving the coordinate's name.")
    coord_1_format_error = ("2nd element of extra coordinate tuple must be None "
                            "or an int or tuple of int giving the data axis "
                            "to which the coordinate corresponds.")
    coord_len_error = ("extra coord ({0}) must have same length as data axis "
                       "to which it is assigned: coord length, {1} != data axis length, {2}")
    for coord in extra_coords:
        # Check extra coord has the right number and types of info.
        if len(coord) != 3:
            raise ValueError(coord_format_error.format(coord))
        if not isinstance(coord[0], str):
            raise ValueError(coord_0_format_error.format(coord))
        # Check coord axis number is valid and convert to a WCS-order axis number.
        if coord[1] is None:
            wcs_coord_axis = None
        else:
            if isinstance(coord[1], numbers.Integral):
                wcs_coord_axis = wcs_utils.convert_between_array_and_pixel_axes(np.array([coord[1]]), naxes)[0]
            elif hasattr(coord[1], "__len__") and all([isinstance(c, numbers.Integral) for c in coord[1]]):
                wcs_coord_axis = tuple(wcs_utils.convert_between_array_and_pixel_axes(np.array(coord[1]), naxes))
            else:
                raise ValueError(coord_1_format_error)

        extra_coords_wcs_axis[coord[0]] = {"wcs axis": wcs_coord_axis, "value": coord[2]}
    return extra_coords_wcs_axis


def convert_extra_coords_dict_to_input_format(extra_coords, pixel_keep, naxes):
    """
    Converts NDCube.extra_coords attribute to format required as input for new NDCube.

    Parameters
    ----------
    extra_coords: `dict`
        An NDCube.extra_coords instance.

    pixel_keep: `list`
        The pixel dimensions of the original WCS to keep.

    naxes: `int`
        The number of axes in the original WCS.

    Returns
    -------
    input_format: `list`
        Infomation on extra coords in format required by `NDCube.__init__`.

    """
    coord_names = list(extra_coords.keys())
    result = []
    for name in coord_names:

        coord_keys = list(extra_coords[name].keys())
        if "wcs axis" in coord_keys and "axis" not in coord_keys:
            axis = wcs_axis_to_data_ape14(extra_coords[name]["wcs axis"], pixel_keep, naxes, old_order=True)
        elif "axis" in coord_keys and "wcs axis" not in coord_keys:
            axis = extra_coords[name]["axis"]
        else:
            raise KeyError("extra coords dict can have keys 'wcs axis' or 'axis'.  Not both.")
        result.append((name, axis, extra_coords[name]["value"]))
    return result


def _get_extra_coord_edges(value, axis=-1):
    """Gets the pixel_edges from the pixel_values
     Parameters
    ----------
    value : `astropy.units.Quantity` or array-like
        The Quantity object containing the values for a given `extra_coords`
     axis : `int`
        The axis about which pixel_edges needs to be calculated
        Default value is -1, which is the last axis for a ndarray
    """

    # Checks for corner cases

    if not isinstance(value, np.ndarray):
        value = np.array(value)

    # Get the shape of the Quantity object
    shape = value.shape
    if len(shape) == 1:

        shape = len(value)
        if isinstance(value, Quantity):
            edges = np.zeros(shape + 1) * value.unit
        else:
            edges = np.zeros(shape + 1)

        # Calculate the pixel_edges from the given pixel_values
        edges[1:-1] = value[:-1] + (value[1:] - value[:-1]) / 2
        edges[0] = value[0] - (value[1] - value[0]) / 2
        edges[-1] = value[-1] + (value[-1] - value[-2]) / 2

    else:
        # Edit the shape of the new ndarray to increase the length
        # by one for a given axis
        shape = list(shape)
        shape[axis] += 1
        shape = tuple(shape)

        if isinstance(value, Quantity):
            edges = np.zeros(shape) * value.unit
        else:
            edges = np.zeros(shape)
        # Shift the axis which is point of interest to last axis
        value = np.moveaxis(value, axis, -1)
        edges = np.moveaxis(edges, axis, -1)

        # Calculate the pixel_edges from the given pixel_values
        edges[..., 1:-1] = value[..., :-1] + (value[..., 1:] - value[..., :-1]) / 2
        edges[..., 0] = value[..., 0] - (value[..., 1] - value[..., 0]) / 2
        edges[..., -1] = value[..., -1] + (value[..., -1] - value[..., -2]) / 2

        # Revert the shape of the edges array
        edges = np.moveaxis(edges, -1, axis)
    return edges
