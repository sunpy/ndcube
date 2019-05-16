# -*- coding: utf-8 -*-

"""
Utilities for ndcube.
"""

import numpy as np

__all__ = ['wcs_axis_to_data_axis', 'data_axis_to_wcs_axis', 'select_order',
           'convert_extra_coords_dict_to_input_format', 'get_axis_number_from_axis_name']


def data_axis_to_wcs_axis(data_axis, missing_axes):
    """Converts a data axis number to the corresponding wcs axis number."""
    if data_axis is None:
        result = None
    else:
        if data_axis < 0:
            data_axis = np.invert(missing_axes).sum() + data_axis
        if data_axis > np.invert(missing_axes).sum()-1 or data_axis < 0:
            raise IndexError("Data axis out of range.  Number data axes = {0}".format(
                np.invert(missing_axes).sum()))
        result = len(missing_axes)-np.where(np.cumsum(
            [b is False for b in missing_axes][::-1]) == data_axis+1)[0][0]-1
    return result


def wcs_axis_to_data_axis(wcs_axis, missing_axes):
    """Converts a wcs axis number to the corresponding data axis number."""
    if wcs_axis is None:
        result = None
    else:
        if wcs_axis < 0:
            wcs_axis = len(missing_axes) + wcs_axis
        if wcs_axis > len(missing_axes)-1 or wcs_axis < 0:
            raise IndexError("WCS axis out of range.  Number WCS axes = {0}".format(
                len(missing_axes)))
        if missing_axes[wcs_axis]:
            result = None
        else:
            data_ordered_wcs_axis = len(missing_axes)-wcs_axis-1
            result = data_ordered_wcs_axis-sum(missing_axes[::-1][:data_ordered_wcs_axis])
    return result


def select_order(axtypes):
    """
    Returns indices of the correct data order axis priority given a list of WCS CTYPEs.

    For example, given ['HPLN-TAN', 'TIME', 'WAVE'] it will return
    [1, 2, 0] because index 1 (time) has the lowest priority, followed by
    wavelength and finally solar-x.

    Parameters
    ----------
    axtypes: str list
        The list of CTYPEs to be modified.

    """
    order = [(0, t) if t in ['TIME', 'UTC'] else
             (1, t) if t == 'WAVE' else
             (2, t) if t == 'HPLT-TAN' else
             (axtypes.index(t) + 3, t) for t in axtypes]
    order.sort()
    result = [axtypes.index(s) for (_, s) in order]
    return result


def _format_input_extra_coords_to_extra_coords_wcs_axis(extra_coords, missing_axes,
                                                        data_shape):
    extra_coords_wcs_axis = {}
    coord_format_error = ("Coord must have three properties supplied, "
                          "name (str), axis (int), values (Quantity or array-like)."
                          " Input coord: {0}")
    coord_0_format_error = ("1st element of extra coordinate tuple must be a "
                            "string giving the coordinate's name.")
    coord_1_format_error = ("2nd element of extra coordinate tuple must be None "
                            "or an int giving the data axis "
                            "to which the coordinate corresponds.")
    coord_len_error = ("extra coord ({0}) must have same length as data axis "
                       "to which it is assigned: coord length, {1} != data axis length, {2}")
    for coord in extra_coords:
        # Check extra coord has the right number and types of info.
        if len(coord) != 3:
            raise ValueError(coord_format_error.format(coord))
        if not isinstance(coord[0], str):
            raise ValueError(coord_0_format_error.format(coord))
        if coord[1] is not None and not isinstance(coord[1], int) and \
                not isinstance(coord[1], np.int64):
            raise ValueError(coord_1_format_error)
        # Unless extra coord corresponds to a missing axis, check length
        # of coord is same is data axis to which is corresponds.
        if coord[1] is not None:
            if not missing_axes[::-1][coord[1]]:

                if len(coord[2]) != data_shape[coord[1]]:
                    raise ValueError(coord_len_error.format(coord[0], len(coord[2]),
                                                            data_shape[coord[1]]))
        # Determine wcs axis corresponding to data axis of coord
        extra_coords_wcs_axis[coord[0]] = {
            "wcs axis": data_axis_to_wcs_axis(coord[1], missing_axes),
            "value": coord[2]}
    return extra_coords_wcs_axis


def convert_extra_coords_dict_to_input_format(extra_coords, missing_axes):
        """
        Converts NDCube.extra_coords attribute to format required as input for new NDCube.

        Parameters
        ----------
        extra_coords: dict
            An NDCube.extra_coords instance.

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
                axis = wcs_axis_to_data_axis(extra_coords[name]["wcs axis"], missing_axes)
            elif "axis" in coord_keys and "wcs axis" not in coord_keys:
                axis = extra_coords[name]["axis"]
            else:
                raise KeyError("extra coords dict can have keys 'wcs axis' or 'axis'.  Not both.")
            result.append((name, axis, extra_coords[name]["value"]))
        return result


def get_axis_number_from_axis_name(axis_name, world_axis_physical_types):
    """
    Returns axis number (numpy ordering) given a substring unique to a world axis type string.

    Parameters
    ----------
    axis_name: `str`
        Name or substring of name of axis as defined by NDCube.world_axis_physical_types

    world_axis_physical_types: iterable of `str`
        Output from NDCube.world_axis_physical_types for relevant cube,
        i.e. iterable of string axis names.

    Returns
    -------
    axis_index[0]: `int`
        Axis number (numpy ordering) corresponding to axis name
    """
    axis_index = [axis_name in world_axis_type for world_axis_type in world_axis_physical_types]
    axis_index = np.arange(len(world_axis_physical_types))[axis_index]
    if len(axis_index) != 1:
        raise ValueError("User defined axis with a string that is not unique to "
                         "a physical axis type. {0} not in any of {1}".format(
                             axis_name, world_axis_physical_types))
    return axis_index[0]
