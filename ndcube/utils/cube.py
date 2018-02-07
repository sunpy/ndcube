# -*- coding: utf-8 -*-

"""
Utilities for ndcube.
"""

import numpy as np


__all__ = ['wcs_axis_to_data_axis', 'data_axis_to_wcs_axis', 'select_order']


def data_axis_to_wcs_axis(data_axis, missing_axis):
    """Converts a data axis number to the corresponding wcs axis number."""
    if data_axis is None:
        result = None
    else:
        result = len(missing_axis)-np.where(np.cumsum(
            [b is False for b in missing_axis][::-1]) == data_axis+1)[0][0]-1
    return result


def wcs_axis_to_data_axis(wcs_axis, missing_axis):
    """Converts a wcs axis number to the corresponding data axis number."""
    if wcs_axis is None:
        result = None
    else:
        if missing_axis[wcs_axis]:
            result = None
        else:
            data_ordered_wcs_axis = len(missing_axis)-wcs_axis-1
            result = data_ordered_wcs_axis-sum(missing_axis[::-1][:data_ordered_wcs_axis])
    return result


def select_order(axtypes):
    """
    Returns the indices of the correct axis priority for the given list of WCS
    CTYPEs. For example, given ['HPLN-TAN', 'TIME', 'WAVE'] it will return
    [1, 2, 0] because index 1 (time) has the highest priority, followed by
    wavelength and finally solar-x. When two or more celestial axes are in the
    list, order is preserved between them (i.e. only TIME, UTC and WAVE are
    moved)

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


def _format_input_extra_coords_to_extra_coords_wcs_axis(extra_coords, missing_axis,
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
            if not missing_axis[::-1][coord[1]]:

                if len(coord[2]) != data_shape[coord[1]]:
                    raise ValueError(coord_len_error.format(coord[0], len(coord[2]),
                                                            data_shape[coord[1]]))
        # Determine wcs axis corresponding to data axis of coord
        extra_coords_wcs_axis[coord[0]] = {
            "wcs axis": data_axis_to_wcs_axis(coord[1], missing_axis),
            "value": coord[2]}
    return extra_coords_wcs_axis
