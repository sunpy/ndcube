# -*- coding: utf-8 -*-

"""
Utilities for ndcube.
"""

import copy

import numpy as np
import astropy.units as u

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


def convert_extra_coords_dict_to_input_format(extra_coords, missing_axis):
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
                axis = wcs_axis_to_data_axis(extra_coords[name]["wcs axis"], missing_axis)
            elif "axis" in coord_keys and "wcs axis" not in coord_keys:
                axis = extra_coords[name]["axis"]
            else:
                raise KeyError("extra coords dict can have keys 'wcs axis' or 'axis'.  Not both.")
            result.append((name, axis, extra_coords[name]["value"]))
        return result


def _get_pixel_quantities_for_dependent_axes(dependent_axes, cube_dimensions):
    """
    Returns list of othogonal pixel quantities for cube axes which have dependent WCS translations.

    Parameters
    ----------
    dependent_axis: `list` of `int`
        Axis numbers in the numpy convention.

    cube_dimensions: Iterable of `int`
        Number of pixels in each dimension

    Returns
    -------
    quantity_list: `list` of `astropy.units.Quantity`
        Othogonal pixel quantities describing all pixels in cube in relevant dimensions.

    """

    print(dependent_axes)
    n_dependent_axes = len(dependent_axes)
    n_dimensions = len(cube_dimensions)
    quantity_list = [u.Quantity(np.zeros(tuple(cube_dimensions[dependent_axes])),
                                unit=u.pix)] * n_dimensions
    for i, dependent_axis in enumerate(dependent_axes):
        # Define list of indices/slice objects for accessing
        # sections of dependent axis arrays that are to be
        # replaced with orthogonal np.arange arrays.
        slice_list = [0] * n_dependent_axes
        slice_list[i] = slice(None)
        # Define array of indices of axes in slice list not
        # including the axis along which we are inserting
        # np.arange.
        other_axes_indices = list(range(n_dependent_axes))
        del(other_axes_indices[i])
        other_axes_indices = np.asarray(other_axes_indices)
        other_axes_products = np.cumprod(cube_dimensions[other_axes_indices][::-1])[::-1]
        other_axes_products = np.append(other_axes_products, 1)[1:]
        # Determine total number of times we are going to
        # insert np.arange into quantity array.
        total_iters = np.prod(cube_dimensions[other_axes_indices])
        coord_axis_array = copy.deepcopy(quantity_list[dependent_axis].value)
        for k in range(total_iters):
            # Determine mapping from iteration int to
            # indices to insert to slice_list.
            mod = k
            l = 0
            # Use while loop so that l will always equal
            # n_dependent_axes[dependent_axis]-2 when loop is exited.
            while l < n_dependent_axes-2:
                slice_list[other_axes_indices[l]] = int(mod/other_axes_products[l])
                mod = mod % other_axes_products[l]
                l += 1
            slice_list[other_axes_indices[l]] = mod
            coord_axis_array[tuple(slice_list)] = np.arange(cube_dimensions[dependent_axis])
        # Replace pixel array for this axis in quantity_list.
        quantity_list[dependent_axis] = u.Quantity(coord_axis_array, unit=u.pix)
    return quantity_list
