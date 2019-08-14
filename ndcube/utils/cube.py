# -*- coding: utf-8 -*-

"""
Utilities for ndcube.
"""

import numpy as np
from astropy.units import Quantity

from ndcube.utils.wcs import _pixel_keep, get_dependent_wcs_axes

__all__ = ['wcs_axis_to_data_axis', 'data_axis_to_wcs_axis', 'select_order','_pixel_centers_or_edges','_get_dimension_for_pixel',
           'convert_extra_coords_dict_to_input_format', 'get_axis_number_from_axis_name','wcs_axis_to_data_ape14','unique_data_axis']



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
        result = len(missing_axes)-np.where(np.cumsum([b is False for b in missing_axes][::-1]) == data_axis+1)[0][0]-1
    return result


def data_axis_to_wcs_ape14(data_axis, pixel_keep, naxes, old_order=False):
    """Converts a data axis number to wcs axis number taking care of the missing axes"""

    # old_order tells us whether data_axis is an axis of before
    # slicing or after slicing
    # old_order=True tells us that data_axis is an axis before slicing

    # Make sure that data_axis is a scalar item
    if data_axis is not None:
        if not isinstance(data_axis, (int, np.int32, np.int64)):
            raise ValueError(f"The data_axis parameter accepts \
                numpy.int64 or numpy.np.int32 datatype, got this {type(data_axis)}")

    # Make sure _pixel_keep is numpy array
    if not isinstance(pixel_keep, np.ndarray):
        raise TypeError(f"The pixel_keep parameter should be np.ndarray, got this {type(pixel_keep)}.")

    # Sanitize the data_axis
    if data_axis is None:
        return None
    else:
        if data_axis < 0:
            data_axis += naxes
        if data_axis > naxes -1 or data_axis < 0:
            raise IndexError("Data axis out of range.  Number Data axes = {0} and the value requested is {1}".format(
                naxes, data_axis))
    if not old_order:
        return naxes - 1 - data_axis
    else:
        # pixel_keep is the old order of all wcs
        # Get the old order of all data axes
        old_data_order = naxes - 1 - pixel_keep

        # Get a mapping of the old order and new order of all data axes
        new_wcs_order = np.unique(pixel_keep, return_inverse=True)[1]

        # Mapping of the order of new wcs axes
        new_data_order = new_wcs_order[::-1]

        # First we check if the data_axis whose wcs_axis we want to calculate
        # is present in the old_data_order
        index = np.where(data_axis == old_data_order)[0]
        if index.size != 0:
            index = index.item()
        else:
            index = None

        if index is None:
            # As we have performed the check for bound,
            # so the data_axis must have been missing if
            # index is None
            return None

        # Return the corresponding wcs_axis for the data axis
        return new_wcs_order[index]


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


def wcs_axis_to_data_ape14(wcs_axis, pixel_keep, naxes, old_order=False):
    """Converts a wcs axis number to data axis number taking care of the missing axes"""

    # old_order tells us whether wcs_axis is an axis of before
    # slicing or after slicing
    # old_order=True tells us that wcs_axis is an axis before slicing

    # Make sure that wcs_axis is a scalar item
    if wcs_axis is not None:
        if not isinstance(wcs_axis,(int, np.int32, np.int64)):
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
        if wcs_axis > naxes -1 or wcs_axis < 0:
            raise IndexError("WCS axis out of range.  Number WCS axes = {0} and the value requested is {1}".format(
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


def _format_input_extra_coords_to_extra_coords_wcs_axis(extra_coords, pixel_keep, naxes,
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

        # Determine wcs axis corresponding to data axis of coord

        extra_coords_wcs_axis[coord[0]] = {
            "wcs axis": data_axis_to_wcs_ape14(coord[1], pixel_keep, naxes),
            "value": coord[2]}
    return extra_coords_wcs_axis


def convert_extra_coords_dict_to_input_format(extra_coords, pixel_keep, naxes):
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
                axis = wcs_axis_to_data_ape14(extra_coords[name]["wcs axis"], pixel_keep, naxes, old_order=True)
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

def _pixel_centers_or_edges(axis_length, edges):
    """
    Returns a range of pixel_values or pixel_edges

    Parameters
    ----------
    axis_length: `int`
        The length of the axis

    edges: `bool`
        Boolean to signify whether pixel_edge or pixel_value requested
        False stands for pixel_value, while True stands for pixel_edge

    Returns
    -------
    `np.ndarray`
        The axis_values for the given input
    """
    if edges is False:
        axis_values = np.arange(axis_length)
    else:
        axis_values = np.arange(-0.5, axis_length+0.5)
    return axis_values

def _get_dimension_for_pixel(axis_length, edges):
    """
    Returns the dimensions for the given edges.

    Parameters
    ----------
    axis_length : `int`
        The length of the axis
    edges : `bool`
        Boolean to signify whether pixel_edge or pixel_value requested
        False stands for pixel_value, while True stands for pixel_edge
    """
    return axis_length+1 if edges else axis_length

def ape14_axes(wcs_object, input_axis):
    """Returns the corresponding wcs axes after a wcs object
    is sliced. The `_pixel_keep` attribute of wcs tells us
    which axis is present, so returns the corresponding wcs
    axes after slicing.

    Parameters
    ----------
    wcs_object : `astropy.wcs.WCS` or similar object
        The WCS object
    input_axis : `int` or `list`
        The list of wcs axes

    Returns
    -------
    `int` or `list`
        The corresponding wcs axes of the sliced wcs object.
    """
    wcomp = wcs_object.world_axis_object_components
    axis_type = np.array([item[0] for item in wcomp])
    axis_type = axis_type[::-1]

    ape14_axes = np.unique(axis_type, return_inverse=True)[1]

    n_rep_ape14_axes = np.unique(ape14_axes[input_axis])

    return n_rep_ape14_axes[::-1]


def unique_data_axis(wcs_object, input_axis):
    """This function helps in returning the corresponding data axis
    after the assigning same data axis to a list of given dependent axis.

    Parameters
    ----------
    wcs_object : `astropy.wcs.WCS` or similar object
        The WCS object
    input_axis : `int` or `list`
        The list of wcs axes

    Examples
    --------
    Suppose, we have a wcs object with such entries:
    Below here is a Numpy ordering
    ['lat','lon','time','wave']
    then the corresponding data entries after adjusting the axis of
    dependent axis as same :
    np.array([0, 0, 1, 2]).

    As the lat and lon are dependent axes, so they get assigned the same data axis.
    """

    wcomp = wcs_object.world_axis_object_components
    axis_type = np.array([item[0] for item in wcomp])

    # Numpy ordering
    axis_type = axis_type[::-1]

    distinct_element = list()
    distinct_element_index = list()

    # Pointer to each non-dependent axis
    # gets incremented after each non-dependent axis
    idx = 0
    prev_el = axis_type[0]
    for element in axis_type:
        if element != prev_el:
            idx += 1
        distinct_element_index.append(idx)
        if element not in distinct_element:
            distinct_element.append(element)

        prev_el = element

    # Convert to numpy array for array indexing
    distinct_element_index = np.array(distinct_element_index)

    if(wcs_object.pixel_n_dim == wcs_object.world_n_dim):
        # If the pixel_dim and world_dim are same, then return the
        # distinct_element_index as it is
        return distinct_element_index[input_axis], np.unique(distinct_element_index)
    else:
        sliced_axis = np.setdiff1d(wcs_object._world_keep, wcs_object._pixel_keep)
        index_sliced_axis = np.where(sliced_axis == wcs_object._world_keep)[0]
        distinct_element_index = np.delete(distinct_element_index[::-1], index_sliced_axis[0])[::-1]

        # Return the corresponding axis/axes after denoting same axis to dependent axis
        return distinct_element_index[input_axis], np.unique(distinct_element_index)

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
            edges = np.zeros(shape+1) * value.unit
        else:
            edges = np.zeros(shape+1)

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
        edges[...,1:-1] = value[...,:-1] + (value[...,1:] - value[...,:-1]) / 2
        edges[...,0] = value[...,0] - (value[...,1] - value[...,0]) / 2
        edges[...,-1] = value[...,-1] + (value[...,-1] - value[...,-2]) / 2

        # Revert the shape of the edges array
        edges = np.moveaxis(edges, -1, axis)
    return edges
