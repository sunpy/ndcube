# Author: Ankit Baruah and Daniel Ryan <ryand5@tcd.ie>

"""
Miscellaneous WCS utilities.
"""

import numbers
from collections import UserDict

import numpy as np
from astropy.wcs.utils import pixel_to_pixel
from astropy.wcs.wcsapi import BaseHighLevelWCS, BaseLowLevelWCS, low_level_api

__all__ = ['array_indices_for_world_objects', 'convert_between_array_and_pixel_axes',
           'calculate_world_indices_from_axes', 'wcs_ivoa_mapping',
           'pixel_axis_to_world_axes', 'world_axis_to_pixel_axes',
           'pixel_axis_to_physical_types', 'physical_type_to_pixel_axes',
           'physical_type_to_world_axis', 'get_dependent_pixel_axes',
           'get_dependent_array_axes', 'get_dependent_world_axes',
           'get_dependent_physical_types', 'array_indices_for_world_objects',
           'validate_physical_types']


class TwoWayDict(UserDict):
    @property
    def inv(self):
        """
        The inverse dictionary.
        """
        return {v: k for k, v in self.items()}


# Define a two way dictionary to hold translations between WCS axis
# types and International Virtual Observatory Alliance vocabulary.
# See http://www.ivoa.net/documents/REC/UCD/UCDlist-20070402.html
wcs_to_ivoa = {
    "HPLT": "custom:pos.helioprojective.lat",
    "HPLN": "custom:pos.helioprojective.lon",
    "TIME": "time",
    "WAVE": "em.wl",
    "RA--": "pos.eq.ra",
    "DEC-": "pos.eq.dec",
    "FREQ": "em.freq",
    "STOKES": "phys.polarization.stokes",
    "PIXEL": "instr.pixel",
    "XPIXEL": "custom:instr.pixel.x",
    "YPIXEL": "custom:instr.pixel.y",
    "ZPIXEL": "custom:instr.pixel.z",
    "HECR": "custom:pos.heliographic.distance",
    "HECH": "pos.bodyrc.alt",
}
wcs_ivoa_mapping = TwoWayDict()
for key in wcs_to_ivoa.keys():
    wcs_ivoa_mapping[key] = wcs_to_ivoa[key]


def convert_between_array_and_pixel_axes(axis, naxes):
    """Reflects axis index about center of number of axes.

    This is used to convert between array axes in numpy order and pixel axes in WCS order.
    Works in both directions.

    Parameters
    ----------
    axis: `numpy.ndarray` of `int`
        The axis number(s) before reflection.

    naxes: `int`
        The number of array axes.

    Returns
    -------
    reflected_axis: `numpy.ndarray` of `int`
        The axis number(s) after reflection.
    """
    # Check type of input.
    if not isinstance(axis, np.ndarray):
        raise TypeError(f"input must be of array type. Got type: {type(axis)}")
    if axis.dtype.char not in np.typecodes['AllInteger']:
        raise TypeError(f"input dtype must be of int type.  Got dtype: {axis.dtype})")
    # Convert negative indices to positive equivalents.
    axis[axis < 0] += naxes
    if any(axis > naxes - 1):
        raise IndexError("Axis out of range.  "
                         f"Number of axes = {naxes}; Axis numbers requested = {axis}")
    # Reflect axis about center of number of axes.
    reflected_axis = naxes - 1 - axis

    return reflected_axis


def pixel_axis_to_world_axes(pixel_axis, axis_correlation_matrix):
    """
    Retrieves the indices of the world axis physical types corresponding to a pixel axis.

    Parameters
    ----------
    pixel_axis: `int`
        The pixel axis index/indices for which the world axes are desired.

    axis_correlation_matrix: `numpy.ndarray` of `bool`
        2D boolean correlation matrix defining the dependence between the pixel and world axes.
        Format same as `astropy.wcs.BaseLowLevelWCS.axis_correlation_matrix`.

    Returns
    -------
    world_axes: `numpy.ndarray`
        The world axis indices corresponding to the pixel axis.
    """
    return np.arange(axis_correlation_matrix.shape[0])[axis_correlation_matrix[:, pixel_axis]]


def world_axis_to_pixel_axes(world_axis, axis_correlation_matrix):
    """
    Gets the pixel axis indices corresponding to the index of a world axis physical type.

    Parameters
    ----------
    world_axis: `int`
        The index of the physical type for which the pixes axes are desired.

    axis_correlation_matrix: `numpy.ndarray` of `bool`
        2D boolean correlation matrix defining the dependence between the pixel and world axes.
        Format same as `astropy.wcs.BaseLowLevelWCS.axis_correlation_matrix`.

    Returns
    -------
    pixel_axes: `numpy.ndarray`
        The pixel axis indices corresponding to the world axis.
    """
    return np.arange(axis_correlation_matrix.shape[1])[axis_correlation_matrix[world_axis]]


def pixel_axis_to_physical_types(pixel_axis, wcs):
    """
    Gets the world axis physical types corresponding to a pixel axis.

    Parameters
    ----------
    pixel_axis: `int`
        The pixel axis number(s) for which the world axis numbers are desired.

    wcs: `astropy.wcs.BaseLowLevelWCS`
        The WCS object defining the relationship between pixel and world axes.

    Returns
    -------
    physical_types: `numpy.ndarray` of `str`
        The physical types corresponding to the pixel axis.
    """
    return np.array(wcs.world_axis_physical_types)[wcs.axis_correlation_matrix[:, pixel_axis]]


def physical_type_to_pixel_axes(physical_type, wcs):
    """
    Gets the pixel axis indices corresponding to a world axis physical type.

    Parameters
    ----------
    physical_type: `int`
        The pixel axis number(s) for which the world axis numbers are desired.

    wcs: `astropy.wcs.BaseLowLevelWCS`
        The WCS object defining the relationship between pixel and world axes.

    Returns
    -------
    pixel_axes: `numpy.ndarray`
        The pixel axis indices corresponding to the physical type.
    """
    world_axis = physical_type_to_world_axis(physical_type, wcs.world_axis_physical_types)
    return world_axis_to_pixel_axes(world_axis, wcs.axis_correlation_matrix)


def physical_type_to_world_axis(physical_type, world_axis_physical_types):
    """
    Returns world axis index of a physical type based on WCS world_axis_physical_types.

    Input can be a substring of a physical type, so long as it is unique.

    Parameters
    ----------
    physical_type: `str`
        The physical type or a substring unique to a physical type.

    world_axis_physical_types: sequence of `str`
        All available physical types.  Ordering must be same as
        `astropy.wcs.BaseLowLevelWCS.world_axis_physical_types`

    Returns
    -------
    world_axis: `numbers.Integral`
        The world axis index of the physical type.
    """
    # Find world axis index described by physical type.
    widx = np.where(world_axis_physical_types == physical_type)[0]
    # If physical type does not correspond to entry in world_axis_physical_types,
    # check if it is a substring of any physical types.
    if len(widx) == 0:
        widx = [physical_type in world_axis_physical_type
                for world_axis_physical_type in world_axis_physical_types]
        widx = np.arange(len(world_axis_physical_types))[widx]
    if len(widx) != 1:
        raise ValueError(
            "Input does not uniquely correspond to a physical type."
            f" Expected unique substring of one of {world_axis_physical_types}."
            f"  Got: {physical_type}"
        )
    # Return axes with duplicates removed.
    return widx[0]


def get_dependent_pixel_axes(pixel_axis, axis_correlation_matrix):
    """
    Find indices of all pixel axes associated with the world axes linked to the input pixel axis.

    For example, say the input pixel axis is 0 and it is associated with two world axes
    corresponding to longitude and latitude. Let's also say that pixel axis 1 is also
    associated with longitude and latitude. Thus, this function would return pixel axes 0 and 1.
    On the other hand let's say pixel axis 2 is associated with only one world axis,
    e.g. wavelength, which does not depend on any other pixel axis (i.e. it is independent).
    In that case this function would only return pixel axis 2.
    Both input and output pixel axis indices are in the WCS ordering convention
    (reverse of numpy ordering convention).
    The returned axis indices include the input axis.

    Parameters
    ----------
    wcs_axis: `int`
        Index of axis (in WCS ordering convention) for which dependent axes are desired.

    axis_correlation_matrix: `numpy.ndarray` of `bool`
        2D boolean correlation matrix defining the dependence between the pixel and world axes.
        Format same as `astropy.wcs.BaseLowLevelWCS.axis_correlation_matrix`.

    Returns
    -------
    dependent_pixel_axes: `np.ndarray` of `int`
        Sorted indices of pixel axes dependent on input axis in WCS ordering convention.
    """
    # The axis_correlation_matrix is (n_world, n_pixel) but we want to know
    # which pixel coordinates are linked to which other pixel coordinates.
    # To do this we take a column from the matrix and find if there are
    # any entries in common with all other columns in the matrix.
    world_dep = axis_correlation_matrix[:, pixel_axis:pixel_axis + 1]
    dependent_pixel_axes = np.sort(np.nonzero((world_dep & axis_correlation_matrix).any(axis=0))[0])
    return dependent_pixel_axes


def get_dependent_array_axes(array_axis, axis_correlation_matrix):
    """
    Find indices of all array axes associated with the world axes linked to the input array axis.

    For example, say the input array axis is 0 and it is associated with two world axes
    corresponding to longitude and latitude. Let's also say that array axis 1 is also
    associated with longitude and latitude. Thus, this function would return array axes 0 and 1.
    Note the the output axes include the input axis. On the other hand let's say
    array axis 2 is associated with only one world axis, e.g. wavelength,
    which does not depend on any other array axis (i.e. it is independent).
    In that case this function would only return array axis 2.
    Both input and output array axis indices are in the numpy array ordering convention
    (reverse of WCS ordering convention).
    The returned axis indices include the input axis.

    Parameters
    ----------
    array_axis: `int`
        Index of array axis (in numpy ordering convention) for which dependent axes are desired.

    axis_correlation_matrix: `numpy.ndarray` of `bool`
        2D boolean correlation matrix defining the dependence between the pixel and world axes.
        Format same as `astropy.wcs.BaseLowLevelWCS.axis_correlation_matrix`.

    Returns
    -------
    dependent_array_axes: `np.ndarray` of `int`
        Sorted indices of array axes dependent on input axis in numpy ordering convention.
    """
    naxes = axis_correlation_matrix.shape[1]
    pixel_axis = convert_between_array_and_pixel_axes(np.array([array_axis], dtype=int), naxes)[0]
    dependent_pixel_axes = get_dependent_pixel_axes(pixel_axis, axis_correlation_matrix)
    dependent_array_axes = convert_between_array_and_pixel_axes(dependent_pixel_axes, naxes)
    return np.sort(dependent_array_axes)


def get_dependent_world_axes(world_axis, axis_correlation_matrix):
    """
    Given a WCS world axis index, return indices of dependent WCS world axes.

    Both input and output axis indices are in the WCS ordering convention
    (reverse of numpy ordering convention). The returned axis indices include the input axis.

    Parameters
    ----------
    world_axis: `int`
        Index of axis (in WCS ordering convention) for which dependent axes are desired.

    axis_correlation_matrix: `numpy.ndarray` of `bool`
        2D boolean correlation matrix defining the dependence between the pixel and world axes.
        Format same as `astropy.wcs.BaseLowLevelWCS.axis_correlation_matrix`.

    Returns
    -------
    dependent_world_axes: `np.ndarray` of `int`
        Sorted indices of pixel axes dependent on input axis in WCS ordering convention.
    """
    # The axis_correlation_matrix is (n_world, n_pixel) but we want to know
    # which world coordinates are linked to which other world coordinates.
    # To do this we take a row from the matrix and find if there are
    # any entries in common with all other rows in the matrix.
    pixel_dep = axis_correlation_matrix[world_axis:world_axis + 1]
    dependent_world_axes = np.sort(np.nonzero((pixel_dep & axis_correlation_matrix).any(axis=1))[0])
    return dependent_world_axes


def get_dependent_physical_types(physical_type, wcs):
    """
    Given a world axis physical type, return the dependent physical types including the input type.

    Parameters
    ----------
    physical_type: `str`
        The world axis physical types whose dependent physical types are desired.

    wcs: `astropy.wcs.BaseLowLevelWCS`
        The WCS object defining the relationship between pixel and world axes.

    Returns
    -------
    dependent_physical_types: `np.ndarray` of `str`
        Physical types dependent on the input physical type.
    """
    world_axis_physical_types = wcs.world_axis_physical_types
    world_axis = physical_type_to_world_axis(physical_type, world_axis_physical_types)
    dependent_world_axes = get_dependent_world_axes(world_axis, wcs.axis_correlation_matrix)
    dependent_physical_types = np.array(world_axis_physical_types)[dependent_world_axes]
    return dependent_physical_types


def validate_physical_types(physical_types):
    """
    Validate a list of physical types against the UCD1+ standard
    """
    try:
        low_level_api.validate_physical_types(physical_types)
    except ValueError as e:
        invalid_type = str(e).split(':')[1].strip()
        raise ValueError(
            f"'{invalid_type}' is not a valid IOVA UCD1+ physical type. "
            "It must be a string specified in the list (http://www.ivoa.net/documents/latest/UCDlist.html) "
            "or if no matching type exists it can be any string prepended with 'custom:'."
        )


def calculate_world_indices_from_axes(wcs, axes):
    """
    Given a string representation of a world axis or a numerical array index, convert it
    to a numerical world index aligning to the position in
    wcs.world_axis_object_components.
    """
    # Convert input axes to WCS world axis indices.
    world_indices = []
    for axis in axes:
        if isinstance(axis, numbers.Integral):
            # If axis is int, it is a numpy order array axis.
            # Convert to pixel axis in WCS order.
            axis = convert_between_array_and_pixel_axes(np.array([axis]), wcs.pixel_n_dim)[0]
            # Get WCS world axis indices that correspond to the WCS pixel axis
            # and add to list of indices of WCS world axes whose coords will be returned.
            world_indices += list(pixel_axis_to_world_axes(axis, wcs.axis_correlation_matrix))
        elif isinstance(axis, str):
            # If axis is str, it is a physical type or substring of a physical type.
            world_indices.append(physical_type_to_world_axis(axis, wcs.world_axis_physical_types))
        else:
            raise TypeError(f"Unrecognized axis type: {axis, type(axis)}. "
                            "Must be of type (numbers.Integral, str)")
    # Use inferred world axes to extract the desired coord value
    # and corresponding physical types.
    return np.unique(np.array(world_indices, dtype=int))


def array_indices_for_world_objects(wcs, axes=None):
    """
    Calculate the array indices corresponding to each high level world object.

    This function is to assist in comparing the return values from
    `.NDCube.axis_world_coords` or
    `~astropy.wcs.wcsapi.BaseHighLevelWCS.world_to_pixel` it returns a tuple of
    the same length as the output from those methods with each element being
    the array indices corresponding to those objects.

    Parameters
    ----------
    wcs : `astropy.wcs.wcsapi.BaseHighLevelWCS`
        The wcs object used to calculate world coordinates.
    axes : iterable of `int` or `str`
        Axis number in numpy ordering or unique substring of
        ``wcs.world_axis_physical_types``
        of axes for which real world coordinates are desired.
        axes=None implies all axes will be returned.

    Returns
    -------
    array_indices : `tuple` of `tuple` of `int`
        For each world object, a tuple of array axes identified by their
        number. Array indices in each sub-tuple are not guaranteed to be
        ordered with respect to the arrays in the object, as the object could
        be an object like ``SkyCoord`` where there is a separation of the two
        coordinates. The array indices will be returned in the sub-tuple in
        array index order, i.e ascending.
    """
    if axes:
        world_indices = calculate_world_indices_from_axes(wcs, axes)
    else:
        world_indices = np.arange(wcs.world_n_dim)
    object_names = np.array([wao_comp[0]
                             for wao_comp in wcs.low_level_wcs.world_axis_object_components])
    array_indices = [[]] * len(object_names)
    for world_index, oname in enumerate(object_names):
        # If this world index is deselected by axes= then skip
        if world_index not in world_indices:
            continue
        # Select the first occurence of the object name.
        # Other occurences are ignored so as to return duplicate coordinate objects.
        oinds = np.atleast_1d(object_names == oname).nonzero()[0][0]
        # Calculate the array axes corresponding the coordinate's world axis
        # and enter them into the element of the array indices array corresponding
        # to the relevant world coordinate object.
        pixel_index = world_axis_to_pixel_axes(world_index, wcs.axis_correlation_matrix)
        array_index = convert_between_array_and_pixel_axes(pixel_index, wcs.pixel_n_dim)
        array_indices[oinds] = tuple(array_index[::-1])  # Invert from pixel order to array order
    return tuple(ai for ai in array_indices if ai)


def get_low_level_wcs(wcs, name='wcs'):
    """
    Returns a low level WCS object from a low level or high level WCS.

    Parameters
    ----------
    wcs: `astropy.wcs.wcsapi.BaseHighLevelWCS` or `astropy.wcs.wcsapi.BaseLowLevelWCS`
        The input WCS for getting the low level WCS object.

    name: `str`, optional
        Any name for the wcs to be used in the exception that could be raised.

    Returns
    -------
    wcs: `astropy.wcs.wcsapi.BaseLowLevelWCS`
    """

    if isinstance(wcs, BaseHighLevelWCS):
        return wcs.low_level_wcs
    elif isinstance(wcs, BaseLowLevelWCS):
        return wcs
    else:
        raise(f'{name} must implement either BaseHighLevelWCS or BaseLowLevelWCS')


def compare_wcs_physical_types(source_wcs, target_wcs):
    """
    Checks to see if two WCS objects have the same physical types in the same order.

    Parameters
    ----------
    source_wcs : `astropy.wcs.wcsapi.BaseHighLevelWCS` or `astropy.wcs.wcsapi.BaseLowLevelWCS`
        The WCS which is currently in use, usually `self.wcs`.

    target_wcs : `astropy.wcs.wcsapi.BaseHighLevelWCS` or `astropy.wcs.wcsapi.BaseLowLevelWCS`
        The WCS object on which the NDCube is to be reprojected.

    Returns
    -------
    result : `bool`
    """

    source_wcs = get_low_level_wcs(source_wcs, 'source_wcs')
    target_wcs = get_low_level_wcs(target_wcs, 'target_wcs')

    return source_wcs.world_axis_physical_types == target_wcs.world_axis_physical_types


def identify_invariant_axes(source_wcs, target_wcs, input_shape, atol=1e-6, rtol=1e-6):
    """
    Performs a pixel to pixel transformation to identify if there are any invariant axes
    between the given source and target WCS objects.

    Parameters
    ----------
    source_wcs: `astropy.wcs.wcsapi.BaseHighLevelWCS` or `astropy.wcs.wcsapi.BaseLowLevelWCS`

    target_wcs: `astropy.wcs.wcsapi.BaseHighLevelWCS` or `astropy.wcs.wcsapi.BaseLowLevelWCS`

    input_shape: `tuple`
        The array shape of the data.

    atol: `float`
        The absolute tolerance parameter for comparison.

    rtol: `float`
        The relative tolerance parameter for comparison.

    Returns
    -------
    result: `list`
        A list of booleans denoting whether the axis is invariant or not.
        Follows the WCS ordering.
    """

    input_pixel_coords = np.meshgrid(*[np.arange(n) for n in input_shape])

    output_pixel_coords = pixel_to_pixel(source_wcs, target_wcs, *input_pixel_coords)

    return [np.allclose(input_coord, output_coord, atol=atol, rtol=rtol)
            for input_coord, output_coord in zip(input_pixel_coords, output_pixel_coords)]
