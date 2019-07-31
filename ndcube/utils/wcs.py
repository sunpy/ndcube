# -*- coding: utf-8 -*-
# Author: Ankit Baruah and Daniel Ryan <ryand5@tcd.ie>

"""Miscellaneous WCS utilities"""

import re
from copy import deepcopy
from collections import UserDict

import numpy as np
from astropy import wcs
from astropy.wcs._wcs import InconsistentAxisTypesError
from astropy.units import Quantity

from ndcube.utils import cube as utils_cube

__all__ = ['WCS', 'reindex_wcs', 'wcs_ivoa_mapping', 'get_dependent_data_axes',
           'get_dependent_data_axes', 'axis_correlation_matrix',
           'append_sequence_axis_to_wcs']


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
    "ZPIXEL": "custom:instr.pixel.z"
    }
wcs_ivoa_mapping = TwoWayDict()
for key in wcs_to_ivoa.keys():
    wcs_ivoa_mapping[key] = wcs_to_ivoa[key]


class WCS(wcs.WCS):

    def __init__(self, header=None, naxis=None, **kwargs):
        """
        Initiates a WCS object with additional functionality to add dummy axes.

        Not all WCS axes are independent.  Some, e.g. latitude and longitude,
        are dependent and one cannot be used without the other.  Therefore this
        WCS class has the ability to determine whether a dependent axis is missing
        and can augment the WCS axes with a dummy axis to enable the translations
        to work.

        Parameters
        ----------
        header: FITS header or `dict` with appropriate FITS keywords.

        naxis: `int`
            Number of axis described by the header.

        """
        self.oriented = False
        self.was_augmented = WCS._needs_augmenting(header)
        if self.was_augmented:
            header = WCS._augment(header, naxis)
            if naxis is not None:
                naxis = naxis + 1
        super(WCS, self).__init__(header=header, naxis=naxis, **kwargs)

    @classmethod
    def _needs_augmenting(cls, header):
        """
        Determines whether a missing dependent axis is missing from the WCS object.

        WCS cannot be created with only one spacial dimension. If
        WCS detects that returns that it needs to be augmented.

        Parameters
        ----------
        header: FITS header or `dict` with appropriate FITS keywords.

        """
        try:
            wcs.WCS(header=header)
        except InconsistentAxisTypesError as err:
            if re.search(r'Unmatched celestial axes', str(err)):
                return True
        return False

    @classmethod
    def _augment(cls, header, naxis):
        """
        Augments WCS with a dummy axis to take the place of a missing dependent axis.

        """
        newheader = deepcopy(header)
        new_wcs_axes_params = {'CRPIX': 0, 'CDELT': 1, 'CRVAL': 0,
                               'CNAME': 'redundant axis', 'CTYPE': 'HPLN-TAN',
                               'CROTA': 0, 'CUNIT': 'deg', 'NAXIS': 1}
        axis = str(max(newheader.get('NAXIS', 0), naxis) + 1)
        for param in new_wcs_axes_params:
            attr = new_wcs_axes_params[param]
            newheader[param + axis] = attr
        try:
            print(wcs.WCS(header=newheader).get_axis_types())
        except InconsistentAxisTypesError as err:
            projection = re.findall(r'expected [^,]+', str(err))[0][9:]
            newheader['CTYPE' + axis] = projection
        return newheader


def _wcs_slicer(wcs, missing_axes, item):
    """
    Returns the new sliced wcs, changed missing axes, and coordinates of dropped axes.

    Parameters
    ---------
    wcs: `astropy.wcs.WCS` or `ndcube.utils.wcs.WCS`
        WCS object to be sliced.

    missing_axes: `list` of `bool`
        Indicates which axes of the WCS are "missing" i.e. do not correspond to a data axis.
        Must be supplied in WCS order.

    item: `int`, `slice` or `tuple` of `int` and/or `slice`.
        Slicing item. Must be supplied in numpy order and not include entries for missing axes.
        See Notes for further explanation.

    Returns
    -------
    new_wcs: `astropy.wcs.WCS` or `ndcube.utils.wcs.WCS`
        Sliced WCS object.

    new_missing_axes: `list` of `bool`
        Altered missing axis list in WCS order.

    dropped_coords:
        Coordinates which have been dropped in the slicing process is collected
        in a dictionary called dropped_coords.

    Notes
    -----
    A clarifying example of axis ordering of wcs, missing_axes and item.

    Let's say we have a wcs object with four axes (2, 1, 5, 4) and a
    missing_axes list of [False, True, False, False].
    This means that the 1th axis of the wcs is "missing".
    Note that the ordering of these axes are in WCS order, i.e. the reverse of
    numpy order which is the order the data axes are given.
    So the data shape must be (4, 5, 2), given that we have one "missing" axis.
    The item object must be in numpy order and mustn't account for missing axes.
    So a valid item object would be (slice(2, 4), slice(0, 1), slice(None)),
    where the 0th entry gives the slice to be applied the WCS axis of length 4,
    the middle slice is applied to WCS axis of length 5,
    the missing axis is not represented,
    and the last entry is applied to the WCS axis to length 2.

    """
    # item is entered in numpy order and does not include entries for missing axes.
    # But it is easier for it to be in WCS order and include entries for missing axes
    # for extracting real world coordinates of dropped axes.
    # Due to the possibility of missing axes, the best thing to do
    # is first reverse the missing_axes variable, then prep the item
    # to account for those missing axes and any non-missing axes not to be sliced.
    # This will make the item the same length as the number of axes in the WCS object
    # meaning it can be safely reversed to WCS order.
    new_missing_axes = deepcopy(missing_axes)
    missing_axes_numpy_order = new_missing_axes[::-1]

    # Next prep the item to include slices foe missing axes and
    # non-missing axes that aren't to be sliced.
    # To do this, create a tuple of slices where
    # elements corresponding to missing axes are set to slice(0,1),
    # non-missing axes with a corresponding slice in item are assigned that slice,
    # and subsequent non-missing axes without an entry in item are set to slice(None).
    # If item or tuple element is an int, convert to the appropriate slice
    # so we easily search for new missing/dropped axes later.
    item_checked = []
    # Case where item is a slice or int.
    if isinstance(item, (slice, int, np.int64)):
        # Create index to track whether we have reached the axis relevant to the item.
        index = 0
        for i, _bool in enumerate(missing_axes_numpy_order):
            if _bool:
                # Enter slice(0, 1) for any missing axis.
                item_checked.append(slice(0, 1))
            else:
                if index == 0:
                    # Enter item into tuple for first non-missing axis.
                    if isinstance(item, slice):
                        item_checked.append(item)
                    else:
                        item_checked.append(slice(item, item + 1))
                else:
                    # As item is a slice or int, subsequent non-missing axes are not to be sliced.
                    item_checked.append(slice(None, None))
                index += 1
            item_ = tuple(item_checked)
    # Case where item is a tuple of slices or ints.
    elif isinstance(item, tuple):
        # Create index to track whether the item tuple elements we are dealing with.
        index = 0
        len_item = len(item)
        for i, _bool in enumerate(missing_axes_numpy_order):
            if _bool:
                # Enter slice(0, 1) for any missing axis.
                item_checked.append(slice(0, 1))
            else:
                if index < len_item:
                    # For non-missing axes with a corresponding sub-item,
                    # append that item here. If the sub-item is an int,
                    # convert to appropriate slice for easy identification
                    # of newly missing/dropped axes later.
                    if isinstance(item[index], (int, np.int64)):
                        item_checked.append(slice(item[index], item[index]+1))
                    elif isinstance(item[index], slice):
                        item_checked.append(item[index])
                    else:
                        raise TypeError("item type at data axis {0} is {1}. ".format(
                            index, type(item[index]) + "Must be int or slice."))
                else:
                    # Subsequent non-missing axes did not have a corresponding slice item
                    # and are therefore not be sliced.
                    item_checked.append(slice(None, None))
                index += 1
        item_ = tuple(item_checked)

    # Case where item is an an invalid format.
    else:
        raise TypeError("item type is {0}.  ".format(type(item)) +
                        "Must be int, slice, or tuple of ints and/or slices.")

    # Now item_checked has an entry for all axes, missing and non-missing,
    # it can be safely reverse to WCS order
    # which makes extracting real world coordinates of newly missing/dropped axes easier.
    item_wcs_order = tuple(item_checked[::-1])

    # Determine which axes will be made missing by this slicing
    # and extract the real world coordinate for each.
    # For this, use WCS-order variables, i.e. missing_axes and item_wcs_order.
    dropped_coords = {}
    for i, slice_element in enumerate(item_wcs_order):
        # If axis is not missing and the difference between its start and stop params is 1,
        # then the slicing will cause the axis to be dropped, i.e. become missing.
        if new_missing_axes[i] is False:
            if isinstance(slice_element, slice):
                # Determine the start index.
                if slice_element.start is None:
                    slice_start = 0
                else:
                    slice_start = slice_element.start
                # Determine the stop index.
                if slice_element.stop is None:
                    # wcs._pixel_shape is a list of the length of each axis.
                    slice_stop = wcs.pixel_shape[i]
                else:
                    slice_stop = slice_element.stop
            elif isinstance(slice_element, int):
                slice_start = slice_element
                slice_stop = slice_element + 1
            # Determine the slice's step.
            # (We will use this is a later version of this code to be more thorough.
            # For now we'll calculate it and not use it.)
            # if slice_element.step is None:
                # slice_step = 1
            # else:
                # slice_step = slice_element.step
            slice_step = 1
            real_world_coords = []
            # If slice results in the axis being of length 1, is will be dropped.
            # Calculate its real world coordinate.c
            if slice_stop - slice_start <= slice_step:
                # Set up a list of pixel coords as input to all_pix2world.
                pix_coords = [0] * len(item_wcs_order)
                # Enter pixel coordinate for this axis.
                pix_coords[i] = slice_start
                # Get real world coordinates of i-th axis.
                real_world_coords = wcs.all_pix2world(*pix_coords, 0)[i]
                # Get IVOA axis name from CTYPE.
                axis_name = _get_ivoa_from_ctype(wcs.wcs.ctype[i])
                # Add dropped coordinate's name, axis and value to dropped_coords dict of dicts.
                dropped_coords[axis_name] = {"wcs axis": i,
                                             "value": real_world_coords * wcs.wcs.cunit[i]}
                new_missing_axes[i] = True
    # Use item_wcs_order to slice WCS.
    new_wcs = wcs.slice(item_wcs_order, numpy_order=False)
    return new_wcs, new_missing_axes, dropped_coords


def _get_ivoa_from_ctype(ctype):
    """
    Find keys in wcs_ivoa_mapping dict that represent start of CTYPE.
    Ensure CTYPE is capitalized.

    """
    keys = list(filter(lambda key: ctype.upper().startswith(key), wcs_ivoa_mapping))
    # If there are multiple valid keys, raise an error.
    n_keys = len(keys)
    if n_keys == 0:
        axis_name = ctype
    elif n_keys == 1:
        axis_name = wcs_ivoa_mapping[key[0]]
    else:
        raise ValueError("Non-unique CTYPE key.")
    return axis_name


def _all_slice(obj):
    """
    Returns True if all the elements in the object are slices else return False
    """
    result = False
    if not isinstance(obj, (tuple, list)):
        return result
    result |= all(isinstance(o, slice) for o in obj)
    return result


def _slice_list(obj):
    """
    Return list of all the slices.

    Example
    -------
    >>> _slice_list((slice(1,2), slice(1,3), 2, slice(2,4), 8))
    [slice(1, 2, None), slice(1, 3, None), slice(2, 3, None), slice(2, 4, None), slice(8, 9, None)]
    """
    result = []
    if not isinstance(obj, (tuple, list)):
        return result
    for i, o in enumerate(obj):
        if isinstance(o, int):
            result.append(slice(o, o+1))
        elif isinstance(o, slice):
            result.append(o)
    return result


def reindex_wcs(wcs, inds):
    # From astropy.spectral_cube.wcs_utils
    """
    Re-index a WCS given indices.  The number of axes may be reduced.

    Parameters
    ----------
    wcs: sunpy.wcs.wcs.WCS
        The WCS to be manipulated
    inds: np.array(dtype='int')
        The indices of the array to keep in the output.
        e.g. swapaxes: [0,2,1,3]
        dropaxes: [0,1,3]
    """

    if not isinstance(inds, np.ndarray):
        raise TypeError("Indices must be an ndarray")

    if inds.dtype.kind != 'i':
        raise TypeError('Indices must be integers')

    outwcs = WCS(naxis=len(inds))
    wcs_params_to_preserve = ['cel_offset', 'dateavg', 'dateobs', 'equinox',
                              'latpole', 'lonpole', 'mjdavg', 'mjdobs', 'name',
                              'obsgeo', 'phi0', 'radesys', 'restfrq',
                              'restwav', 'specsys', 'ssysobs', 'ssyssrc',
                              'theta0', 'velangl', 'velosys', 'zsource']
    for par in wcs_params_to_preserve:
        setattr(outwcs.wcs, par, getattr(wcs.wcs, par))

    cdelt = wcs.wcs.cdelt

    try:
        outwcs.wcs.pc = wcs.wcs.pc[inds[:, None], inds[None, :]]
    except AttributeError:
        outwcs.wcs.pc = np.eye(wcs.naxis)

    outwcs.wcs.crpix = wcs.wcs.crpix[inds]
    outwcs.wcs.cdelt = cdelt[inds]
    outwcs.wcs.crval = wcs.wcs.crval[inds]
    outwcs.wcs.cunit = [wcs.wcs.cunit[i] for i in inds]
    outwcs.wcs.ctype = [wcs.wcs.ctype[i] for i in inds]
    outwcs.wcs.cname = [wcs.wcs.cname[i] for i in inds]
    outwcs._naxis = [wcs._naxis[i] for i in inds]

    return outwcs


def get_dependent_data_axes(wcs_object, data_axis, missing_axes):
    """
    Given a data axis index, return indices of dependent data axes.

    Both input and output axis indices are in the numpy ordering convention
    (reverse of WCS ordering convention). The returned axis indices include the input axis.
    Returned axis indices do NOT include any WCS axes that do not have a
    corresponding data axis, i.e. "missing" axes.

    Parameters
    ----------
    wcs_object: `astropy.wcs.WCS` or `ndcube.utils.wcs.WCS`
        The WCS object describing the axes.

    data_axis: `int`
        Index of axis (in numpy ordering convention) for which dependent axes are desired.

    missing_axes: iterable of `bool`
        Indicates which axes of the WCS are "missing", i.e. do not correspond to a data axis.

    Returns
    -------
    dependent_data_axes: `tuple` of `int`
        Sorted indices of axes dependent on input data_axis in numpy ordering convention.

    """
    # In order to correctly account for "missing" axes in this process,
    # we must determine what axes are dependent based on WCS axis indices.
    # Convert input data axis index to WCS axis index.
    wcs_axis = utils_cube.data_axis_to_wcs_axis(data_axis, missing_axes)
    # Determine dependent axes, including "missing" axes, using WCS ordering.
    wcs_dependent_axes = np.asarray(get_dependent_wcs_axes(wcs_object, wcs_axis))
    # Remove "missing" axes from output.
    non_missing_wcs_dependent_axes = wcs_dependent_axes[
        np.invert(missing_axes)[wcs_dependent_axes]]
    # Convert dependent axes back to numpy/data ordering.
    dependent_data_axes = tuple(np.sort([utils_cube.wcs_axis_to_data_axis(i, missing_axes)
                                         for i in non_missing_wcs_dependent_axes]))
    return dependent_data_axes


def get_dependent_wcs_axes(wcs_object, wcs_axis):
    """
    Given a WCS axis index, return indices of dependent WCS axes.

    Both input and output axis indices are in the WCS ordering convention
    (reverse of numpy ordering convention). The returned axis indices include the input axis.
    Returned axis indices DO include WCS axes that do not have a
    corresponding data axis, i.e. "missing" axes.

    Parameters
    ----------
    wcs_object: `astropy.wcs.WCS` or `ndcube.utils.wcs.WCS`
        The WCS object describing the axes.

    wcs_axis: `int`
        Index of axis (in WCS ordering convention) for which dependent axes are desired.

    Returns
    -------
    dependent_data_axes: `tuple` of `int`
        Sorted indices of axes dependent on input data_axis in WCS ordering convention.

    """
    # Pre-compute dependent axes. The matrix returned by
    # axis_correlation_matrix is (n_world, n_pixel) but we want to know
    # which pixel coordinates are linked to which other pixel coordinates.
    # So to do this we take a column from the matrix and find if there are
    # any entries in common with all other columns in the matrix.
    matrix = axis_correlation_matrix(wcs_object)
    world_dep = matrix[:, wcs_axis:wcs_axis + 1]
    dependent_wcs_axes = tuple(np.sort(np.nonzero((world_dep & matrix).any(axis=0))[0]))
    return dependent_wcs_axes


def axis_correlation_matrix(wcs_object):
    """
    Return True/False matrix indicating which WCS axes are dependent on others.

    Parameters
    ----------
    wcs_object: `astropy.wcs.WCS` or `ndcube.utils.wcs.WCS`
        The WCS object describing the axes.

    Returns
    -------
    matrix: `numpy.ndarray` of `bool`
        Square True/False matrix indicating which axes are dependent.
        For example, whether WCS axis 0 is dependent on WCS axis 1 is given by matrix[0, 1].

    """
    n_world = len(wcs_object.wcs.ctype)
    n_pixel = wcs_object.naxis

    # If there are any distortions present, we assume that there may be
    # correlations between all axes. Maybe if some distortions only apply
    # to the image plane we can improve this
    for distortion_attribute in ('sip', 'det2im1', 'det2im2'):
        if getattr(wcs_object, distortion_attribute):
            return np.ones((n_world, n_pixel), dtype=bool)

    # Assuming linear world coordinates along each axis, the correlation
    # matrix would be given by whether or not the PC matrix is zero
    matrix = wcs_object.wcs.get_pc() != 0

    # We now need to check specifically for celestial coordinates since
    # these can assume correlations because of spherical distortions. For
    # each celestial coordinate we copy over the pixel dependencies from
    # the other celestial coordinates.
    celestial = (wcs_object.wcs.axis_types // 1000) % 10 == 2
    celestial_indices = np.nonzero(celestial)[0]
    for world1 in celestial_indices:
        for world2 in celestial_indices:
            if world1 != world2:
                matrix[world1] |= matrix[world2]
                matrix[world2] |= matrix[world1]

    return matrix


def append_sequence_axis_to_wcs(wcs_object):
    """Appends a 1-to-1 dummy axis to a WCS object."""
    dummy_number = wcs_object.naxis+1
    wcs_header = wcs_object.to_header()
    wcs_header.append(("CTYPE{0}".format(dummy_number), "ITER",
                       "A unitless iteration-by-one axis."))
    wcs_header.append(("CRPIX{0}".format(dummy_number), 0.,
                       "Pixel coordinate of reference point"))
    wcs_header.append(("CDELT{0}".format(dummy_number), 1.,
                       "Coordinate increment at reference point"))
    wcs_header.append(("CRVAL{0}".format(dummy_number), 0.,
                       "Coordinate value at reference point"))
    wcs_header.append(("CUNIT{0}".format(dummy_number), "pix",
                       "Coordinate value at reference point"))
    wcs_header["WCSAXES"] = dummy_number
    return WCS(wcs_header)
