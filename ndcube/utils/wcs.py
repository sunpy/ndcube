# Author: Ankit Baruah and Daniel Ryan <ryand5@tcd.ie>

"""
Miscellaneous WCS utilities.
"""

import re
from copy import deepcopy
from collections import UserDict
import numbers

import numpy as np
from astropy import wcs
from astropy.wcs._wcs import InconsistentAxisTypesError

from ndcube.utils import cube as utils_cube

__all__ = ['WCS', 'reindex_wcs', 'wcs_ivoa_mapping', 'get_dependent_data_axes',
           'get_dependent_wcs_axes', 'append_sequence_axis_to_wcs']


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
        super().__init__(header=header, naxis=naxis, **kwargs)

    @classmethod
    def _needs_augmenting(cls, header):
        """
        Determines whether a missing dependent axis is missing from the WCS
        object.

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
        Augments WCS with a dummy axis to take the place of a missing dependent
        axis.
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


def _all_slice(obj):
    """
    Returns True if all the elements in the object are slices else return
    False.
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
            result.append(slice(o, o + 1))
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


def get_dependent_data_axes(wcs_object, data_axis):
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

    Returns
    -------
    dependent_data_axes: `tuple` of `int`
        Sorted indices of axes dependent on input data_axis in numpy ordering convention.
    """
    # Convert input data axis index to WCS axis index.
    wcs_axis = utils_cube.data_axis_to_wcs_ape14(data_axis, _pixel_keep(wcs_object), wcs_object.pixel_n_dim)
    # Determine dependent axes, using WCS ordering.
    wcs_dependent_axes = np.asarray(get_dependent_wcs_axes(wcs_object, wcs_axis))

    # Convert dependent axes back to numpy/data ordering.
    dependent_data_axes = tuple(np.sort([utils_cube.wcs_axis_to_data_ape14(
        i, _pixel_keep(wcs_object), wcs_object.pixel_n_dim) for i in wcs_dependent_axes]))
    return dependent_data_axes


def get_dependent_wcs_axes(wcs_object, wcs_axis):
    """
    Given a WCS axis index, return indices of dependent WCS axes.

    Both input and output axis indices are in the WCS ordering convention
    (reverse of numpy ordering convention). The returned axis indices include the input axis.

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

    # Using APE14 for generating the correlation matrix
    matrix = wcs_object.axis_correlation_matrix
    world_dep = matrix[:, wcs_axis:wcs_axis + 1]
    dependent_wcs_axes = tuple(np.sort(np.nonzero((world_dep & matrix).any(axis=0))[0]))
    return dependent_wcs_axes


def append_sequence_axis_to_wcs(wcs_object):
    """
    Appends a 1-to-1 dummy axis to a WCS object.
    """
    dummy_number = wcs_object.naxis + 1
    wcs_header = wcs_object.to_header()
    wcs_header.append((f"CTYPE{dummy_number}", "ITER",
                       "A unitless iteration-by-one axis."))
    wcs_header.append((f"CRPIX{dummy_number}", 0.,
                       "Pixel coordinate of reference point"))
    wcs_header.append((f"CDELT{dummy_number}", 1.,
                       "Coordinate increment at reference point"))
    wcs_header.append((f"CRVAL{dummy_number}", 0.,
                       "Coordinate value at reference point"))
    wcs_header.append((f"CUNIT{dummy_number}", "pix",
                       "Coordinate value at reference point"))
    wcs_header["WCSAXES"] = dummy_number
    return WCS(wcs_header)


def _pixel_keep(wcs_object):
    """Returns the value of the _pixel_keep attribute if available
    else returns the array of all pixel dimension present.

    Parameters
    ----------
    wcs_object : `astropy.wcs.WCS` or alike object

    Returns
    -------
    list or `np.ndarray` object
    """
    if hasattr(wcs_object, "_pixel_keep"):
        return wcs_object._pixel_keep
    return np.arange(wcs_object.pixel_n_dim)
