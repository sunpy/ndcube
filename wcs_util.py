# -*- coding: utf-8 -*-
# Author: Mateo Inchaurrandieta <mateo.inchaurrandieta@gmail.com>
# pylint: disable=E1101
'''Miscellaneous WCS utilities'''
import re
from copy import deepcopy

import numpy as np

from astropy import wcs
from astropy.wcs._wcs import InconsistentAxisTypesError
from sunpycube.cube import cube_utils as cu


class WCS(wcs.WCS):

    def __init__(self, header=None, naxis=None, **kwargs):
        self.oriented = False
        self.was_augmented = WCS._needs_augmenting(header)
        if self.was_augmented:
            header = WCS._augment(header, naxis)
            if naxis is not None:
                naxis = naxis + 1
        super(WCS, self).__init__(header=header, naxis=naxis, **kwargs)
        self._bool_sliced = [False]*self.wcs.naxis

    @classmethod
    def _needs_augmenting(cls, header):
        """
        WCS cannot be created with only one spacial dimension. If
        WCS detects that returns that it needs to be augmented.
        """
        try:
            wcs.WCS(header=header)
        except InconsistentAxisTypesError as err:
            if re.search(r'Unmatched celestial axes', str(err)):
                return True
        return False

    @classmethod
    def _augment(cls, header, naxis):
        newheader = deepcopy(header)
        new_wcs_axes_params = {'CRPIX': 0, 'CDELT': 1, 'CRVAL': 0,
                               'CNAME': 'redundant axis', 'CTYPE': 'HPLN-TAN',
                               'CROTA': 0, 'CUNIT': 'deg', 'NAXIS': 0}
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


def _wcs_slicer(wcs, item):
    # normal slice.
    if isinstance(item, slice):
        new_wcs = wcs.slice((item))
    # item is int then slicing axis.
    elif isinstance(item, int):
        if item < wcs._naxis[-1] and item >= 0:
            new_wcs = wcs.slice((slice(item, item+1)))
            new_wcs._bool_sliced[0] = True
        elif item < 0:
            new_wcs = wcs.slice(slice(0, 1))
            new_wcs._bool_sliced[0] = True
        else:
            raise ValueError(
                "indexed value {0} is out of bounds for axis {1} with size {2}".format(item, 0, wcs._naxis[-1]))
    # if it a tuple like [0:2, 0:3, 2] or [0:2, 1:3]
    elif isinstance(item, tuple):
        # if all are slices
        if _all_slice(item):
            new_wcs = wcs.slice((item))
        # if all are not slices some of them are int then
        else:
            # axis to change to 1
            # searching all indexes to change
            item_ = _slice_list(item, wcs._naxis[::-1])
            new_wcs = wcs.slice(item_)
            for i, it in enumerate(item):
                if isinstance(it, int):
                    new_wcs._bool_sliced[i] = True
    return new_wcs


def _all_slice(obj):
    """
    Returns True if all the elements in the object are slices else return False
    """
    result = False
    if not isinstance(obj, (tuple, list)):
        return result
    result |= all(isinstance(o, slice) for o in obj)
    return result


def _slice_list(obj, _naxis):
    """
    Return list of all the slices.
    Example
    -------
    >>> _slice_list((slice(1,2), slice(1,3), 2, slice(2,4), 8))
    >>> (slice(1,2), slice(1,3), slice(2, 3), slice(2,4), slice(8, 9))
    """
    result = []
    if not isinstance(obj, (tuple, list)):
        return result
    for i, o in enumerate(obj):
        if isinstance(o, int):
            if o < _naxis[i] and o >= 0:
                result.append(slice(o, o+1))
            elif o < 0:
                result.append(slice(0, 1))
            else:
                raise ValueError(
                    "indexed value {0} is out of bounds for axis {1} with size {2}".format(o, i, _naxis[i]))
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


def add_celestial_axis(wcs):
    '''
    Creates a copy of the given wcs and returns it, with an extra meaningless
    celestial axes to allow for certain operations. The given WCS must already
    have an unmatched celestial axis.

    Parameters
    ----------
    wcs: sunpy.wcs.wcs.WCS object
        The world coordinate system to add an axis to.
    '''
    outwcs = WCS(naxis=wcs.naxis + 1)
    wcs_params_to_preserve = ['cel_offset', 'dateavg', 'dateobs', 'equinox',
                              'latpole', 'lonpole', 'mjdavg', 'mjdobs', 'name',
                              'obsgeo', 'phi0', 'radesys', 'restfrq',
                              'restwav', 'specsys', 'ssysobs', 'ssyssrc',
                              'theta0', 'velangl', 'velosys', 'zsource']
    for par in wcs_params_to_preserve:
        setattr(outwcs.wcs, par, getattr(wcs.wcs, par))

    new_wcs_axes_params = {'crpix': [0], 'cdelt': [1], 'crval': [0],
                           'cname': ['redundant axis'], 'ctype': ['HPLN-TAN'],
                           'crota': [0], 'cunit': ['deg']}

    try:
        naxis = wcs.naxis
        oldpc = wcs.wcs.pc
        newpc = np.eye(naxis + 1)
        newpc[:naxis, :naxis] = oldpc
        outwcs.wcs.pc = newpc
    except AttributeError:
        pass

    for param in new_wcs_axes_params:
        try:
            oldattr = list(getattr(wcs.wcs, param))
            newattr = oldattr + new_wcs_axes_params[param]
            setattr(outwcs.wcs, param, newattr)
        except AttributeError:  # Some attributes may not be present. Ignore.
            pass

    # Change the projection if we have two redundant celestial axes.
    try:
        outwcs.get_axis_types()
    except InconsistentAxisTypesError as err:
        projection = re.findall(r'expected [^,]+', err.value.args[0])[0][9:]
        outwcs.wcs.ctype[-1] = projection

    return outwcs
