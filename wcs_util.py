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


def _wcs_slicer(wcs, missing_axis, item):
    """
    Returns the new sliced wcs and changed missing axis
    """
    # normal slice.
    item_checked = []
    if isinstance(item, slice):
        index = 0
        # creating a new tuple of slice where if the axis is dead i.e missing
        # then slice(0,1) added else slice(None, None, None) is appended and
        # if the check of missing_axis gives that this is the index where it
        # needs to be appended then it gets appended there.
        for _bool in missing_axis:
            if not _bool:
                if index is not 1:
                    item_checked.append(item)
                    index += 1
                else:
                    item_checked.append(slice(None, None, None))
            else:
                item_checked.append(slice(0, 1))
        new_wcs = wcs.slice((item_checked))
    # item is int then slicing axis.
    elif isinstance(item, int):
        # using index to keep track of whether the int(which is converted to
        # slice(int_value, int_value+1)) is already added or not. It checks
        # the dead axis i.e missing_axis to check if it is dead than slice(0,1)
        # is appended in it. if the index value has reached 1 then the
        # slice(None, None, None) is added.
        index = 0
        for i, _bool in enumerate(missing_axis):
            if not _bool:
                if index is not 1:
                    item_checked.append(slice(item, item+1))
                    missing_axis[i] = True
                    index += 1
                else:
                    item_checked.append(slice(None, None, None))
            else:
                item_checked.append(slice(0, 1))
        new_wcs = wcs.slice(item_checked)
    # if it a tuple like [0:2, 0:3, 2] or [0:2, 1:3]
    elif isinstance(item, tuple):
        # this is used to not exceed the range of the item tuple
        # if the check of the missing_axis which is False if not dead
        # is a success than the the item of the tuple is added one by
        # one and if the end of tuple is reached than slice(None, None, None)
        # is appended.
        index = 0
        for _bool in missing_axis:
            if not _bool:
                if index is not len(item):
                    item_checked.append(item[index])
                    index += 1
                else:
                    item_checked.append(slice(None, None, None))
            else:
                item_checked.append(slice(0, 1))
        # if all are slice in the item tuple
        if _all_slice(item_checked):
            new_wcs = wcs.slice((item_checked))
        # if all are not slices some of them are int then
        else:
            # this will make all the item in item_checked as slice.
            item_ = _slice_list(item_checked)
            new_wcs = wcs.slice(item_)
            for i, it in enumerate(item_checked):
                if isinstance(it, int):
                    missing_axis[i] = True
    # returning the reverse list of missing axis as in the item here was reverse of
    # what was inputed so we had a reverse missing_axis.
    return new_wcs, missing_axis[::-1]


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
    >>> (slice(1,2), slice(1,3), slice(2, 3), slice(2,4), slice(8, 9))
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


def assert_wcs_are_equal(wcs1, wcs2):
    """
    Assert function for testing two wcs object.
    Used in testing NDCube.
    """
    assert list(wcs1.wcs.ctype) == list(wcs2.wcs.ctype)
    assert list(wcs1.wcs.crval) == list(wcs2.wcs.crval)
    assert list(wcs1.wcs.crpix) == list(wcs2.wcs.crpix)
    assert list(wcs1.wcs.cdelt) == list(wcs2.wcs.cdelt)
    assert list(wcs1.wcs.cunit) == list(wcs2.wcs.cunit)
    assert wcs1.wcs.naxis == wcs2.wcs.naxis
