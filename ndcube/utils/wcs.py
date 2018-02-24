# -*- coding: utf-8 -*-
# Author: Ankit Baruah and Daniel Ryan <ryand5@tcd.ie>

"""Miscellaneous WCS utilities"""

import re
from copy import deepcopy
from collections import UserDict

import numpy as np
from astropy import wcs
from astropy.wcs._wcs import InconsistentAxisTypesError

__all__ = ['WCS', 'reindex_wcs', 'wcs_ivoa_mapping', 'get_dependent_axes',
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
    "WAVE": "em.wl"
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


def _wcs_slicer(wcs, missing_axis, item):
    """
    Returns the new sliced wcs and changed missing axis.

    Paramters
    ---------
    wcs: `astropy.wcs.WCS` or `ndcube.utils.wcs.WCS`
        WCS object to be sliced.

    missing_axis: `list` of `bool`
        Indicates which axes of the WCS are "missing", i.e. do not correspond to a data axis.
        Note that unlike in other places in this package, missing_axis has the same axis
        ordering as the WCS object, i.e. the reverse of the data order.

    item: `int`, `slice` or `tuple` of `int` and/or `slice`.
        Slicing item.  Note that unlike in other places in this package, the item has the
        same axis ordering as the WCS object, i.e. the reverse of the data order.

    Returns
    -------
    new_wcs: `astropy.wcs.WCS` or `ndcube.utils.wcs.WCS`
        Sliced WCS object.

    missing_axis: `list` of `bool`
        Altered missing axis list.  Note the ordering has been reversed to reflect the data
        (numpy) axis ordering convention.

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
    elif isinstance(item, int) or isinstance(item, np.int64):
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


def get_dependent_axes(wcs_object, axis, missing_axis):
    # Given an axis number in numpy ordering, returns the axes whose
    # WCS translations are dependent, including itself.  Again,
    # returned axes are in numpy ordering convention.
    # Copied from WCSCoordinates class in glue-viz/glue github repo.

    # TODO: we should cache this

    # if distorted, all bets are off
    try:
        if any([wcs_object.sip, wcs_object.det2im1, wcs_object.det2im2]):
            return tuple(range(wcs_object.naxis))
    except AttributeError:
        pass

    # here, axis is the index number in numpy convention
    # we flip with [::-1] because WCS and numpy index
    # conventions are reversed
    pc = np.array(wcs_object.wcs.get_pc()[::-1, ::-1])
    ndim = pc.shape[0]
    pc[np.eye(ndim, dtype=np.bool)] = 0
    axes = wcs_object.get_axis_types()
    axes = np.array(axes)[np.invert(missing_axis)][::-1]

    # axes rotated.  In a departure from where this was copied,
    # ensure any missing axes are not returned.
    if pc[axis, :].any() or pc[:, axis].any():
        return tuple(np.arange(ndim)[np.invert(missing_axis[::-1])])

    # XXX can spectral still couple with other axes by this point??
    if axes[axis].get('coordinate_type') != 'celestial':
        return (axis,)

    # in some cases, even the celestial coordinates are
    # independent. We don't catch that here.
    return tuple(i for i, a in enumerate(axes) if a.get('coordinate_type') == 'celestial')


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
