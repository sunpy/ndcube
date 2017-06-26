# -*- coding: utf-8 -*-
# Author: Mateo Inchaurrandieta <mateo.inchaurrandieta@gmail.com>
# pylint: disable=E1101
'''Miscellaneous WCS utilities'''
import re
from copy import deepcopy

import numpy as np

from astropy import wcs
from astropy.wcs._wcs import InconsistentAxisTypesError

class WCS(wcs.WCS):

    def __init__(self, header=None, naxis=None, **kwargs):
        self.oriented = False
        if WCS._needs_augmenting(header):
            self.was_augmented = True
            header = WCS._augment(header, naxis)
            if naxis is not None:
                naxis = naxis + 1
        else:
            self.was_augmented = False
        wcs.WCS.__init__(self, header=header, naxis=naxis, **kwargs)

    @classmethod
    def _needs_augmenting(cls, header):
        try:
            wcs.WCS(header=header)
        except InconsistentAxisTypesError as err:
            if re.search(r'Unmatched celestial axes', err.message):
                return True
        return False

    @classmethod
    def _augment(cls, header, naxis):
        newheader = deepcopy(header)
        new_wcs_axes_params = {'CRPIX': 0, 'CDELT': 1, 'CRVAL': 0,
                               'CNAME': 'redundant axis', 'CTYPE': 'HPLN-TAN',
                               'CROTA': 0, 'CUNIT': 'deg', 'NAXIS': 0}
        axis = max(newheader.get('NAXIS', 0), naxis) + 1
        axis = str(axis)
        for param in new_wcs_axes_params:
            attr = new_wcs_axes_params[param]
            newheader[param + axis] = attr
        try:
            wcs.WCS(header=newheader).get_axis_types()
        except InconsistentAxisTypesError as err:
            projection = re.findall(r'expected [^,]+', err.message)[0][9:]
            header['CTYPE' + axis] = projection
        return newheader


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
        projection = re.findall(r'expected [^,]+', err.message)[0][9:]
        outwcs.wcs.ctype[-1] = projection

    return outwcs
