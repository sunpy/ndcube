from __future__ import absolute_import, division, unicode_literals, print_function

'''
This module contains prototype code to convert FITS WCS representations into
GWCS equivalent representations.
'''

import numpy as np
from astropy import units as u
from astropy import coordinates as coord
import astropy.modeling.rotations as rotations
import astropy.modeling.projections as projections
from astropy.modeling.mappings import Identity, Mapping
from astropy.modeling.functional_models import Shift, Const1D
from astropy.modeling.polynomial import Polynomial2D
from gwcs import wcs as ggwcs
from gwcs import coordinate_frames as cf


def assign_coefficients(poly2d, coeff):
    '''
    Given coefficients in a nxn array, map them to the relevant parameters
    in a 2d polynomial model
    '''
    # Check that orders are consistent
    coeff_degree = coeff.shape[1]-1
    poly_degree = poly2d.degree
    if coeff_degree != poly_degree:
        raise ValueError("shape of coefficient array inconsistent with polynomial model")
    # parse the poly model coeff names
    cnames = poly2d.param_names
    indices = [item[1:].split('_') for item in cnames]
    indices1 = [int(item[0]) for item in indices]
    indices2 = [int(item[1]) for item in indices]
    poly2d.parameters = coeff[indices1, indices2]


def convert_wcs(fitswcs):
    '''
    Accepts an astropy.wcs wcs object (based on the FITS standard).
    Returns the GWCS equivalent WCS object.
    '''
    # this first version only handles non distorted ra,dec tangent projections
    # check that it is that case
    radesys_dict = {
        'ICRS': coord.ICRS,
        'FK5': coord.FK5,
        'FK4': coord.FK4,
    }
    projection_dict = {
        'TAN': projections.Pix2Sky_TAN(),
        'SIN': projections.Pix2Sky_SIN()
    }
    breakpoint()
    fctypes = fitswcs.wcs.ctype
    fcrval = fitswcs.wcs.crval
    fcrpix = fitswcs.wcs.crpix
    if fitswcs.naxis != 2:
        raise ValueError("currently only handles 2d images")
    print(fctypes)
    ptypes = [ct[5:8] for ct in fctypes]
    for ptype in ptypes:
        print(ptype)
        if ptype not in ['TAN', 'SIN']:
            raise ValueError("currently only supports TAN and SIN projections")
        tptype = ptype # temporary since this part is only for celestial coordinates
    if fitswcs.cpdis1 or fitswcs.cpdis2: ### error here
        raise ValueError("currently doesn't support distortion")

    # Check for SIP correction
    fsip = fitswcs.sip
    if fsip:
        sipa = Polynomial2D(fsip.a_order)
        sipb = Polynomial2D(fsip.b_order)
        assign_coefficients(sipa, fsip.a)
        assign_coefficients(sipb, fsip.b)
        # now the inverse, if it exists
        if fsip.ap_order and fsip.bp_order:
            sipap = Polynomial2D(fsip.ap_order)
            sipbp = Polynomial2D(fsip.bp_order)
            assign_coefficients(sipap, fsip.ap)
            assign_coefficients(sipbp, fsip.bp)
        else:
            sipap = None
        siptrans = Identity(2) + (Mapping((0, 1, 0, 1)) | (sipa & sipb))
        if sipap:
            siptrans.inverse = Identity(2) + (Mapping((0, 1, 0, 1)) | (sipap & sipbp))
    # construct transformation
    if fitswcs.wcs.has_cd():
        trans1 = (Shift(-fcrpix[0]) & Shift(-fcrpix[1]))
        trans2 = (projections.AffineTransformation2D(fitswcs.wcs.cd) |
                  projection_dict[tptype] |
                  rotations.RotateNative2Celestial(fcrval[0], fcrval[1], 180.))
    elif fitswcs.wcs.has_pc():
        trans1 = (Shift(-fcrpix[0]) & Shift(-fcrpix[1]))
        pcmatrix = np.array(fitswcs.wcs.cdelt) * fitswcs.wcs.pc
        trans2 = (projections.AffineTransformation2D(pcmatrix) |
                  projection_dict[tptype] |
                  rotations.RotateNative2Celestial(fcrval[0], fcrval[1], 180.))
    else:
        cdelt = fitswcs.wcs.cdelt
        crota2 = fitswcs.wcs.crota[1] * np.pi / 180 # unware of any crota1 case
        pscale_ratio = cdelt[1] / cdelt[0]
        pcmatrix = np.array([[np.cos(crota2)*cdelt[0], -np.sin(crota2)*cdelt[1]],
                             [np.sin(crota2)*cdelt[0], np.cos(crota2)*cdelt[1]]])
        trans1 = (Shift(-fcrpix[0]) & Shift(-fcrpix[1]))
        trans2 = (projections.AffineTransformation2D(pcmatrix) | 
                 projection_dict[tptype] | 
                 rotations.RotateNative2Celestial(fcrval[0], fcrval[1], 180.))
    if fsip:
        trans = trans1 | siptrans | trans2
    else:
        trans = trans1 | trans2
    detector_frame = cf.Frame2D(name="detector", axes_names=('x', 'y'),
                                unit=(u.pix, u.pix))
    # Now see if a standard frame is referenced.
    if fitswcs.wcs.radesys:
        if fitswcs.wcs.radesys in radesys_dict:
            reference_frame = radesys_dict[fitswcs.wcs.radesys]()
            sky_frame = cf.CelestialFrame(reference_frame=reference_frame,
                                          name=fitswcs.wcs.radesys.lower())
    else:
        sky_frame = '' # or None?

    wcsobj = ggwcs.WCS(forward_transform=trans, input_frame=detector_frame,
                       output_frame=sky_frame)
    return wcsobj
