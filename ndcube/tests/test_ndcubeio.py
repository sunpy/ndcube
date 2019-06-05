# -*- coding: utf-8 -*-

# Tests to perform for to_hdulist
# Test HDU0
    # Test the meta
    # Test the wcs axes
    # Test the missing_axes
# Test HDU1
    # Test the units
# Test HDU2
    # Test the uncertainty
# Test HDU3
    # Test the mask
# Test HDU4
    # Test the extra_coords
import pytest
import numpy as np

import astropy.units as u
from astropy.io import fits
# Import the NDCube objects from test_ndcube rather than defining them out here
# from ndcube.tests import test_ndcube
from ndcube.mixins.ndcubeio import NDCubeIOMixin
from ndcube import NDCube
from ndcube.utils.wcs import WCS
from ndcube.tests.helpers import comparerecords


# The NDCube objects
data = np.array([[[1, 2, 3, 4], [2, 4, 5, 3], [0, -1, 2, 3]],
                 [[2, 4, 5, 1], [10, 5, 2, 2], [10, 3, 3, 0]]])

hm = {'CTYPE1': 'WAVE    ', 'CUNIT1': 'Angstrom', 'CDELT1': 0.2, 'CRPIX1': 0, 'CRVAL1': 10,
      'NAXIS1': 4,
      'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5,
      'NAXIS2': 3,
      'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 2}
wm = WCS(header=hm, naxis=3)
uncertaintym = data
mask_cubem = data > 0
meta = {"Description": "This is example NDCube metadata."}
cubem = NDCube(
    data,
    wm,
    mask=mask_cubem,
    meta=meta,
    uncertainty=uncertaintym,
    extra_coords=[('time', 0, u.Quantity(range(data.shape[0]), unit=u.pix)),
                  ('hello', 1, u.Quantity(range(data.shape[1]), unit=u.pix)),
                  ('bye', 2, u.Quantity(range(data.shape[2]), unit=u.pix))])


# HDUList object
cubem_hdulist = NDCubeIOMixin.to_hdulist(cubem)

# Testing the HDU0
@pytest.mark.parametrize(
    "header_obj, header_values1, header_values2, header_values3, header_values4, header_values5, missing_axes",[
        ((cubem_hdulist[0].header),(3, 0.0, 2.0, 2.0),(2E-11, 0.5, 0.4), ('m','deg','deg'),('WAVE','HPLT-TAN','HPLN-TAN'),(1E-09, 0.5, 1.0),(0, 0, 0))
    ]
)
def test_meta_from_hdulist(header_obj, header_values1, header_values2, header_values3, header_values4, header_values5, missing_axes):
    wcs_axes, crpix1, crpix2, crpix3 = header_values1
    cdelt1, cdelt2, cdelt3 = header_values2
    cunit1, cunit2, cunit3 = header_values3
    ctype1, ctype2, ctype3 = header_values4
    crval1, crval2, crval3 = header_values5
    misng1, misng2, misng3 = missing_axes
    # breakpoint()
    
    assert isinstance(header_obj, fits.header.Header)

    # assert the header values
    assert header_obj['WCSAXES'] == wcs_axes
    
    assert header_obj['CRPIX1'] == crpix1
    assert header_obj['CRPIX2'] == crpix2
    assert header_obj['CRPIX3'] == crpix3
    
    assert header_obj['CDELT1'] == cdelt1
    assert header_obj['CDELT2'] == cdelt2
    assert header_obj['CDELT3'] == cdelt3
    
    assert header_obj['CUNIT1'] == cunit1
    assert header_obj['CUNIT2'] == cunit2
    assert header_obj['CUNIT3'] == cunit3

    assert header_obj['CTYPE1'] == ctype1
    assert header_obj['CTYPE2'] == ctype2
    assert header_obj['CTYPE3'] == ctype3

    assert header_obj['CRVAL1'] == crval1
    assert header_obj['CRVAL2'] == crval2
    assert header_obj['CRVAL3'] == crval3

    assert header_obj['MISNG0'] == misng1
    assert header_obj['MISNG1'] == misng2
    assert header_obj['MISNG2'] == misng3

# Testing HDU1
@pytest.mark.parametrize(
    "header_obj, xtension, naxis, bunit, extname",[((cubem_hdulist[1].header),'IMAGE',(3, 4, 3, 2),'','DATA')
    ]
)
def test_data_from_hdulist(header_obj, xtension, naxis, bunit, extname):

    naxis, naxis1, naxis2, naxis3 = naxis

    assert isinstance(header_obj, fits.header.Header)

    assert header_obj['XTENSION'] == xtension
    assert header_obj['NAXIS'] == naxis
    assert header_obj['NAXIS1'] == naxis1
    assert header_obj['NAXIS2'] == naxis2
    assert header_obj['NAXIS3'] == naxis3
    assert header_obj['BUNIT'] == bunit
    assert header_obj['EXTNAME'] == extname


# Testing HDU2
@pytest.mark.parametrize(
    "header_obj, xtension, naxis, utype, extname",[((cubem_hdulist[2].header),'IMAGE',(3, 4, 3, 2),'unknown','UNCERT')
    ]
)
def test_uncert_from_hdulist(header_obj, xtension, naxis, utype, extname):

    naxis, naxis1, naxis2, naxis3 = naxis

    assert isinstance(header_obj, fits.header.Header)

    assert header_obj['XTENSION'] == xtension
    assert header_obj['NAXIS'] == naxis
    assert header_obj['NAXIS1'] == naxis1
    assert header_obj['NAXIS2'] == naxis2
    assert header_obj['NAXIS3'] == naxis3
    assert header_obj['UTYPE'] == utype
    assert header_obj['EXTNAME'] == extname


# Testing HDU3
@pytest.mark.parametrize(
    "header_obj, xtension, naxis, extname",[((cubem_hdulist[3].header),'IMAGE',(3, 4, 3, 2),'MASK')
    ]
)
def test_mask_from_hdulist(header_obj, xtension, naxis, extname):

    naxis, naxis1, naxis2, naxis3 = naxis

    assert isinstance(header_obj, fits.header.Header)

    assert header_obj['XTENSION'] == xtension
    assert header_obj['NAXIS'] == naxis
    assert header_obj['NAXIS1'] == naxis1
    assert header_obj['NAXIS2'] == naxis2
    assert header_obj['NAXIS3'] == naxis3
    assert header_obj['EXTNAME'] == extname

# Testing HDU4
@pytest.mark.parametrize(
    "header_obj, xtension, extrcr, extname",[((cubem_hdulist[4].header),'BINTABLE',('time', 'hello', 'bye'),'EXTRA_COORDS')
    ]
)
def test_extra_coords_from_hdulist(header_obj, xtension, extrcr, extname):

    extrcr1, extrcr2, extrcr3 = extrcr
    assert isinstance(header_obj, fits.header.Header)

    assert header_obj['XTENSION'] == xtension
    assert header_obj['EXTCR0'] == extrcr1
    assert header_obj['EXTCR1'] == extrcr2
    assert header_obj['EXTCR2'] == extrcr3
    assert header_obj['EXTNAME'] == extname


# Test the data
# The Primary header has no data so need to test it

# Test the data of HDU1
@pytest.mark.parametrize(
    "hdu_obj, data",[(
        (cubem_hdulist[1]),(np.array([[[ 1,  2,  3,  4],
                                       [ 2,  4,  5,  3],
                                       [ 0, -1,  2,  3]],

                                      [[ 2,  4,  5,  1],
                                       [10,  5,  2,  2],
                                       [10,  3,  3,  0]]]))
    )]
)
def test_data_of_hdulist_unit(hdu_obj, data):

    assert isinstance(hdu_obj, fits.hdu.image.ImageHDU)
    assert np.array_equal(hdu_obj.data, data)

# Test the data of HDU2
@pytest.mark.parametrize(
    "hdu_obj, data",[(
        (cubem_hdulist[2]),(np.array([[[ 1,  2,  3,  4],
                                       [ 2,  4,  5,  3],
                                       [ 0, -1,  2,  3]],

                                      [[ 2,  4,  5,  1],
                                       [10,  5,  2,  2],
                                       [10,  3,  3,  0]]]))
    )]
)
def test_data_of_hdulist_uncert(hdu_obj, data):

    assert isinstance(hdu_obj, fits.hdu.image.ImageHDU)
    assert np.array_equal(hdu_obj.data, data)

# Test the data of HDU3
@pytest.mark.parametrize(
    "hdu_obj, data",[(
        (cubem_hdulist[3]),(np.array([[[1, 1, 1, 1],
                                       [1, 1, 1, 1],
                                       [0, 0, 1, 1]],

                                     [[1, 1, 1, 1],
                                      [1, 1, 1, 1],
                                      [1, 1, 1, 0]]], dtype=np.uint8))
    )]
)
def test_data_of_hdulist_mask(hdu_obj, data):

    assert isinstance(hdu_obj, fits.hdu.image.ImageHDU)
    assert np.array_equal(hdu_obj.data, data)

# Test the data of HDU4
@pytest.mark.parametrize(
    "hdu_obj, data",[(
        (cubem_hdulist[4]),(np.array([([0, 1], [0, 1, 2], [0, 1, 2, 3])], 
                            dtype=([('time', '<i4', (2,)), ('hello', '<i4', (3,)),
                                    ('bye', '<i4', (4,))])))
    )]
)
def test_data_of_hdulist_mask(hdu_obj, data):

    assert isinstance(hdu_obj, fits.hdu.table.BinTableHDU)
    assert comparerecords(hdu_obj.data, data.view(fits.FITS_rec))


