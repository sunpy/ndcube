"""
This file contains a set of common fixtures to get a set of different but
predicable NDCube objects.
"""
import datetime

import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS

from ndcube import NDCube


@pytest.fixture
def wcs_4d_t_l_lt_ln():
    header = {
        'CTYPE1': 'TIME    ',
        'CUNIT1': 'min',
        'CDELT1': 0.4,
        'CRPIX1': 0,
        'CRVAL1': 0,

        'CTYPE2': 'WAVE    ',
        'CUNIT2': 'Angstrom',
        'CDELT2': 0.2,
        'CRPIX2': 0,
        'CRVAL2': 0,

        'CTYPE3': 'HPLT-TAN',
        'CUNIT3': 'arcsec',
        'CDELT3': 20,
        'CRPIX3': 0,
        'CRVAL3': 0,

        'CTYPE4': 'HPLN-TAN',
        'CUNIT4': 'arcsec',
        'CDELT4': 5,
        'CRPIX4': 5,
        'CRVAL4': 0,

        'DATEREF': "2020-01-01T00:00:00"
        }
    return WCS(header=header)


@pytest.fixture
def wcs_3d_l_lt_ln():
    header = {
        'CTYPE1': 'WAVE    ',
        'CUNIT1': 'Angstrom',
        'CDELT1': 0.2,
        'CRPIX1': 0,
        'CRVAL1': 10,

        'CTYPE2': 'HPLT-TAN',
        'CUNIT2': 'arcsec',
        'CDELT2': 5,
        'CRPIX2': 5,
        'CRVAL2': 0,

        'CTYPE3': 'HPLN-TAN',
        'CUNIT3': 'arcsec',
        'CDELT3': 10,
        'CRPIX3': 0,
        'CRVAL3': 0,
        }

    return WCS(header=header)


@pytest.fixture
def wcs_2d_lt_ln():
    spatial = {
        'CTYPE1': 'HPLT-TAN',
        'CUNIT1': 'arcsec',
        'CDELT1': 2,
        'CRPIX1': 5,
        'CRVAL1': 0,

        'CTYPE2': 'HPLN-TAN',
        'CUNIT2': 'arcsec',
        'CDELT2': 4,
        'CRPIX2': 5,
        'CRVAL2': 0,
    }
    return WCS(header=spatial)

@pytest.fixture
def wcs_1d_l():
    spatial = {
        'CTYPE1': 'WAVE',
        'CUNIT1': 'nm',
        'CDELT1': 0.5,
        'CRPIX1': 2,
        'CRVAL1': 0.5,
    }
    return WCS(header=spatial)


@pytest.fixture
def wcs_3d_ln_lt_t_rotated():
    h_rotated = {
        'CTYPE1': 'HPLN-TAN',
        'CUNIT1': 'arcsec',
        'CDELT1': 0.4,
        'CRPIX1': 0,
        'CRVAL1': 0,
        'NAXIS1': 5,

        'CTYPE2': 'HPLT-TAN',
        'CUNIT2': 'arcsec',
        'CDELT2': 0.5,
        'CRPIX2': 0,
        'CRVAL2': 0,
        'NAXIS2': 5,

        'CTYPE3': 'TIME    ',
        'CUNIT3': 'seconds',
        'CDELT3': 3,
        'CRPIX3': 0,
        'CRVAL3': 0,
        'NAXIS3': 2,

        'DATEREF': "2020-01-01T00:00:00",

        'PC1_1': 0.714963912964,
        'PC1_2': -0.699137151241,
        'PC1_3': 0.0,
        'PC2_1': 0.699137151241,
        'PC2_2': 0.714963912964,
        'PC2_3': 0.0,
        'PC3_1': 0.0,
        'PC3_2': 0.0,
        'PC3_3': 1.0
    }
    return WCS(header=h_rotated)


@pytest.fixture
def simple_extra_coords_3d():
    data = data_nd((2, 3, 4))
    return [('time', 0, u.Quantity(range(data.shape[0]), unit=u.pix)),
            ('hello', 1, u.Quantity(range(data.shape[1]), unit=u.pix)),
            ('bye', 2, u.Quantity(range(data.shape[2]), unit=u.pix))]


def data_nd(shape):
    nelem = np.product(shape)
    return np.arange(nelem).reshape(shape)


def extra_coords(data_cube):
    return [
        ('time', 0, u.Quantity(range(data_cube.shape[1]), unit=u.s)),
        ('hello', 1, u.Quantity(range(data_cube.shape[2]), unit=u.W)),
        ('bye', 2, u.Quantity(range(data_cube.shape[3]), unit=u.m)),
        ('another time', 2, np.array(
            [datetime.datetime(2000, 1, 1) + datetime.timedelta(minutes=i)
             for i in range(data_cube.shape[2])])),
        ('array coord', 2, np.arange(100, 100 + data_cube.shape[3]))
    ]


@pytest.fixture
def ndcube_4d_ln_lt_l_t(wcs_4d_t_l_lt_ln):
    shape = (5, 8, 10, 12)
    wcs_4d_t_l_lt_ln.array_shape = shape
    data_cube = data_nd(shape)
    return NDCube(data_cube, wcs=wcs_4d_t_l_lt_ln)


@pytest.fixture
def ndcube_4d_uncertainty(wcs_4d_t_l_lt_ln):
    shape = (5, 8, 10, 12)
    data_cube = data_nd(shape)
    uncertainty = np.sqrt(data_cube)
    return NDCube(data_cube, wcs=wcs_4d_t_l_lt_ln, uncertainty=uncertainty)


@pytest.fixture
def ndcube_4d_mask(wcs_4d_t_l_lt_ln):
    shape = (5, 8, 10, 12)
    data_cube = data_nd(shape)
    uncertainty = np.sqrt(data_cube)
    mask = data_cube % 2
    return NDCube(data_cube, wcs=wcs_4d_t_l_lt_ln, uncertainty=uncertainty, mask=mask)


@pytest.fixture
def ndcube_4d_extra_coords(wcs_4d_t_l_lt_ln):
    shape = (5, 8, 10, 12)
    data_cube = data_nd(shape)
    ec = extra_coords(data_cube)
    return NDCube(data_cube, wcs=wcs_4d_t_l_lt_ln, extra_coords=ec)


@pytest.fixture
def ndcube_4d_unit_uncertainty(wcs_4d_t_l_lt_ln):
    shape = (5, 8, 10, 12)
    data_cube = data_nd(shape)
    uncertainty = np.sqrt(data_cube)
    return NDCube(data_cube, wcs=wcs_4d_t_l_lt_ln,
                  unit=u.J, uncertainty=uncertainty)


@pytest.fixture
def ndcube_4d(request):
    """
    This is a meta fixture for parametrizing all the 4D ndcubes.
    """
    return request.getfixturevalue("ndcube_4d_" + request.param)


@pytest.fixture
def ndcube_3d_ln_lt_l(wcs_3d_l_lt_ln, simple_extra_coords_3d):
    shape = (2, 3, 4)
    wcs_3d_l_lt_ln.array_shape = shape
    data = data_nd(shape)
    mask = data > 0
    return NDCube(
        data,
        wcs_3d_l_lt_ln,
        uncertainty=data,
        extra_coords=simple_extra_coords_3d
    )


@pytest.fixture
def ndcube_3d_rotated(wcs_3d_ln_lt_t_rotated, simple_extra_coords_3d):
    data_rotated = np.array([[[1, 2, 3, 4, 6], [2, 4, 5, 3, 1], [0, -1, 2, 4, 2], [3, 5, 1, 2, 0]],
                             [[2, 4, 5, 1, 3], [1, 5, 2, 2, 4], [2, 3, 4, 0, 5], [0, 1, 2, 3, 4]]])
    mask_rotated = data_rotated >= 0
    return NDCube(
        data_rotated,
        wcs_3d_ln_lt_t_rotated,
        mask=mask_rotated,
        uncertainty=data_rotated,
        extra_coords=simple_extra_coords_3d
    )


@pytest.fixture
def ndcube_2d_ln_lt(wcs_2d_lt_ln):
    shape = (10, 12)
    data_cube = data_nd(shape)
    return NDCube(data_cube, wcs=wcs_2d_lt_ln)


@pytest.fixture
def ndcube_2d(request):
    """
    This is a meta fixture for parametrizing all the 2D ndcubes.
    """
    return request.getfixturevalue("ndcube_2d_" + request.param)


@pytest.fixture
def ndcube_1d_l(wcs_1d_l):
    shape = (10,)
    data_cube = data_nd(shape)
    return NDCube(data_cube, wcs=wcs_1d_l)


@pytest.fixture(params=[
    "ndcube_4d_ln_lt_l_t",
    "ndcube_4d_uncertainty",
    "ndcube_4d_mask",
    "ndcube_4d_extra_coords",
    "ndcube_4d_unit_uncertainty",
    "ndcube_3d_ln_lt_l",
    "ndcube_3d_rotated",
    "ndcube_2d_ln_lt",
    "ndcube_1d_l",
])
def all_ndcubes(request):
    """
    All the above ndcube fixtures in order.
    """
    return request.getfixturevalue(request.param)


@pytest.fixture
def ndc(request):
    """
    A fixture for use with indirect to lookup other fixtures.
    """
    return request.getfixturevalue(request.param)
