"""
This file contains a set of common fixtures to get a set of different but
predicable NDCube objects.
"""
import logging

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS

from ndcube import ExtraCoords, GlobalCoords, NDCube, NDCubeSequence

# Force MPL to use non-gui backends for testing.
try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    HAVE_MATPLOTLIB = False
else:
    HAVE_MATPLOTLIB = True
    matplotlib.use('Agg')


console_logger = logging.getLogger()
console_logger.setLevel('INFO')

################################################################################
# Helper Functions
################################################################################


def skycoord_2d_lut(shape):
    total_len = np.product(shape)
    data = (np.arange(total_len).reshape(shape),
            np.arange(total_len, total_len * 2).reshape(shape))
    return SkyCoord(*data, unit=u.deg)


def data_nd(shape):
    nelem = np.product(shape)
    return np.arange(nelem).reshape(shape)


def time_extra_coords(shape, axis, base):
    return ExtraCoords.from_lookup_tables(
        ('time',),
        (axis,),
        (base + TimeDelta([i * 60 for i in range(shape[axis])], format='sec'),))


def gen_ndcube_3d_l_ln_lt_ectime(wcs_3d_lt_ln_l, time_axis, time_base, global_coords=None):
    shape = (10, 5, 8)
    wcs_3d_lt_ln_l.array_shape = shape
    data_cube = data_nd(shape)
    mask = data_cube < 0
    extra_coords = time_extra_coords(shape, time_axis, time_base)
    cube = NDCube(data_cube,
                  wcs_3d_lt_ln_l,
                  mask=mask,
                  uncertainty=data_cube)
    cube._extra_coords = extra_coords

    if global_coords:
        cube._global_coords = global_coords

    return cube


################################################################################
# WCS Fixtures
################################################################################


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
def wcs_4d_lt_t_l_ln():
    header = {
        'CTYPE1': 'HPLT-TAN',
        'CUNIT1': 'arcsec',
        'CDELT1': 20,
        'CRPIX1': 0,
        'CRVAL1': 0,

        'CTYPE2': 'TIME    ',
        'CUNIT2': 'min',
        'CDELT2': 0.4,
        'CRPIX2': 0,
        'CRVAL2': 0,

        'CTYPE3': 'WAVE    ',
        'CUNIT3': 'Angstrom',
        'CDELT3': 0.2,
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
def wcs_3d_lt_ln_l():
    header = {

        'CTYPE1': 'HPLN-TAN',
        'CUNIT1': 'arcsec',
        'CDELT1': 10,
        'CRPIX1': 0,
        'CRVAL1': 0,

        'CTYPE2': 'HPLT-TAN',
        'CUNIT2': 'arcsec',
        'CDELT2': 5,
        'CRPIX2': 5,
        'CRVAL2': 0,

        'CTYPE3': 'WAVE    ',
        'CUNIT3': 'Angstrom',
        'CDELT3': 0.2,
        'CRPIX3': 0,
        'CRVAL3': 10,
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
        'CNAME1': 'spectral',
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


################################################################################
# Extra and Global Coords Fixtures
################################################################################


@pytest.fixture
def simple_extra_coords_3d():
    return ExtraCoords.from_lookup_tables(('time', 'hello', 'bye'),
                                          (0, 1, 2),
                                          (list(range(2)) * u.pix,
                                           list(range(3)) * u.pix,
                                           list(range(4)) * u.pix
                                           )
                                          )


################################################################################
# NDCube Fixtures
################################################################################

@pytest.fixture
def ndcube_4d_ln_l_t_lt(wcs_4d_lt_t_l_ln):
    shape = (5, 10, 12, 8)
    wcs_4d_lt_t_l_ln.array_shape = shape
    data_cube = data_nd(shape)
    return NDCube(data_cube, wcs=wcs_4d_lt_t_l_ln)


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
def ndcube_4d_extra_coords(wcs_4d_t_l_lt_ln, simple_extra_coords_3d):
    shape = (5, 8, 10, 12)
    data_cube = data_nd(shape)
    cube = NDCube(data_cube, wcs=wcs_4d_t_l_lt_ln)
    cube._extra_coords = simple_extra_coords_3d
    return cube


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
    cube = NDCube(
        data,
        wcs_3d_l_lt_ln,
        mask=mask,
        uncertainty=data,
    )
    cube._extra_coords = simple_extra_coords_3d
    return cube


@pytest.fixture
def ndcube_3d_rotated(wcs_3d_ln_lt_t_rotated, simple_extra_coords_3d):
    data_rotated = np.array([[[1, 2, 3, 4, 6], [2, 4, 5, 3, 1], [0, -1, 2, 4, 2], [3, 5, 1, 2, 0]],
                             [[2, 4, 5, 1, 3], [1, 5, 2, 2, 4], [2, 3, 4, 0, 5], [0, 1, 2, 3, 4]]])
    mask_rotated = data_rotated >= 0
    cube = NDCube(
        data_rotated,
        wcs_3d_ln_lt_t_rotated,
        mask=mask_rotated,
        uncertainty=data_rotated,
    )
    cube._extra_coords = simple_extra_coords_3d
    return cube


@pytest.fixture
def ndcube_3d_l_ln_lt_ectime(wcs_3d_lt_ln_l):
    return gen_ndcube_3d_l_ln_lt_ectime(wcs_3d_lt_ln_l,
                                        1,
                                        Time('2000-01-01', format='fits', scale='utc'))

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


################################################################################
# NDCubeSequence Fixtures
################################################################################


@pytest.fixture
def ndcubesequence_4c_ln_lt_l(ndcube_3d_ln_lt_l):
    cube1 = ndcube_3d_ln_lt_l
    cube2 = ndcube_3d_ln_lt_l
    cube3 = ndcube_3d_ln_lt_l
    cube4 = ndcube_3d_ln_lt_l
    cube2.data[:] *= 2
    cube3.data[:] *= 3
    cube4.data[:] *= 4
    return NDCubeSequence([cube1, cube2, cube3, cube4])


@pytest.fixture
def ndcubesequence_4c_ln_lt_l_cax1(ndcube_3d_ln_lt_l):
    cube1 = ndcube_3d_ln_lt_l
    cube2 = ndcube_3d_ln_lt_l
    cube3 = ndcube_3d_ln_lt_l
    cube4 = ndcube_3d_ln_lt_l
    cube2.data[:] *= 2
    cube3.data[:] *= 3
    cube4.data[:] *= 4
    return NDCubeSequence([cube1, cube2, cube3, cube4], common_axis=1)


@pytest.fixture
def ndcubesequence_3c_l_ln_lt_cax1(wcs_3d_lt_ln_l):
    common_axis = 1

    base_time1 = Time('2000-01-01', format='fits', scale='utc')
    gc1 = GlobalCoords()
    gc1.add('distance', 'custom:distance', 1*u.m)
    cube1 = gen_ndcube_3d_l_ln_lt_ectime(wcs_3d_lt_ln_l, 1, base_time1, gc1)

    shape = cube1.data.shape
    base_time2 = base_time1 + TimeDelta([shape[common_axis] * 60], format='sec')
    gc2 = GlobalCoords()
    gc2.add('distance', 'custom:distance', 2*u.m)
    gc2.add('global coord', 'custom:physical_type', 0*u.pix)
    cube2 = gen_ndcube_3d_l_ln_lt_ectime(wcs_3d_lt_ln_l, 1, base_time2, gc2)
    cube2.data[:] *= 2

    base_time3 = base_time2 + TimeDelta([shape[common_axis] * 60], format='sec')
    gc3 = GlobalCoords()
    gc3.add('distance', 'custom:distance', 3*u.m)
    cube3 = gen_ndcube_3d_l_ln_lt_ectime(wcs_3d_lt_ln_l, 1, base_time3, gc3)
    cube3.data[:] *= 3

    return NDCubeSequence([cube1, cube2, cube3], common_axis=common_axis)


def pytest_runtest_teardown(item):
    # Clear the pyplot figure stack if it is not empty after the test
    # You can see these log messages by passing "-o log_cli=true" to pytest on the command line
    if HAVE_MATPLOTLIB and plt.get_fignums():
        console_logger.info(f"Removing {len(plt.get_fignums())} pyplot figure(s) "
                            f"left open by {item.name}")
        plt.close('all')
