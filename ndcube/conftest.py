"""
This file contains a set of common fixtures to get a set of different but
predictable NDCube objects.
"""
import logging

import dask.array
import numpy as np
import pytest
from gwcs import coordinate_frames as cf
from gwcs import wcs

import astropy.nddata
import astropy.units as u
from astropy import coordinates as coord
from astropy.coordinates import SkyCoord
from astropy.modeling import models
from astropy.nddata import StdDevUncertainty
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

def time_lut(shape):
    base_time = Time('2000-01-01', format='fits', scale='utc')
    timestamps = Time([base_time + TimeDelta(60 * i, format='sec') for i in range(shape[0])])
    return timestamps

def skycoord_2d_lut(shape):
    total_len = np.prod(shape)
    data = (np.arange(total_len).reshape(shape),
            np.arange(total_len, total_len * 2).reshape(shape))
    return SkyCoord(*data, unit=u.deg)


def data_nd(shape, dtype=float):
    nelem = np.prod(shape)
    return np.arange(nelem, dtype=dtype).reshape(shape)


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
    meta = {"message": "hello world"}
    unit = u.ph
    extra_coords = time_extra_coords(shape, time_axis, time_base)
    cube = NDCube(data_cube,
                  wcs_3d_lt_ln_l,
                  mask=mask,
                  uncertainty=data_cube,
                  meta=meta,
                  unit=unit)
    cube._extra_coords = extra_coords

    if global_coords:
        cube._global_coords = global_coords

    return cube


################################################################################
# WCS Fixtures
################################################################################

@pytest.fixture
def gwcs_4d_t_l_lt_ln():
    """
    Creates a 4D GWCS object with time, wavelength, and celestial coordinates.

    - Time: Axis 0
    - Wavelength: Axis 1
    - Sky: Axes 2 and 3

    Returns:
        wcs.WCS: 4D GWCS object.
    """

    time_model = models.Identity(1)
    time_frame = cf.TemporalFrame(axes_order=(0, ), unit=u.s,
                                  reference_frame=Time("2000-01-01T00:00:00"))

    wave_frame = cf.SpectralFrame(axes_order=(1, ), unit=u.m, axes_names=('wavelength',))
    wave_model = models.Scale(0.2)

    shift  = models.Shift(-5) & models.Shift(0)
    scale  = models.Scale(5) & models.Scale(20)
    tan = models.Pix2Sky_TAN()
    celestial_rotation = models.RotateNative2Celestial(0, 0, 180)
    cel_model = shift | scale | tan | celestial_rotation
    sky_frame = cf.CelestialFrame(axes_order=(2, 3), name='icrs',
                                    reference_frame=coord.ICRS(),
                                    axes_names=("longitude", "latitude"))

    transform = time_model & wave_model & cel_model

    frame = cf.CompositeFrame([time_frame, wave_frame, sky_frame])
    detector_frame = cf.CoordinateFrame(name="detector", naxes=4,
                                        axes_order=(0, 1, 2, 3),
                                        axes_type=("pixel", "pixel", "pixel", "pixel"),
                                        unit=(u.pix, u.pix, u.pix, u.pix))

    return (wcs.WCS(forward_transform=transform, output_frame=frame, input_frame=detector_frame))

@pytest.fixture
def gwcs_3d_lt_ln_l():
    """
    Creates a 3D GWCS object with celestial coordinates and wavelength.

    - Sky: Axes 0 and 1
    - Wavelength: Axis 2

    Returns:
        wcs.WCS: 3D GWCS object.
    """

    shift  = models.Shift(-5) & models.Identity(1)
    scale  = models.Scale(5) & models.Scale(10)
    tan = models.Pix2Sky_TAN()
    celestial_rotation = models.RotateNative2Celestial(0, 0, 180)
    cel_model = shift | scale | tan | celestial_rotation
    sky_frame = cf.CelestialFrame(axes_order=(0, 1), name='icrs',
                                    reference_frame=coord.ICRS(),
                                    axes_names=("longitude", "latitude"))

    wave_model = models.Identity(1) | models.Scale(0.2) | models.Shift(10)
    wave_frame = cf.SpectralFrame(axes_order=(2, ), unit=u.nm, axes_names=("wavelength",))

    transform = cel_model & wave_model

    frame = cf.CompositeFrame([sky_frame, wave_frame])
    detector_frame = cf.CoordinateFrame(name="detector", naxes=3,
                                        axes_order=(0, 1, 2),
                                        axes_type=("pixel", "pixel", "pixel"),
                                        axes_names=("x", "y", "z"), unit=(u.pix, u.pix, u.pix))

    return (wcs.WCS(forward_transform=transform, output_frame=frame, input_frame=detector_frame))

@pytest.fixture
def gwcs_3d_ln_lt_t_rotated():
    """
    Creates a 3D GWCS object with celestial coordinates and wavelength, including rotation.

    - Sky: Axes 0 and 1
    - Wavelength: Axis 2

    Returns:
        wcs.WCS: 3D GWCS object with rotation.
    """
    shift  = models.Shift(-5) & models.Identity(1)
    scale  = models.Scale(5) & models.Scale(10)
    matrix = np.array([[1.290551569736E-05, 5.9525007864732E-06],
                    [5.0226382102765E-06 , -1.2644844123757E-05]])
    rotation = models.AffineTransformation2D(matrix)
    tan = models.Pix2Sky_TAN()
    celestial_rotation = models.RotateNative2Celestial(0, 0, 180)
    cel_model = shift | scale| rotation | tan | celestial_rotation
    sky_frame = cf.CelestialFrame(axes_order=(0, 1), name='icrs',
                                    reference_frame=coord.ICRS(),
                                    axes_names=("longitude", "latitude"))

    wave_model = models.Identity(1) | models.Scale(0.2) | models.Shift(10)
    wave_frame = cf.SpectralFrame(axes_order=(2, ), unit=u.nm, axes_names=("wavelength",))

    transform = cel_model & wave_model

    frame = cf.CompositeFrame([sky_frame, wave_frame])
    detector_frame = cf.CoordinateFrame(name="detector", naxes=3,
                                        axes_order=(0, 1, 2),
                                        axes_type=("pixel", "pixel", "pixel"),
                                        axes_names=("x", "y", "z"), unit=(u.pix, u.pix, u.pix))

    return (wcs.WCS(forward_transform=transform, output_frame=frame, input_frame=detector_frame))

@pytest.fixture
def gwcs_2d_lt_ln():
    """
    Creates a 2D GWCS object with celestial coordinates.

    - Sky: Axes 0 and 1

    Returns:
        wcs.WCS: 2D GWCS object.
    """
    shift  = models.Shift(-5) & models.Shift(-5)
    scale  = models.Scale(2) & models.Scale(4)
    tan = models.Pix2Sky_TAN()
    celestial_rotation = models.RotateNative2Celestial(0, 0, 180)
    cel_model = shift | scale | tan | celestial_rotation
    input_frame = cf.Frame2D(name="detector", axes_names=("x", "y"))
    sky_frame = cf.CelestialFrame(axes_order=(0, 1), name='icrs',
                                    reference_frame=coord.ICRS(),
                                    axes_names=("longitude", "latitude"))

    return (wcs.WCS(forward_transform=cel_model, output_frame=sky_frame, input_frame=input_frame))

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
        'CUNIT3': 's',
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


@pytest.fixture
def time_and_simple_extra_coords_2d():
    return ExtraCoords.from_lookup_tables(("time", "hello"),
                                          (0, 1),
                                          (Time(["2000-01-01T12:00:00", "2000-01-02T12:00:00"],
                                                scale="utc", format="fits"),
                                           list(range(3)) * u.pix)
                                          )


@pytest.fixture
def extra_coords_3d():
    coord0 = Time(["2000-01-01T12:00:00", "2000-01-02T12:00:00"], scale="utc", format="fits")
    coord1 = list(range(3)) * u.pix
    coord2 = list(range(4)) * u.m
    return ExtraCoords.from_lookup_tables(('time', 'bye', 'hello'),
                                          (0, 1, 2),
                                          (coord0, coord1, coord2)
                                          )


@pytest.fixture
def extra_coords_sharing_axis():
    return ExtraCoords.from_lookup_tables(('hello', 'bye'),
                                          (1, 1),
                                          (list(range(3)) * u.m,
                                           list(range(3)) * u.keV,
                                           )
                                          )


################################################################################
# NDCube Fixtures
################################################################################

@pytest.fixture
def ndcube_gwcs_4d_ln_lt_l_t(gwcs_4d_t_l_lt_ln):
    shape = (5, 8, 10, 12)
    gwcs_4d_t_l_lt_ln.array_shape = shape
    data_cube = data_nd(shape)
    return NDCube(data_cube, wcs=gwcs_4d_t_l_lt_ln)

@pytest.fixture
def ndcube_gwcs_3d_ln_lt_l(gwcs_3d_lt_ln_l):
    shape = (2, 3, 4)
    gwcs_3d_lt_ln_l.array_shape = shape
    data_cube = data_nd(shape)
    return NDCube(data_cube, wcs=gwcs_3d_lt_ln_l)

@pytest.fixture
def ndcube_gwcs_3d_rotated(gwcs_3d_lt_ln_l, simple_extra_coords_3d):
    data_rotated = np.array([[[1, 2, 3, 4, 6], [2, 4, 5, 3, 1], [0, -1, 2, 4, 2], [3, 5, 1, 2, 0]],
                             [[2, 4, 5, 1, 3], [1, 5, 2, 2, 4], [2, 3, 4, 0, 5], [0, 1, 2, 3, 4]]])
    cube = NDCube(
        data_rotated,
        wcs=gwcs_3d_lt_ln_l)
    cube._extra_coords = simple_extra_coords_3d
    return cube

@pytest.fixture
def ndcube_gwcs_3d_ln_lt_l_ec_dropped_dim(gwcs_3d_lt_ln_l, time_and_simple_extra_coords_2d):
    shape = (2, 3, 4)
    gwcs_3d_lt_ln_l.array_shape = shape
    data_cube = data_nd(shape)
    cube =  NDCube(data_cube, wcs=gwcs_3d_lt_ln_l)
    cube._extra_coords = time_and_simple_extra_coords_2d[0]
    return cube

@pytest.fixture
def ndcube_gwcs_3d_ln_lt_l_ec_q_t_gc(gwcs_3d_lt_ln_l):
    shape = (3, 3, 4)
    gwcs_3d_lt_ln_l.array_shape = shape
    data_cube = data_nd(shape)
    cube =  NDCube(data_cube, wcs=gwcs_3d_lt_ln_l)
    coord1 = 1 * u.m
    cube.global_coords.add('name1', 'custom:physical_type1', coord1)
    cube.extra_coords.add("time", 0, time_lut(shape))
    cube.extra_coords.add("exposure_lut", 1, range(shape[1]) * u.s)
    return cube

@pytest.fixture
def ndcube_gwcs_2d_ln_lt_mask(gwcs_2d_lt_ln):
    shape = (10, 12)
    data_cube = data_nd(shape)
    mask = np.zeros(shape, dtype=bool)
    mask[1, 1] = True
    mask[2, 0] = True
    mask[3, 3] = True
    mask[4:6, :4] = True
    return NDCube(data_cube, wcs=gwcs_2d_lt_ln, mask=mask)

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
    data_cube = data_nd(shape, dtype=int)
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
    cube._extra_coords._ndcube = cube
    return cube


@pytest.fixture
def ndcube_3d_ln_lt_l_ec_all_axes(wcs_3d_l_lt_ln, extra_coords_3d):
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
    cube._extra_coords = extra_coords_3d
    cube._extra_coords._ndcube = cube
    return cube


@pytest.fixture
def ndcube_3d_ln_lt_l_ec_sharing_axis(wcs_3d_l_lt_ln, extra_coords_sharing_axis):
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
    cube._extra_coords = extra_coords_sharing_axis
    cube._extra_coords._ndcube = cube
    return cube


@pytest.fixture
def ndcube_3d_ln_lt_l_ec_time(wcs_3d_l_lt_ln, time_and_simple_extra_coords_2d):
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
    cube._extra_coords = time_and_simple_extra_coords_2d
    cube._extra_coords._ndcube = cube
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
def ndcube_2d_ln_lt_uncert(wcs_2d_lt_ln):
    shape = (10, 12)
    data_cube = data_nd(shape)
    uncertainty = astropy.nddata.StdDevUncertainty(data_cube * 0.1)
    cube = NDCube(data_cube, wcs=wcs_2d_lt_ln, uncertainty=uncertainty)
    return cube


@pytest.fixture
def ndcube_2d_ln_lt_mask_uncert(wcs_2d_lt_ln):
    shape = (10, 12)
    data_cube = data_nd(shape)
    uncertainty = astropy.nddata.StdDevUncertainty(data_cube * 0.1)
    mask = np.zeros(shape, dtype=bool)
    mask[1, 1] = True
    mask[2, 0] = True
    mask[3, 3] = True
    mask[4:6, :4] = True
    cube = NDCube(data_cube, wcs=wcs_2d_lt_ln, uncertainty=uncertainty, mask=mask)
    return cube


@pytest.fixture
def ndcube_2d_ln_lt_uncert_ec(wcs_2d_lt_ln):
    shape = (4, 9)
    data_cube = data_nd(shape)
    uncertainty = astropy.nddata.StdDevUncertainty(data_cube * 0.1)
    cube = NDCube(data_cube, wcs=wcs_2d_lt_ln, uncertainty=uncertainty)
    cube.extra_coords.add(
        "time", 0,
        Time("2000-01-01 00:00", scale="utc") + TimeDelta(np.arange(shape[0])*60, format="sec"))
    return cube


@pytest.fixture
def ndcube_2d_ln_lt_units(wcs_2d_lt_ln):
    shape = (10, 12)
    data_cube = data_nd(shape).astype(float)
    return NDCube(data_cube, wcs=wcs_2d_lt_ln, unit=u.ct)


@pytest.fixture
def ndcube_2d_dask(wcs_2d_lt_ln):
    shape = (8, 4)
    chunks = 2
    data = data_nd(shape).astype(float)
    da = dask.array.asarray(data, chunks=chunks)
    mask = np.zeros(shape, dtype=bool)
    da_mask = dask.array.asarray(mask, chunks=chunks)
    uncert = data * 0.1
    da_uncert = StdDevUncertainty(dask.array.asarray(uncert, chunks=chunks))
    return NDCube(da, wcs=wcs_2d_lt_ln, uncertainty=da_uncert, mask=da_mask, unit=u.J)


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
    return NDCube(data_cube, wcs=wcs_1d_l,
                  uncertainty=StdDevUncertainty(data_cube*0.1), unit=u.J)


@pytest.fixture(params=[
    "ndcube_4d_ln_lt_l_t",
    "ndcube_4d_uncertainty",
    "ndcube_4d_mask",
    "ndcube_4d_extra_coords",
    "ndcube_4d_unit_uncertainty",
    "ndcube_3d_ln_lt_l",
    "ndcube_3d_rotated",
    "ndcube_2d_ln_lt",
    "ndcube_2d_ln_lt_units",
    "ndcube_2d_dask",
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
