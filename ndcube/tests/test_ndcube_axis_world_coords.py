import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time


def test_axis_world_coords_wave_ec(ndcube_3d_l_ln_lt_ectime):
    cube = ndcube_3d_l_ln_lt_ectime

    coords = cube.axis_world_coords('em.wl')
    assert u.allclose(coords, [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09, 1.10e-09,
                               1.12e-09, 1.14e-09, 1.16e-09, 1.18e-09, 1.20e-09] * u.m)

    coords = cube.axis_world_coords()
    assert len(coords) == 2
    assert isinstance(coords[0], SkyCoord)
    assert coords[0].shape == (5, 8)
    assert isinstance(coords[1], SpectralCoord)
    assert coords[1].shape == (10,)

    coords = cube.axis_world_coords(wcs=cube.combined_wcs)
    assert len(coords) == 3
    assert isinstance(coords[0], SkyCoord)
    assert coords[0].shape == (5, 8)
    assert isinstance(coords[1], SpectralCoord)
    assert coords[1].shape == (10,)
    assert isinstance(coords[2], Time)
    assert coords[2].shape == (5,)

    coords = cube.axis_world_coords(wcs=cube.extra_coords)
    assert len(coords) == 1
    assert isinstance(coords[0], Time)
    assert coords[0].shape == (5,)

    coords = cube.axis_world_coords_values(wcs=cube.extra_coords)
    assert len(coords) == 1
    assert isinstance(coords[0], u.Quantity)
    assert coords[0].shape == (5,)


@pytest.mark.limit_memory("12 MB")
def test_axis_world_coords_wave_coupled_dims(ndcube_3d_coupled):
    cube = ndcube_3d_coupled

    cube.axis_world_coords('em.wl')


@pytest.mark.limit_memory("12 MB")
def test_axis_world_coords_time_coupled_dims(ndcube_3d_coupled_time):
    cube = ndcube_3d_coupled_time

    cube.axis_world_coords('time')


def test_axis_world_coords_empty_ec(ndcube_3d_l_ln_lt_ectime):
    cube = ndcube_3d_l_ln_lt_ectime
    sub_cube = cube[:, 0]

    # slice the cube so extra_coords is empty, and then try and run axis_world_coords
    awc = sub_cube.axis_world_coords(wcs=sub_cube.extra_coords)
    assert awc == ()


@pytest.mark.xfail(reason=">1D Tables not supported")
def test_axis_world_coords_complex_ec(ndcube_4d_ln_lt_l_t):
    cube = ndcube_4d_ln_lt_l_t
    ec_shape = cube.data.shape[1:3]
    data = np.arange(np.prod(ec_shape)).reshape(ec_shape) * u.m / u.s

    # The lookup table has to be in world order so transpose it.
    cube.extra_coords.add('velocity', (2, 1), data.T)

    coords = cube.axis_world_coords(wcs=cube.extra_coords)
    assert len(coords) == 1
    assert isinstance(coords[0], u.Quantity)
    assert u.allclose(coords[0], data)

    coords = cube.axis_world_coords(wcs=cube.combined_wcs)
    assert len(coords) == 4
    assert u.allclose(coords[3], data)


@pytest.mark.parametrize("axes", [[-1], [2], ["em"]])
def test_axis_world_coords_single(axes, ndcube_3d_ln_lt_l):
    coords = ndcube_3d_ln_lt_l.axis_world_coords_values(*axes)
    assert len(coords) == 1
    assert isinstance(coords[0], u.Quantity)
    assert u.allclose(coords[0], [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09] * u.m)

    coords = ndcube_3d_ln_lt_l.axis_world_coords(*axes)
    assert len(coords) == 1
    assert isinstance(coords[0], u.Quantity)
    assert u.allclose(coords[0], [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09] * u.m)


def test_axis_world_coords_combined_wcs(ndcube_3d_wave_lt_ln_ec_time):
    # This replicates a specific NDCube object in visualization.rst
    coords = ndcube_3d_wave_lt_ln_ec_time.axis_world_coords('time', wcs=ndcube_3d_wave_lt_ln_ec_time.combined_wcs)
    assert len(coords) == 1
    assert isinstance(coords[0], Time)
    assert np.all(coords[0] == Time(['2000-01-01T00:00:00.000', '2000-01-01T00:01:00.000', '2000-01-01T00:02:00.000']))

    coords = ndcube_3d_wave_lt_ln_ec_time.axis_world_coords_values('time', wcs=ndcube_3d_wave_lt_ln_ec_time.combined_wcs)
    assert len(coords) == 1
    assert isinstance(coords.time, u.Quantity)
    assert_quantity_allclose(coords.time, [0, 60, 120] * u.second)


@pytest.mark.parametrize("axes", [[-1], [2], ["em"]])
def test_axis_world_coords_single_pixel_corners(axes, ndcube_3d_ln_lt_l):

    # We go from 4 pixels to 6 pixels when we add pixel corners
    coords = ndcube_3d_ln_lt_l.axis_world_coords_values(*axes, pixel_corners=False)
    assert u.allclose(coords[0], [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09] * u.m)

    coords = ndcube_3d_ln_lt_l.axis_world_coords_values(*axes, pixel_corners=True)
    assert u.allclose(coords, [1.01e-09, 1.03e-09, 1.05e-09, 1.07e-09, 1.09e-09] * u.m)

    coords = ndcube_3d_ln_lt_l.axis_world_coords(*axes, pixel_corners=True)
    assert u.allclose(coords, [1.01e-09, 1.03e-09, 1.05e-09, 1.07e-09, 1.09e-09] * u.m)


@pytest.mark.parametrize(("ndc", "item"),
                         [
                             ("ndcube_3d_ln_lt_l", np.s_[0, 0, :]),
                             ("ndcube_3d_ln_lt_l", np.s_[0, 0, ...]),
                         ],
                         indirect=("ndc",))
def test_axis_world_coords_sliced_all_3d(ndc, item):
    coords = ndc[item].axis_world_coords_values()
    assert u.allclose(coords[0], [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09] * u.m)

    coords = ndc[item].axis_world_coords()
    assert u.allclose(coords[0], [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09] * u.m)


@pytest.mark.parametrize(("ndc", "item"),
                         [
                             ("ndcube_4d_ln_lt_l_t", np.s_[0, 0, :, 0]),
                             ("ndcube_4d_ln_lt_l_t", np.s_[0, 0, ..., 0]),
                         ],
                         indirect=("ndc",))
def test_axis_world_coords_sliced_all_4d(ndc, item):
    coords = ndc[item].axis_world_coords_values()

    expected = [2.0e-11, 4.0e-11, 6.0e-11, 8.0e-11, 1.0e-10,
                1.2e-10, 1.4e-10, 1.6e-10, 1.8e-10, 2.0e-10] * u.m

    assert u.allclose(coords, expected)


def test_axis_world_coords_all_4d_split(ndcube_4d_ln_l_t_lt):
    coords = ndcube_4d_ln_l_t_lt.axis_world_coords()
    assert len(coords) == 3
    assert isinstance(coords[0], SkyCoord)
    assert coords[0].shape == (5, 8)

    assert isinstance(coords[1], Time)
    assert coords[1].shape == (12,)

    assert isinstance(coords[2], u.Quantity)
    assert u.allclose(coords[2], [2.0e-11, 4.0e-11, 6.0e-11, 8.0e-11, 1.0e-10,
                                  1.2e-10, 1.4e-10, 1.6e-10, 1.8e-10, 2.0e-10] * u.m)


@pytest.mark.parametrize('wapt', [
    ('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat', 'em.wl'),
    ('custom:pos.helioprojective.lat', 'em.wl'),
    (0, 1),
    (0, 1, 3)
])
def test_axis_world_coords_all_4d_split_sub(ndcube_4d_ln_l_t_lt, wapt):
    coords = ndcube_4d_ln_l_t_lt.axis_world_coords(*wapt)
    assert len(coords) == 2

    assert isinstance(coords[0], SkyCoord)
    assert coords[0].shape == (5, 8)

    assert isinstance(coords[1], u.Quantity)
    assert u.allclose(coords[1], [2.0e-11, 4.0e-11, 6.0e-11, 8.0e-11, 1.0e-10,
                                  1.2e-10, 1.4e-10, 1.6e-10, 1.8e-10, 2.0e-10] * u.m)


def test_axis_world_coords_all(ndcube_3d_ln_lt_l):
    coords = ndcube_3d_ln_lt_l.axis_world_coords()
    assert len(coords) == 2
    assert isinstance(coords[0], u.Quantity)
    assert u.allclose(coords[0], [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09] * u.m)

    assert isinstance(coords[1], SkyCoord)

    assert u.allclose(coords[1].Tx, [[9.99999999, 9.99999999, 9.99999999],
                                     [19.99999994, 19.99999994, 19.99999994]] * u.arcsec)
    assert u.allclose(coords[1].Ty, [[-19.99999991, -14.99999996, -9.99999998],
                                     [-19.99999984, -14.9999999, -9.99999995]] * u.arcsec)


def test_axis_world_coords_wave(ndcube_3d_ln_lt_l):
    coords = ndcube_3d_ln_lt_l.axis_world_coords('em.wl')
    assert len(coords) == 1
    assert isinstance(coords[0], u.Quantity)
    assert u.allclose(coords[0], [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09] * u.m)


@pytest.mark.parametrize('wapt', ['custom:pos.helioprojective.lon',
                                  'custom:pos.helioprojective.lat'])
def test_axis_world_coords_sky(ndcube_3d_ln_lt_l, wapt):
    coords = ndcube_3d_ln_lt_l.axis_world_coords(wapt)
    assert len(coords) == 1

    assert isinstance(coords[0], SkyCoord)

    assert u.allclose(coords[0].Tx, [[9.99999999, 9.99999999, 9.99999999],
                                     [19.99999994, 19.99999994, 19.99999994]] * u.arcsec)
    assert u.allclose(coords[0].Ty, [[-19.99999991, -14.99999996, -9.99999998],
                                     [-19.99999984, -14.9999999, -9.99999995]] * u.arcsec)


def test_axes_world_coords_sky_only(ndcube_2d_ln_lt):
    coords = ndcube_2d_ln_lt.axis_world_coords()

    assert len(coords) == 1
    assert isinstance(coords[0], SkyCoord)
    assert u.allclose(coords[0].Tx[:, 0], [-16, -12, -8, -4, 0, 4, 8,
                                           12, 16, 20] * u.arcsec, atol=1e-5 * u.arcsec)
    assert u.allclose(coords[0].Ty[0, :], [-8, -6, -4, -2, 0, 2, 4, 6, 8, 10,
                                           12, 14] * u.arcsec, atol=1e-5 * u.arcsec)


def test_axis_world_coords_values_all(ndcube_3d_ln_lt_l):
    coords = ndcube_3d_ln_lt_l.axis_world_coords_values()
    assert len(coords) == 3
    assert all(isinstance(c, u.Quantity) for c in coords)

    assert u.allclose(coords[0], [[0.00277778, 0.00277778, 0.00277778],
                                  [0.00555556, 0.00555556, 0.00555556]] * u.deg)
    assert u.allclose(coords[1], [[-0.00555556, -0.00416667, -0.00277778],
                                  [-0.00555556, -0.00416667, -0.00277778]] * u.deg)
    assert u.allclose(coords[2], [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09] * u.m)


def test_axis_world_coords_values_wave(ndcube_3d_ln_lt_l):
    coords = ndcube_3d_ln_lt_l.axis_world_coords_values('em.wl')
    assert len(coords) == 1
    assert isinstance(coords[0], u.Quantity)
    assert u.allclose(coords[0], [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09] * u.m)


def test_axis_world_coords_values_lon(ndcube_3d_ln_lt_l):
    coords = ndcube_3d_ln_lt_l.axis_world_coords_values('custom:pos.helioprojective.lon')
    assert len(coords) == 1
    assert all(isinstance(c, u.Quantity) for c in coords)

    assert u.allclose(coords[0], [[0.00277778, 0.00277778, 0.00277778],
                                  [0.00555556, 0.00555556, 0.00555556]] * u.deg)


def test_axis_world_coords_values_lat(ndcube_3d_ln_lt_l):
    coords = ndcube_3d_ln_lt_l.axis_world_coords_values('custom:pos.helioprojective.lat')
    assert len(coords) == 1
    assert all(isinstance(c, u.Quantity) for c in coords)
    assert u.allclose(coords[0], [[-0.00555556, -0.00416667, -0.00277778],
                                  [-0.00555556, -0.00416667, -0.00277778]] * u.deg)
