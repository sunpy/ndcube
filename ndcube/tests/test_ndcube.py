import re
import copy
from inspect import signature
from textwrap import dedent

import dask.array
import numpy as np
import pytest
from specutils import Spectrum1D

import astropy.units as u
import astropy.wcs
from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.io import fits
from astropy.nddata import NDData, StdDevUncertainty, UnknownUncertainty
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time
from astropy.units import UnitsError
from astropy.wcs import WCS
from astropy.wcs.utils import wcs_to_celestial_frame
from astropy.wcs.wcsapi import BaseHighLevelWCS, BaseLowLevelWCS
from astropy.wcs.wcsapi.wrappers import SlicedLowLevelWCS

from ndcube import ExtraCoords, NDCube, NDMeta
from ndcube.tests import helpers
from ndcube.tests.helpers import assert_cubes_equal
from ndcube.utils.exceptions import NDCubeUserWarning


def generate_data(shape):
    data = np.arange(np.prod(shape))
    return data.reshape(shape)


def test_wcs_object(all_ndcubes):
    assert isinstance(all_ndcubes.wcs.low_level_wcs, BaseLowLevelWCS)
    assert isinstance(all_ndcubes.wcs, BaseHighLevelWCS)


@pytest.mark.parametrize(("ndc", "item"),
                         [
                             ("ndcube_3d_ln_lt_l", np.s_[:, :, 0]),
                             ("ndcube_3d_ln_lt_l", np.s_[..., 0]),
                             ("ndcube_3d_ln_lt_l", np.s_[1:2, 1:2, 0]),
                             ("ndcube_4d_ln_lt_l_t", np.s_[:, :, 0, 0]),
                             ("ndcube_4d_ln_lt_l_t", np.s_[..., 0, 0]),
                             ("ndcube_4d_ln_lt_l_t", np.s_[1:2, 1:2, 1, 1]),
                         ],
                         indirect=("ndc",))
def test_slicing_ln_lt(ndc, item):
    sndc = ndc[item]
    assert len(sndc.shape) == 2
    assert set(sndc.wcs.world_axis_physical_types) == {"custom:pos.helioprojective.lat",
                                                       "custom:pos.helioprojective.lon"}
    if sndc.uncertainty is not None:
        assert np.allclose(sndc.data, sndc.uncertainty.array)
    if sndc.mask is not None:
        assert np.allclose(sndc.data > 0, sndc.mask)

    if ndc.extra_coords and ndc.extra_coords.keys():
        ec = sndc.extra_coords
        assert set(ec.keys()) == {"time", "hello"}

    wcs = sndc.wcs
    assert isinstance(wcs, BaseHighLevelWCS)
    assert isinstance(wcs.low_level_wcs, SlicedLowLevelWCS)
    assert wcs.pixel_n_dim == 2
    assert wcs.world_n_dim == 2
    assert np.allclose(wcs.array_shape, sndc.data.shape)
    assert np.allclose(sndc.wcs.axis_correlation_matrix, np.ones(2, dtype=bool))


@pytest.mark.parametrize(("ndc", "item"),
                         [
                             ("ndcube_3d_ln_lt_l", np.s_[0, 0, :]),
                             ("ndcube_3d_ln_lt_l", np.s_[0, 0, ...]),
                             ("ndcube_3d_ln_lt_l", np.s_[1, 1, 1:2]),
                             ("ndcube_4d_ln_lt_l_t", np.s_[0, 0, :, 0]),
                             ("ndcube_4d_ln_lt_l_t", np.s_[0, 0, ..., 0]),
                             ("ndcube_4d_ln_lt_l_t", np.s_[1, 1, 1:2, 1]),
                         ],
                         indirect=("ndc",))
def test_slicing_wave(ndc, item):
    sndc = ndc[item]
    assert len(sndc.shape) == 1
    assert set(sndc.wcs.world_axis_physical_types) == {"em.wl"}
    if sndc.uncertainty is not None:
        assert np.allclose(sndc.data, sndc.uncertainty.array)
    if sndc.mask is not None:
        assert np.allclose(sndc.data > 0, sndc.mask)

    if ndc.extra_coords and ndc.extra_coords.keys():
        ec = sndc.extra_coords
        assert set(ec.keys()) == {"bye"}

    wcs = sndc.wcs
    assert isinstance(wcs, BaseHighLevelWCS)
    assert isinstance(wcs.low_level_wcs, SlicedLowLevelWCS)
    assert wcs.pixel_n_dim == 1
    assert wcs.world_n_dim == 1
    assert np.allclose(wcs.array_shape, sndc.data.shape)
    assert np.allclose(sndc.wcs.axis_correlation_matrix, np.ones(1, dtype=bool))


@pytest.mark.parametrize(("ndc", "item"),
                         [
                             ("ndcube_3d_ln_lt_l", np.s_[0, :, :]),
                             ("ndcube_3d_ln_lt_l", np.s_[0, ...]),
                             ("ndcube_3d_ln_lt_l", np.s_[1, 1:2]),
                             ("ndcube_3d_ln_lt_l", np.s_[1, :, 1:2]),
                             ("ndcube_4d_ln_lt_l_t", np.s_[0, :, :, 0]),
                             ("ndcube_4d_ln_lt_l_t", np.s_[0, ..., 0]),
                             ("ndcube_4d_ln_lt_l_t", np.s_[1, 1:2, 1:2, 1]),
                         ],
                         indirect=("ndc",))
def test_slicing_split_celestial(ndc, item):
    sndc = ndc[item]
    assert len(sndc.shape) == 2
    if sndc.uncertainty is not None:
        assert np.allclose(sndc.data, sndc.uncertainty.array)
    if sndc.mask is not None:
        assert np.allclose(sndc.data > 0, sndc.mask)

    if ndc.extra_coords and ndc.extra_coords.keys():
        ec = sndc.extra_coords
        assert set(ec.keys()) == {"hello", "bye"}

    assert isinstance(sndc.wcs, BaseHighLevelWCS)
    assert isinstance(sndc.wcs.low_level_wcs, SlicedLowLevelWCS)
    wcs = sndc.wcs
    assert wcs.pixel_n_dim == 2
    assert wcs.world_n_dim == 3
    assert np.allclose(wcs.array_shape, sndc.data.shape)
    assert set(wcs.world_axis_physical_types) == {"custom:pos.helioprojective.lat",
                                                  "custom:pos.helioprojective.lon",
                                                  "em.wl"}
    assert np.allclose(wcs.axis_correlation_matrix, np.array([[True, False],
                                                              [False, True],
                                                              [False, True]], dtype=bool))


def test_slicing_preserves_global_coords(ndcube_3d_ln_lt_l):
    ndc = ndcube_3d_ln_lt_l
    ndc.global_coords.add('distance', 'pos.distance', 1 * u.m)
    sndc = ndc[0]
    assert sndc._global_coords._internal_coords == ndc._global_coords._internal_coords


def test_slicing_removed_world_coords(ndcube_3d_ln_lt_l):
    ndc = ndcube_3d_ln_lt_l
    # Run this test without extra coords
    ndc._extra_coords = ExtraCoords()
    lat_key = "custom:pos.helioprojective.lat"
    lon_key = "custom:pos.helioprojective.lon"
    wl_key = "em.wl"
    celestial_key = "helioprojective"

    sndc = ndc[:, 0, :]
    assert sndc.global_coords._all_coords == {}

    sndc = ndc[0, 0, :]
    all_coords = sndc.global_coords._all_coords
    assert isinstance(all_coords[celestial_key][1], SkyCoord)
    assert u.allclose(all_coords[celestial_key][1].Ty, -0.00555556 * u.deg)
    assert u.allclose(all_coords[celestial_key][1].Tx, 0.00277778 * u.deg)
    assert all_coords[celestial_key][0] == (lat_key, lon_key)

    sndc = ndc[:, :, 0]
    all_coords = sndc.global_coords._all_coords
    assert u.allclose(all_coords[wl_key][1], 1.02e-9 * u.m)
    assert all_coords[wl_key][0] == wl_key


def test_slicing_with_meta():
    # Define meta.
    raw_meta = {"salutation": "hello", "name": "world",
                "exposure time": u.Quantity([2] * 4, unit=u.s),
                "pixel response": np.ones((4, 5))}
    axes = {"exposure time": 0, "pixel response": (1, 2)}
    meta = NDMeta(raw_meta, axes=axes)
    # Define data.
    data = np.ones((4, 4, 5))
    # Define WCS transformations in an astropy WCS object.
    wcs = astropy.wcs.WCS(naxis=3)
    wcs.wcs.ctype = 'WAVE', 'HPLT-TAN', 'HPLN-TAN'
    wcs.wcs.cunit = 'Angstrom', 'deg', 'deg'
    wcs.wcs.cdelt = 0.2, 0.5, 0.4
    wcs.wcs.crpix = 0, 2, 2
    wcs.wcs.crval = 10, 0.5, 1
    cube = NDCube(data, wcs=wcs, meta=meta)
    sliced_cube = cube[0, 1:3]
    sliced_meta = sliced_cube.meta
    assert sliced_meta.keys() == meta.keys()
    assert tuple(sliced_meta.axes.keys()) == ("pixel response",)
    assert sliced_meta["salutation"] == meta["salutation"]
    assert (sliced_meta["pixel response"] == meta["pixel response"][1:3]).all()
    assert sliced_meta["exposure time"] == 2 * u.s
    assert cube.meta is meta

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


def test_array_axis_physical_types(ndcube_3d_ln_lt_l):
    expected = [
        ('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat', 'custom:PIXEL'),
        ('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat', 'custom:PIXEL'),
        ('em.wl', 'custom:PIXEL')]
    output = ndcube_3d_ln_lt_l.array_axis_physical_types
    for i in range(len(expected)):
        assert all(physical_type in expected[i] for physical_type in output[i])


def test_crop(ndcube_4d_ln_lt_l_t):
    cube = ndcube_4d_ln_lt_l_t
    intervals = cube.wcs.array_index_to_world([1, 2], [0, 1], [0, 1], [0, 2])
    lower_corner = [coord[0] for coord in intervals]
    upper_corner = [coord[-1] for coord in intervals]
    expected = cube[1:3, 0:2, 0:2, 0:3]
    output = cube.crop(lower_corner, upper_corner)
    helpers.assert_cubes_equal(output, expected)


def test_crop_tuple_non_tuple_input(ndcube_2d_ln_lt):
    cube = ndcube_2d_ln_lt
    frame = astropy.wcs.utils.wcs_to_celestial_frame(cube.wcs)
    lower_corner = SkyCoord(Tx=359.99667, Ty=-0.0011111111, unit="deg", frame=frame)
    upper_corner = SkyCoord(Tx=0.0044444444, Ty=0.0011111111, unit="deg", frame=frame)
    cropped_by_tuples = cube.crop((lower_corner,), (upper_corner,))
    cropped_by_coords = cube.crop(lower_corner, upper_corner)
    helpers.assert_cubes_equal(cropped_by_tuples, cropped_by_coords)


def test_crop_with_nones(ndcube_4d_ln_lt_l_t):
    cube = ndcube_4d_ln_lt_l_t
    lower_corner = [None] * 3
    upper_corner = [None] * 3
    interval0 = cube.wcs.array_index_to_world([1, 2], [0, 1], [0, 1], [0, 2])[0]
    lower_corner[0] = interval0[0]
    upper_corner[0] = interval0[-1]
    expected = cube[:, :, :, 0:3]
    output = cube.crop(lower_corner, upper_corner)
    helpers.assert_cubes_equal(output, expected)


def test_crop_1d_independent(ndcube_4d_ln_lt_l_t):
    cube_1d = ndcube_4d_ln_lt_l_t[0, 0, :, 0]
    wl_range = SpectralCoord([3e-11, 4.5e-11], unit=u.m)
    expected = cube_1d[0:2]
    output = cube_1d.crop([wl_range[0]], [wl_range[-1]])
    helpers.assert_cubes_equal(output, expected)


def test_crop_1d_dependent(ndcube_4d_ln_lt_l_t):
    cube_1d = ndcube_4d_ln_lt_l_t[0, :, 0, 0]
    sky_range = cube_1d.wcs.array_index_to_world([0, 1])
    expected = cube_1d[0:2]
    output = cube_1d.crop([sky_range[0]], [sky_range[-1]])
    helpers.assert_cubes_equal(output, expected)


def test_crop_reduces_dimensionality(ndcube_4d_ln_lt_l_t):
    cube = ndcube_4d_ln_lt_l_t
    point = (None, SpectralCoord([3e-11], unit=u.m), None)
    expected = cube[:, :, 0, :]
    output = cube.crop(point)
    helpers.assert_cubes_equal(output, expected)


def test_crop_keepdims(ndcube_4d_ln_lt_l_t):
    cube = ndcube_4d_ln_lt_l_t
    point = (None, SpectralCoord([3e-11], unit=u.m), None)
    output = cube.crop(point, keepdims=True)
    expected = cube[:, :, 0:1, :]
    assert output.shape == (5, 8, 1, 12)
    helpers.assert_cubes_equal(output, expected)


def test_crop_scalar_valuerror(ndcube_2d_ln_lt):
    cube = ndcube_2d_ln_lt
    frame = astropy.wcs.utils.wcs_to_celestial_frame(cube.wcs)
    point = SkyCoord(Tx=359.99667, Ty=-0.0011111111, unit="deg", frame=frame)
    with pytest.raises(ValueError, match=r'Input points causes cube to be cropped to a single pix'):
        cube.crop(point)


def test_crop_missing_dimensions(ndcube_4d_ln_lt_l_t):
    """Test bbox coordinates not being the same length as cube WCS"""
    cube = ndcube_4d_ln_lt_l_t
    interval0 = cube.wcs.array_index_to_world([1, 2], [0, 1], [0, 1], [0, 2])[0]
    lower_corner = [interval0[0], None]
    upper_corner = [interval0[-1], None]
    with pytest.raises(ValueError, match=r'2 components in point 0 do not match WCS with 3'):
        cube.crop(lower_corner, upper_corner)


def test_crop_mismatch_class(ndcube_4d_ln_lt_l_t):
    """Test bbox coordinates not being the same length as cube WCS"""
    cube = ndcube_4d_ln_lt_l_t
    intervals = cube.wcs.array_index_to_world([1, 2], [0, 1], [0, 1], [0, 2])
    intervals[0] = SpectralCoord([3e-11, 4.5e-11], unit=u.m)
    lower_corner = [coord[0] for coord in intervals]
    upper_corner = [coord[-1] for coord in intervals]
    with pytest.raises(TypeError, match=r"<class .*.SpectralCoord'> of component 0 in point 0 is "
                                        r"incompatible with WCS component time"):
        cube.crop(lower_corner, upper_corner)


def test_crop_by_values(ndcube_4d_ln_lt_l_t):
    cube = ndcube_4d_ln_lt_l_t
    intervals = cube.wcs.array_index_to_world_values([1, 2], [0, 1], [0, 1], [0, 2])
    units = [u.min, u.m, u.deg, u.deg]
    lower_corner = [coord[0] * unit for coord, unit in zip(intervals, units)]
    upper_corner = [coord[-1] * unit for coord, unit in zip(intervals, units)]
    # Ensure some quantities are in units different from each other
    # and those stored in the WCS.
    lower_corner[0] = lower_corner[0].to(units[0])
    lower_corner[-1] = lower_corner[-1].to(units[-1])
    upper_corner[-1] = upper_corner[-1].to(units[-1])
    expected = cube[1:3, 0:2, 0:2, 0:3]
    output = cube.crop_by_values(lower_corner, upper_corner)
    helpers.assert_cubes_equal(output, expected)


def test_crop_by_values_keepdims(ndcube_4d_ln_lt_l_t):
    cube = ndcube_4d_ln_lt_l_t
    intervals = list(cube.wcs.array_index_to_world_values([1, 2], [0], [0, 1], [0, 2]))
    units = [u.min, u.m, u.deg, u.deg]
    lower_corner = [coord[0] * unit for coord, unit in zip(intervals, units)]
    upper_corner = [coord[-1] * unit for coord, unit in zip(intervals, units)]
    expected = cube[1:3, 0:1, 0:2, 0:3]
    output = cube.crop_by_values(lower_corner, upper_corner, keepdims=True)
    assert output.shape == (2, 1, 2, 3)
    helpers.assert_cubes_equal(output, expected)


def test_crop_by_values_with_units(ndcube_4d_ln_lt_l_t):
    intervals = ndcube_4d_ln_lt_l_t.wcs.array_index_to_world_values([1, 2], [0, 1], [0, 1], [0, 2])
    units = [u.min, u.m, u.deg, u.deg]
    lower_corner = [coord[0] for coord in intervals]
    upper_corner = [coord[-1] for coord in intervals]
    lower_corner[0] *= u.min
    upper_corner[0] *= u.min
    lower_corner[1] *= u.m
    upper_corner[1] *= u.m
    lower_corner[2] *= u.deg
    units[0] = None
    expected = ndcube_4d_ln_lt_l_t[1:3, 0:2, 0:2, 0:3]
    output = ndcube_4d_ln_lt_l_t.crop_by_values(lower_corner, upper_corner, units=units)
    helpers.assert_cubes_equal(output, expected)


def test_crop_by_values_with_equivalent_units(ndcube_2d_ln_lt):
    # test cropping when passed units that are not identical to the cube wcs.world_axis_units
    intervals = ndcube_2d_ln_lt.wcs.array_index_to_world_values([0, 3], [1, 6])
    lower_corner = [(coord[0]*u.deg).to(u.arcsec) for coord in intervals]
    upper_corner = [(coord[-1]*u.deg).to(u.arcsec) for coord in intervals]
    expected = ndcube_2d_ln_lt[0:4, 1:7]
    output = ndcube_2d_ln_lt.crop_by_values(lower_corner,  upper_corner)
    helpers.assert_cubes_equal(output, expected)


def test_crop_by_values_with_nones(ndcube_4d_ln_lt_l_t):
    cube = ndcube_4d_ln_lt_l_t
    lower_corner = [None] * 4
    lower_corner[0] = 0.5 * u.min
    upper_corner = [None] * 4
    upper_corner[0] = 1.1 * u.min
    expected = cube[:, :, :, 0:3]
    output = cube.crop_by_values(lower_corner, upper_corner)
    helpers.assert_cubes_equal(output, expected)


def test_crop_by_values_all_nones(ndcube_4d_ln_lt_l_t):
    cube = ndcube_4d_ln_lt_l_t
    lower_corner = [None] * 4
    upper_corner = [None] * 4
    output = cube.crop_by_values(lower_corner, upper_corner)
    helpers.assert_cubes_equal(output, cube)


def test_crop_by_values_valueerror1(ndcube_4d_ln_lt_l_t):
    """Test units not being the same length as the inputs"""
    lower_corner = [None] * 4
    lower_corner[0] = 0.5
    upper_corner = [None] * 4
    upper_corner[0] = 1.1
    with pytest.raises(ValueError, match=r'Units must be None or have same length 4 as corner inp'):
        ndcube_4d_ln_lt_l_t.crop_by_values(lower_corner, upper_corner, units=["m"])


def test_crop_by_values_valueerror2(ndcube_4d_ln_lt_l_t):
    """Test upper and lower coordinates not being the same length"""
    with pytest.raises(ValueError, match=r'All points must have same number of coordinate objects'):
        ndcube_4d_ln_lt_l_t.crop_by_values([0], [1, None])


def test_crop_by_values_missing_dimensions(ndcube_4d_ln_lt_l_t):
    """Test bbox coordinates not being the same length as cube WCS"""
    with pytest.raises(ValueError, match=r'3 dimensions in point 0 do not match WCS with 4'):
        ndcube_4d_ln_lt_l_t.crop_by_values([0, None, None], [1, None, None])


def test_crop_by_values_with_wrong_units(ndcube_4d_ln_lt_l_t):
    intervals = ndcube_4d_ln_lt_l_t.wcs.array_index_to_world_values([1, 2], [0, 1], [0, 1], [0, 2])
    units = [None, u.m, u.km, u.km]
    lower_corner = [coord[0] for coord in intervals]
    upper_corner = [coord[-1] for coord in intervals]
    lower_corner[0] *= u.min
    upper_corner[0] *= u.min
    lower_corner[1] *= u.m
    upper_corner[1] *= u.m
    lower_corner[2] *= u.km
    with pytest.raises(UnitsError, match=r"Unit 'km' of coordinate object 2 in point 0 is "
                                         r"incompatible with WCS unit 'deg'"):
        ndcube_4d_ln_lt_l_t.crop_by_values(lower_corner, upper_corner, units=units)


def test_crop_by_values_1d_dependent(ndcube_4d_ln_lt_l_t):
    cube_1d = ndcube_4d_ln_lt_l_t[0, :, 0, 0]
    lat_range, lon_range = cube_1d.wcs.low_level_wcs.array_index_to_world_values([0, 1])
    lower_corner = [lat_range[0] * u.deg, lon_range[0] * u.deg]
    upper_corner = [lat_range[-1] * u.deg, lon_range[-1] * u.deg]
    expected = cube_1d[0:2]
    output = cube_1d.crop_by_values(lower_corner, upper_corner)
    helpers.assert_cubes_equal(output, expected)


def test_crop_by_extra_coords(ndcube_3d_ln_lt_l_ec_time):
    cube = ndcube_3d_ln_lt_l_ec_time
    lower_corner = (Time("2000-01-01T15:00:00", scale="utc", format="fits"), None)
    upper_corner = (Time("2000-01-01T20:00:00", scale="utc", format="fits"), None)
    output = cube.crop(lower_corner, upper_corner, wcs=cube.extra_coords)
    expected = cube[0]
    helpers.assert_cubes_equal(output, expected)


def test_crop_by_extra_coords_values(ndcube_3d_ln_lt_l_ec_time):
    cube = ndcube_3d_ln_lt_l_ec_time
    lower_corner = (3 * 60 * 60 * u.s, 0 * u.pix)
    upper_corner = (8 * 60 * 60 * u.s, 2 * u.pix)
    output = cube.crop_by_values(lower_corner, upper_corner, wcs=cube.extra_coords)
    expected = cube[0]
    helpers.assert_cubes_equal(output, expected)


def test_crop_by_extra_coords_all_axes_with_coord(ndcube_3d_ln_lt_l_ec_all_axes):
    cube = ndcube_3d_ln_lt_l_ec_all_axes
    interval0 = Time(["2000-01-01T15:00:00", "2000-01-01T20:00:00"], scale="utc", format="fits")
    interval1 = [0, 1] * u.pix
    interval2 = [1, 3] * u.m
    lower_corner = (interval0[0], interval1[0], interval2[0])
    upper_corner = (interval0[1], interval1[1], interval2[1])
    output = cube.crop(lower_corner, upper_corner, wcs=cube.extra_coords)
    expected = cube[0, 0:2, 1:4]
    helpers.assert_cubes_equal(output, expected)


def test_crop_by_extra_coords_values_all_axes_with_coord(ndcube_3d_ln_lt_l_ec_all_axes):
    cube = ndcube_3d_ln_lt_l_ec_all_axes
    interval0 = [3 * 60 * 60, 8 * 60 * 60] * u.s
    interval1 = [0, 1] * u.pix
    interval2 = [1, 3] * u.m
    lower_corner = (interval0[0], interval1[0], interval2[0])
    upper_corner = (interval0[1], interval1[1], interval2[1])
    output = cube.crop_by_values(lower_corner, upper_corner, wcs=cube.extra_coords)
    expected = cube[0, 0:2, 1:4]
    helpers.assert_cubes_equal(output, expected)


def test_crop_by_extra_coords_shared_axis(ndcube_3d_ln_lt_l_ec_sharing_axis):
    cube = ndcube_3d_ln_lt_l_ec_sharing_axis
    lower_corner = (1 * u.m, 1 * u.keV)
    upper_corner = (2 * u.m, 2 * u.keV)
    output = cube.crop(lower_corner, upper_corner, wcs=cube.extra_coords)
    expected = cube[:, 1:3]
    helpers.assert_cubes_equal(output, expected)


def test_crop_by_extra_coords_values_shared_axis(ndcube_3d_ln_lt_l_ec_sharing_axis):
    cube = ndcube_3d_ln_lt_l_ec_sharing_axis
    lower_corner = (1 * u.m, 1 * u.keV)
    upper_corner = (2 * u.m, 2 * u.keV)
    output = cube.crop_by_values(lower_corner, upper_corner, wcs=cube.extra_coords)
    expected = cube[:, 1:3]
    helpers.assert_cubes_equal(output, expected)


def test_crop_rotated_celestial(ndcube_4d_ln_lt_l_t):
    # This is a regression test for a highly rotated image where all 4 corners
    # of the spatial ROI have to be used.

    header = dedent("""\
        WCSAXES =                    2 / Number of coordinate axes
        CRPIX1  =          2053.459961 / Pixel coordinate of reference point
        CRPIX2  =          2047.880005 / Pixel coordinate of reference point
        PC1_1   =     0.70734471922412 / Coordinate transformation matrix element
        PC1_2   =     0.70686876305701 / Coordinate transformation matrix element
        PC2_1   =    -0.70686876305701 / Coordinate transformation matrix element
        PC2_2   =     0.70734471922412 / Coordinate transformation matrix element
        CDELT1  =  0.00016652472222222 / [deg] Coordinate increment at reference point
        CDELT2  =  0.00016652472222222 / [deg] Coordinate increment at reference point
        CUNIT1  = 'deg'                / Units of coordinate increment and value
        CUNIT2  = 'deg'                / Units of coordinate increment and value
        CTYPE1  = 'HPLN-TAN'           / Coordinate type codegnomonic projection
        CTYPE2  = 'HPLT-TAN'           / Coordinate type codegnomonic projection
        CRVAL1  =                  0.0 / [deg] Coordinate value at reference point
        CRVAL2  =                  0.0 / [deg] Coordinate value at reference point
        LONPOLE =                180.0 / [deg] Native longitude of celestial pole
        LATPOLE =                  0.0 / [deg] Native latitude of celestial pole
        MJDREF  =                  0.0 / [d] MJD of fiducial time
        DATE-OBS= '2014-04-09T06:00:12.970' / ISO-8601 time of observation
        MJD-OBS =      56756.250150116 / [d] MJD of observation
        RSUN_REF=          696000000.0 / [m] Solar radius
        DSUN_OBS=      149860273889.04 / [m] Distance from centre of Sun to observer
        HGLN_OBS=  -0.0058904803279347 / [deg] Stonyhurst heliographic lng of observer
        HGLT_OBS=     -6.0489216362492 / [deg] Heliographic latitude of observer
        """)
    wcs = WCS(fits.Header.fromstring(header, sep="\n"))
    data = np.zeros((4096, 4096))

    cube = NDCube(data, wcs=wcs)

    bottom_left = SkyCoord(-100, -100, unit=u.arcsec, frame=wcs_to_celestial_frame(wcs))
    bottom_right = SkyCoord(600, -100, unit=u.arcsec, frame=wcs_to_celestial_frame(wcs))
    top_left = SkyCoord(-100, 600, unit=u.arcsec, frame=wcs_to_celestial_frame(wcs))
    top_right = SkyCoord(600, 600, unit=u.arcsec, frame=wcs_to_celestial_frame(wcs))

    small = cube.crop(bottom_left, bottom_right, top_left, top_right)

    assert small.data.shape == (1652, 1652)


def test_initialize_from_ndcube(ndcube_3d_l_ln_lt_ectime):
    cube = ndcube_3d_l_ln_lt_ectime
    cube.global_coords.add('distance', 'pos.distance', 1 * u.m)
    cube2 = NDCube(cube)

    assert cube.global_coords is cube2.global_coords
    assert cube.extra_coords is cube2.extra_coords

    cube3 = NDCube(cube, copy=True)
    ec = cube.extra_coords
    ec3 = cube3.extra_coords

    assert cube.global_coords == cube3.global_coords
    assert cube.global_coords is not cube3.global_coords
    assert ec.keys() == ec3.keys()
    assert ec.mapping == ec3.mapping
    assert np.allclose(ec.wcs.pixel_to_world_values(1), ec3.wcs.pixel_to_world_values(1))
    assert ec is not ec3


def test_reproject_interpolation(ndcube_4d_ln_l_t_lt, wcs_4d_lt_t_l_ln):
    target_wcs_header = wcs_4d_lt_t_l_ln.low_level_wcs.to_header()
    target_wcs_header['CDELT3'] = 0.1   # original value = 0.2
    target_wcs = astropy.wcs.WCS(header=target_wcs_header)
    shape_out = (5, 20, 12, 8)

    resampled_cube = ndcube_4d_ln_l_t_lt.reproject_to(target_wcs, shape_out=shape_out)

    assert ndcube_4d_ln_l_t_lt.data.shape == (5, 10, 12, 8)
    assert resampled_cube.data.shape == (5, 20, 12, 8)


def test_reproject_invalid_wcs(ndcube_4d_ln_l_t_lt, wcs_3d_lt_ln_l):
    shape_out = (5, 20, 12, 8)

    with pytest.raises(Exception):
        _ = ndcube_4d_ln_l_t_lt.reproject_to(wcs_3d_lt_ln_l, shape_out=shape_out)


def test_reproject_with_header(ndcube_4d_ln_l_t_lt, wcs_4d_lt_t_l_ln):
    target_wcs_header = wcs_4d_lt_t_l_ln.low_level_wcs.to_header()
    shape_out = (5, 20, 12, 8)

    _ = ndcube_4d_ln_l_t_lt.reproject_to(target_wcs_header, shape_out=shape_out)


def test_reproject_return_footprint(ndcube_4d_ln_l_t_lt, wcs_4d_lt_t_l_ln):
    target_wcs_header = wcs_4d_lt_t_l_ln.low_level_wcs.to_header()
    target_wcs_header['CDELT3'] = 0.1   # original value = 0.2
    target_wcs = astropy.wcs.WCS(header=target_wcs_header)
    shape_out = (5, 20, 12, 8)

    resampled_cube, footprint = ndcube_4d_ln_l_t_lt.reproject_to(target_wcs, shape_out=shape_out,
                                                                 return_footprint=True)

    assert ndcube_4d_ln_l_t_lt.data.shape == (5, 10, 12, 8)
    assert resampled_cube.data.shape == (5, 20, 12, 8)
    assert footprint.shape == (5, 20, 12, 8)


def test_reproject_shape_out(ndcube_4d_ln_l_t_lt, wcs_4d_lt_t_l_ln):
    # should raise an exception when neither shape_out is specified nor
    # target_wcs has the pixel_shape or array_shape attribute
    wcs_4d_lt_t_l_ln.pixel_shape = None
    with pytest.raises(Exception):
        _ = ndcube_4d_ln_l_t_lt.reproject_to(wcs_4d_lt_t_l_ln)
    # should not raise an exception when shape_out is specified
    shape = (5, 10, 12, 8)
    _ = ndcube_4d_ln_l_t_lt.reproject_to(wcs_4d_lt_t_l_ln, shape_out=shape)

    # should not raise an exception when target_wcs has pixel_shape or array_shape attribute
    wcs_4d_lt_t_l_ln.array_shape = shape
    _ = ndcube_4d_ln_l_t_lt.reproject_to(wcs_4d_lt_t_l_ln, shape_out=shape)


def test_wcs_type_after_init(ndcube_3d_ln_lt_l, wcs_3d_l_lt_ln):
    # Generate a low level WCS
    slices = np.s_[:, :, 0]
    low_level_wcs = SlicedLowLevelWCS(wcs_3d_l_lt_ln, slices)
    # Generate an NDCube using the low level WCS
    cube = NDCube(ndcube_3d_ln_lt_l.data[slices], low_level_wcs)
    # Check the WCS has been converted to high level but NDCube init.
    assert isinstance(cube.wcs, BaseHighLevelWCS)


@pytest.mark.parametrize("bin_shape", [(10, 2, 1), (-1, 2, 1)])
def test_rebin(ndcube_3d_l_ln_lt_ectime, bin_shape):
    cube = ndcube_3d_l_ln_lt_ectime[:, 1:]
    with pytest.warns(NDCubeUserWarning, match="The uncertainty on this NDCube has no known way to propagate forward"):
        output = cube.rebin(bin_shape, operation=np.sum, propagate_uncertainties=True)
    output_sc, output_spec = output.axis_world_coords(wcs=output.wcs)
    output_time, = output.axis_world_coords(wcs=output.extra_coords)

    # Build expected cube contents.
    expected_data = np.array([[3840, 3860, 3880, 3900, 3920, 3940, 3960, 3980],
                              [4160, 4180, 4200, 4220, 4240, 4260, 4280, 4300]])
    expected_mask = np.zeros(expected_data.shape, dtype=bool)
    expected_uncertainty = None
    expected_unit = cube.unit
    expected_meta = cube.meta
    expected_Tx = np.array([[9.99999999, 19.99999994, 29.99999979, 39.9999995,
                             49.99999902, 59.99999831, 69.99999731, 79.99999599],
                            [9.99999999, 19.99999994, 29.99999979, 39.9999995,
                             49.99999902, 59.99999831, 69.99999731, 79.99999599]]) * u.arcsec
    expected_Ty = np.array([[-14.99999996, -14.9999999, -14.99999981, -14.99999969,
                             -14.99999953, -14.99999934, -14.99999911, -14.99999885],
                            [-4.99999999, -4.99999998, -4.99999995, -4.9999999,
                             -4.99999985, -4.99999979, -4.99999971, -4.99999962]]) * u.arcsec
    expected_spec = SpectralCoord([1.02e-09], unit=u.m)
    expected_time = Time([51544.00104167, 51544.00243056], format="mjd", scale="utc")
    expected_time.format = "fits"

    # Confirm output is as expected.
    assert np.all(output.shape == np.array([1, 2, 8]))
    assert np.all(output.data == expected_data)
    assert np.all(output.mask == expected_mask)
    assert output.uncertainty == expected_uncertainty
    assert output.unit == expected_unit
    assert output.meta == expected_meta
    assert u.allclose(output_sc.Tx, expected_Tx)
    assert u.allclose(output_sc.Ty, expected_Ty)
    assert u.allclose(output_spec, expected_spec)
    assert output_time.scale == expected_time.scale
    assert output_time.format == expected_time.format
    assert np.allclose(output_time.mjd, expected_time.mjd)


def test_rebin_dask(ndcube_2d_dask):
    # Execute rebin.
    cube = ndcube_2d_dask
    bin_shape = (4, 2)
    output = cube.rebin(bin_shape, propagate_uncertainties=True)
    dask_type = dask.array.core.Array
    assert isinstance(output.data, dask_type)
    assert isinstance(output.uncertainty.array, dask_type)
    assert isinstance(output.mask, dask_type)


def test_rebin_bin_shape_quantity(ndcube_3d_l_ln_lt_ectime):
    # Confirm rebin's bin_shape argument handles being a astropy unit
    cube = ndcube_3d_l_ln_lt_ectime[:, 1:]
    cube._extra_coords = ExtraCoords(cube)
    bin_shape = (10, 2, 1) * u.pix
    output = cube.rebin(bin_shape)
    np.testing.assert_allclose(output.shape, cube.shape / bin_shape.to_value())
    with pytest.raises(u.UnitConversionError, match=re.escape("'m' (length) and 'pix' are not convertible")):
        cube.rebin((10, 2, 1) * u.m)


def test_rebin_no_ec(ndcube_3d_l_ln_lt_ectime):
    # Confirm rebin does not try to handle extra coords when there aren't any.
    cube = ndcube_3d_l_ln_lt_ectime[:, 1:]
    cube._extra_coords = ExtraCoords(cube)
    bin_shape = (10, 2, 1)
    with pytest.warns(NDCubeUserWarning, match="The uncertainty on this NDCube has no known way to propagate forward"):
        output = cube.rebin(bin_shape, operation=np.mean, propagate_uncertainties=True)
    assert output.extra_coords.is_empty


def test_rebin_uncerts(ndcube_2d_ln_lt_uncert):
    cube = ndcube_2d_ln_lt_uncert
    bin_shape = (2, 4)
    output = cube.rebin(bin_shape, operation=np.mean, propagate_uncertainties=True)
    output_uncert = output.uncertainty.array
    expected_uncert = (np.array([[2.73495887,  3.68239053,  4.7116876],
                                 [9.07524104, 10.1882285, 11.30486621],
                                 [15.79240324, 16.91744662, 18.0432813],
                                 [22.55216176, 23.68037162, 24.80886938],
                                 [29.32507459, 30.45455631, 31.58417325]])
                       / np.array(bin_shape).prod())
    assert np.allclose(output_uncert, expected_uncert)


def test_rebin_some_masked_uncerts(ndcube_2d_ln_lt_mask_uncert):
    cube = ndcube_2d_ln_lt_mask_uncert
    bin_shape = (2, 4)
    expected_data = np.array([[7.5,  11.5,  15.5],
                              [31.5,  35.5,  39.5],
                              [55.5,  59.5,  63.5],
                              [79.5,  83.5,  87.5],
                              [103.5, 107.5, 111.5]])
    expected_uncert = np.array([[0.34186986, 0.46029882, 0.58896095],
                                [1.13440513, 1.27352856, 1.41310828],
                                [1.9740504, 2.11468083, 2.25541016],
                                [2.81902022, 2.96004645, 3.10110867],
                                [3.66563432, 3.80681954, 3.94802166]])
    expected_mask = np.zeros((5, 3), dtype=bool)
    expected_mask[:3, 0] = True
    # Execute function and assert result is as expected.
    output = cube.rebin(bin_shape, operation=np.mean, operation_ignores_mask=True,
                        handle_mask=np.any, propagate_uncertainties=True)
    assert np.allclose(output.data, expected_data)
    assert np.allclose(output.uncertainty.array[np.logical_not(expected_mask)],
                       expected_uncert[np.logical_not(expected_mask)])
    assert (output.mask == expected_mask).all()


def test_rebin_some_masked_uncerts_exclude_masked_values(ndcube_2d_ln_lt_mask_uncert):
    cube = ndcube_2d_ln_lt_mask_uncert
    bin_shape = (2, 4)
    expected_data = np.array([[6.71428571,  11.5,  15.5],
                              [31.5,  35.5,  39.5],
                              [0.,  59.5,  63.5],
                              [79.5,  83.5,  87.5],
                              [103.5, 107.5, 111.5]])
    expected_uncert = np.array([[0.34374884, 0.46029882, 0.58896095],
                                [1.30586285, 1.27352856, 1.41310828],
                                [0., 2.11468083, 2.25541016],
                                [2.81902022, 2.96004645, 3.10110867],
                                [3.66563432, 3.80681954, 3.94802166]])
    expected_mask = np.zeros((5, 3), dtype=bool)
    expected_mask[2, 0] = True
    # Execute function and assert result is as expected.
    output = cube.rebin(bin_shape, operation=np.mean, operation_ignores_mask=False,
                        handle_mask=np.all, propagate_uncertainties=True)
    assert np.allclose(output.data, expected_data)
    assert np.allclose(output.uncertainty.array[np.logical_not(expected_mask)],
                       expected_uncert[np.logical_not(expected_mask)])
    assert (output.mask == expected_mask).all()


def test_rebin_errors(ndcube_3d_l_ln_lt_ectime):
    cube = ndcube_3d_l_ln_lt_ectime
    # Wrong number of axes in bin_shape)
    with pytest.raises(ValueError):
        cube.rebin((2,))

    # bin_shape not integer multiple of data shape.
    with pytest.raises(ValueError):
        cube.rebin((9, 2, 1))


def test_rebin_no_propagate(ndcube_2d_ln_lt_mask_uncert):
    # Execute rebin.
    cube = ndcube_2d_ln_lt_mask_uncert
    bin_shape = (2, 4)

    cube._mask[:] = True
    with pytest.warns(NDCubeUserWarning, match="Uncertainties cannot be propagated as all values are masked and operation_ignores_mask is False."):
        output = cube.rebin(bin_shape, operation=np.sum, propagate_uncertainties=True,
                            operation_ignores_mask=False)
    assert output.uncertainty is None

    cube._mask = True
    with pytest.warns(NDCubeUserWarning, match="Uncertainties cannot be propagated as all values are masked and operation_ignores_mask is False."):
        output = cube.rebin(bin_shape, operation=np.sum, propagate_uncertainties=True,
                            operation_ignores_mask=False)
    assert output.uncertainty is None

    cube._mask = False
    cube._uncertainty = UnknownUncertainty(cube.data * 0.1)
    with pytest.warns(NDCubeUserWarning, match="The uncertainty on this NDCube has no known way to propagate forward"):
        output = cube.rebin(bin_shape, operation=np.sum, propagate_uncertainties=True)
    assert output.uncertainty is None


def test_rebin_axis_aware_meta(ndcube_4d_axis_aware_meta):
    # Execute rebin.
    cube = ndcube_4d_axis_aware_meta
    bin_shape = (1, 2, 5, 1)
    output = cube.rebin(bin_shape, operation=np.sum)

    # Build expected meta
    expected_meta = copy.deepcopy(cube.meta)
    del expected_meta._axes["pixel label"]
    del expected_meta._axes["line"]
    expected_meta._data_shape = np.array([5, 4, 2, 12], dtype=int)

    # Confirm output meta is as expected.
    helpers.assert_metas_equal(output.meta, expected_meta)


def test_rebin_specutils():
    # Tests for https://github.com/sunpy/ndcube/issues/717
    y = np.arange(4000)*u.ct
    x = np.arange(200, 4200)*u.nm
    spec = Spectrum1D(flux=y, spectral_axis=x, bin_specification='centers', mask=x > 2000*u.nm)
    output = spec.rebin((10,), operation=np.sum, operation_ignores_mask=False)
    assert output.shape == (400,)


def test_reproject_adaptive(ndcube_2d_ln_lt, wcs_2d_lt_ln):
    shape_out = (10, 12)
    resampled_cube = ndcube_2d_ln_lt.reproject_to(wcs_2d_lt_ln, algorithm='adaptive',
                                                  shape_out=shape_out)

    assert ndcube_2d_ln_lt.data.shape == (10, 12)
    assert resampled_cube.data.shape == (10, 12)


def test_reproject_exact(ndcube_2d_ln_lt, wcs_2d_lt_ln):
    shape_out = (10, 12)
    resampled_cube = ndcube_2d_ln_lt.reproject_to(wcs_2d_lt_ln, algorithm='exact',
                                                  shape_out=shape_out)

    assert ndcube_2d_ln_lt.data.shape == (10, 12)
    assert resampled_cube.data.shape == (10, 12)


def test_reproject_invalid_algorithm(ndcube_4d_ln_l_t_lt, wcs_4d_lt_t_l_ln):
    with pytest.raises(ValueError):
        _ = ndcube_4d_ln_l_t_lt.reproject_to(wcs_4d_lt_t_l_ln, algorithm='my_algorithm',
                                             shape_out=(5, 10, 12, 8))


def test_reproject_adaptive_incompatible_wcs(ndcube_4d_ln_l_t_lt, wcs_4d_lt_t_l_ln,
                                             wcs_1d_l, ndcube_1d_l):
    with pytest.raises(ValueError):
        _ = ndcube_1d_l.reproject_to(wcs_1d_l, algorithm='adaptive',
                                     shape_out=(10,))

    with pytest.raises(ValueError):
        _ = ndcube_4d_ln_l_t_lt.reproject_to(wcs_4d_lt_t_l_ln, algorithm='adaptive',
                                             shape_out=(5, 10, 12, 8))


def test_reproject_exact_incompatible_wcs(ndcube_4d_ln_l_t_lt, wcs_4d_lt_t_l_ln,
                                          wcs_1d_l, ndcube_1d_l):
    with pytest.raises(ValueError):
        _ = ndcube_1d_l.reproject_to(wcs_1d_l, algorithm='exact',
                                     shape_out=(10,))

    with pytest.raises(ValueError):
        _ = ndcube_4d_ln_l_t_lt.reproject_to(wcs_4d_lt_t_l_ln, algorithm='exact',
                                             shape_out=(5, 10, 12, 8))


def test_plot_docstring():
    cube = NDCube([], astropy.wcs.WCS())

    assert cube.plot.__doc__ == cube.plotter.plot.__doc__
    assert signature(cube.plot) == signature(cube.plotter.plot)
# This function is used in the arithmetic tests below


def check_arithmetic_value_and_units(cube_new, data_expected):
    cube_quantity = u.Quantity(cube_new.data, cube_new.unit)
    assert u.allclose(cube_quantity, data_expected)


@pytest.mark.parametrize('value', [
    10 * u.ct,
    u.Quantity([10], u.ct),
    u.Quantity(np.random.rand(12), u.ct),
    u.Quantity(np.random.rand(10, 12), u.ct),
])
def test_cube_arithmetic_add(ndcube_2d_ln_lt_units, value): # this test methods aims for the special scenario of only integers being added together.
    cube_quantity = u.Quantity(ndcube_2d_ln_lt_units.data, ndcube_2d_ln_lt_units.unit)
    # Add
    new_cube = ndcube_2d_ln_lt_units + value
    check_arithmetic_value_and_units(new_cube, cube_quantity + value)


# Only one of them has a unit.
# An expected typeError should be raised.
@pytest.mark.parametrize(("ndc", "value"),
                        [
                            ("ndcube_2d_uncertainty_no_unit", NDData(np.ones((10, 12)),
                                                            wcs=None,
                                                            unit=u.m,
                                                            uncertainty=StdDevUncertainty(np.ones((10, 12)) * 0.1))
                            ),
                            ("ndcube_2d_ln_lt_units", NDData(np.ones((10, 12)),
                                                            wcs=None,
                                                            uncertainty=StdDevUncertainty(np.ones((10, 12)) * 0.1))
                            ),
                        ],
                        indirect=("ndc",))
def test_arithmetic_add_one_unit(ndc, value):
    assert isinstance(ndc, NDCube)
    with pytest.raises(TypeError, match="Adding objects requires that both have a unit or neither has a unit."):
        ndc + value


# Both NDData and NDCube have unit and uncertainty. No mask is involved.
# Test different scenarios when units are equivalent and when they are not. TODO (bc somewhere is checking the units are the same)
# what is an equivalent unit in astropy for count (ct)?
@pytest.mark.parametrize(("ndc", "value"),
                        [
                            ("ndcube_2d_unit_unc", NDData(np.ones((10, 12)), # pass in the values to be tested as a set of ones.
                                                          wcs=None,
                                                          unit=u.ct,
                                                          uncertainty=StdDevUncertainty(np.ones((10, 12))*0.1, unit=u.ct))
                            ),
                        ],
                        indirect=("ndc",))
def test_arithmetic_add_cube_unit_unc_nddata_unit_unc(ndc, value):
    output_cube = ndc + value # perform the addition
    # Construct expected cube
    expected_unit = u.ct
    expected_data = ((ndc.data * ndc.unit) + (value.data * value.unit)).to_value(expected_unit)
    expected_uncertainty = ndc.uncertainty.propagate(
                            operation=np.add,
                            other_nddata=value,
                            result_data=expected_data*expected_unit,
                            correlation=0,
    )
    expected_cube = NDCube(expected_data, ndc.wcs, uncertainty=expected_uncertainty, unit=expected_unit)
    # Assert output cube is same as expected cube
    assert_cubes_equal(output_cube, expected_cube, check_uncertainty_values=True)

# Both have unit, NDCube has no uncertainty and NDData has uncertainty.
@pytest.mark.parametrize(("ndc", "value"),
                        [
                            ("ndcube_2d_ln_lt_units", NDData(np.ones((10, 12)), # pass in the values to be tested as a set of ones.
                                                          wcs=None,
                                                          unit=u.ct,
                                                          uncertainty=StdDevUncertainty(np.ones((10, 12))*0.1, unit=u.ct))
                            ),
                        ],
                        indirect=("ndc",))
def test_arithmetic_add_cube_unit_nddata_unit_unc(ndc, value):
    output_cube = ndc + value # perform the addition

    # Construct expected cube
    expected_unit = u.ct
    expected_data = ((ndc.data * ndc.unit) + (value.data * value.unit)).to_value(expected_unit)
    expected_uncertainty = value.uncertainty

    expected_cube = NDCube(expected_data, ndc.wcs, uncertainty=expected_uncertainty, unit=expected_unit)
    # Assert output cube is same as expected cube
    assert_cubes_equal(output_cube, expected_cube, check_uncertainty_values=True)


# Both have units, NDData has no uncertainty and NDCube has uncertainty.
@pytest.mark.parametrize(("ndc", "value"),
                        [
                            ("ndcube_2d_unit_unc", NDData(np.ones((10, 12)), # pass in the values to be tested as a set of ones.
                                                          wcs=None,
                                                          unit=u.ct)
                            ),
                        ],
                        indirect=("ndc",))
def test_arithmetic_add_cube_unit_unc_nddata_unit(ndc, value):
    output_cube = ndc + value # perform the addition

    # Construct expected cube
    expected_unit = u.ct
    expected_data = ((ndc.data * ndc.unit) + (value.data * value.unit)).to_value(expected_unit)
    expected_uncertainty = ndc.uncertainty

    expected_cube = NDCube(expected_data, ndc.wcs, uncertainty=expected_uncertainty, unit=expected_unit)
    # Assert output cube is same as expected cube
    assert_cubes_equal(output_cube, expected_cube, check_uncertainty_values=True)


# Both have units, neither has uncertainty.
@pytest.mark.parametrize(("ndc", "value"),
                        [
                            ("ndcube_2d_ln_lt_units", NDData(np.ones((10, 12)), # pass in the values to be tested as a set of ones.
                                                          wcs=None,
                                                          unit=u.ct)
                            ),
                        ],
                        indirect=("ndc",))
def test_arithmetic_add_cube_unit_nddata_unit(ndc, value):
    output_cube = ndc + value # perform the addition

    # Construct expected cube
    expected_unit = u.ct
    expected_data = ((ndc.data * ndc.unit) + (value.data * value.unit)).to_value(expected_unit)
    expected_cube = NDCube(expected_data, ndc.wcs, unit=expected_unit)

    # Assert output cube is same as expected cube
    assert_cubes_equal(output_cube, expected_cube)


# Neither has a unit, both have uncertainty.
@pytest.mark.parametrize(("ndc", "value"),
                        [
                            ("ndcube_2d_uncertainty_no_unit", NDData(np.ones((10, 12)), # pass in the values to be tested as a set of ones.
                                                                      wcs=None,
                                                                      uncertainty=StdDevUncertainty(np.ones((10, 12))*0.1))
                            ),
                        ],
                        indirect=("ndc",))
def test_arithmetic_add_cube_unc_nddata_unc(ndc, value):
    output_cube = ndc + value # perform the addition
    # Construct expected cube
    expected_data = ndc.data + value.data
    expected_uncertainty = ndc.uncertainty.propagate(  # is this the correct way to test uncertainty? no need to test propagate
                            operation=np.add,
                            other_nddata=value,
                            result_data=expected_data,
                            correlation=0,
    )
    expected_cube = NDCube(expected_data, ndc.wcs, uncertainty=expected_uncertainty)
    # Assert output cube is same as expected cube
    assert_cubes_equal(output_cube, expected_cube, check_uncertainty_values=True)


# Neither has a unit, NDData has uncertainty and NDCube has no uncertainty.
@pytest.mark.parametrize(("ndc", "value"),
                        [
                            ("ndcube_2d_ln_lt_no_unit_no_unc", NDData(np.ones((10, 12)), # pass in the values to be tested as a set of ones.
                                                                      wcs=None,
                                                                      uncertainty=StdDevUncertainty(np.ones((10, 12))*0.1))
                            ),
                        ],
                        indirect=("ndc",))
def test_arithmetic_add_cube_nddata_unc(ndc, value):
    output_cube = ndc + value # perform the addition

    # Construct expected cube
    expected_data = ndc.data + value.data
    expected_uncertainty = value.uncertainty
    expected_cube = NDCube(expected_data, ndc.wcs, uncertainty=expected_uncertainty)

    # Assert output cube is same as expected cube
    assert_cubes_equal(output_cube, expected_cube)


# Neither has a unit, NDData has no uncertainty and NDCube has uncertainty.
@pytest.mark.parametrize(("ndc", "value"),
                        [
                            ("ndcube_2d_uncertainty_no_unit", NDData(np.ones((10, 12)), # pass in the values to be tested as a set of ones.
                                                                      wcs=None)
                            ),
                        ],
                        indirect=("ndc",))
def test_arithmetic_add_cube_unc_nddata(ndc, value):
    output_cube = ndc + value # perform the addition

    # Construct expected cube
    expected_data = ndc.data + value.data
    expected_uncertainty = ndc.uncertainty
    expected_cube = NDCube(expected_data, ndc.wcs, uncertainty=expected_uncertainty)

    # Assert output cube is same as expected cube
    assert_cubes_equal(output_cube, expected_cube)


# Neither has unit or uncertainty.
@pytest.mark.parametrize(("ndc", "value"),
                        [
                            ("ndcube_2d_ln_lt_no_unit_no_unc", NDData(np.ones((10, 12)), # pass in the values to be tested as a set of ones.
                                                                      wcs=None)
                            ),
                        ],
                        indirect=("ndc",))
def test_arithmetic_add_cube_nddata(ndc, value):
    output_cube = ndc + value # perform the addition

    # Construct expected cube
    expected_data = ndc.data + value.data
    expected_cube = NDCube(expected_data, ndc.wcs)

    # Assert output cube is same as expected cube
    assert_cubes_equal(output_cube, expected_cube)


# Test the three different with-mask scenarios for the add method.
# 1, both have masks. To test: data, combined mask, uncertainty
@pytest.mark.parametrize(
    ("value"),
    [(NDData(np.ones((2, 3)),
            wcs=None,
            uncertainty=StdDevUncertainty(np.ones((2, 3)) * 0.05),
            mask=np.ones((2, 3), dtype=bool)))]
)
def test_arithmetic_add_both_mask(ndcube_2d_ln_lt_mask2, value):
    output_cube = ndcube_2d_ln_lt_mask2 + value  # perform the addition

    # Construct expected cube
    expected_data = ndcube_2d_ln_lt_mask2.data + value.data
    expected_uncertainty = ndcube_2d_ln_lt_mask2.uncertainty
    expected_mask = np.ones((2, 3), dtype=bool)
    expected_cube = NDCube(expected_data, ndcube_2d_ln_lt_mask2.wcs, uncertainty=expected_uncertainty, mask=expected_mask)

    # Assert output cube is same as expected cube
    assert_cubes_equal(output_cube, expected_cube)


# Test the three different with-mask scenarios for the add method.
# 2, The NDCube object has masks. To test: data, combined mask, uncertainty
@pytest.mark.parametrize('value', [
    NDData(np.ones((2, 3)),
           wcs=None,
           uncertainty=StdDevUncertainty(np.ones((2, 3)) * 0.05))
])
def test_arithmetic_add_cube_mask(ndcube_2d_ln_lt_mask2, value):
    output_cube = ndcube_2d_ln_lt_mask2 + value  # perform the addition

    # Construct expected cube
    expected_data = ndcube_2d_ln_lt_mask2.data + value.data
    expected_uncertainty = ndcube_2d_ln_lt_mask2.uncertainty
    expected_mask = np.array([[False, True, True],
                              [True, True, True]])
    expected_cube = NDCube(expected_data, ndcube_2d_ln_lt_mask2.wcs, uncertainty=expected_uncertainty, mask=expected_mask)

    # Assert output cube is same as expected cube
    assert_cubes_equal(output_cube, expected_cube)


# Test the three different with-mask scenarios for the add method.
# 1, The NDData object has masks. To test: data, combined mask, uncertainty
@pytest.mark.parametrize('value', [
    NDData(np.ones((2, 3)),
           wcs=None,
           uncertainty=StdDevUncertainty(np.ones((2, 3)) * 0.05),
           mask=np.ones((2, 3), dtype=bool))
])
def test_arithmetic_add_nddata_mask(ndcube_2d_ln_lt_nomask, value):
    output_cube = ndcube_2d_ln_lt_nomask + value  # perform the addition

    # Construct expected cube
    expected_data = ndcube_2d_ln_lt_nomask.data + value.data
    expected_uncertainty = ndcube_2d_ln_lt_nomask.uncertainty
    expected_mask = np.ones((2, 3), dtype=bool)
    expected_cube = NDCube(expected_data, ndcube_2d_ln_lt_nomask.wcs, uncertainty=expected_uncertainty, mask=expected_mask)

    # Assert output cube is same as expected cube
    assert_cubes_equal(output_cube, expected_cube)


@pytest.mark.parametrize('value', [
    10 * u.ct,
    u.Quantity([10], u.ct),
    u.Quantity(np.random.rand(12), u.ct),
    u.Quantity(np.random.rand(10, 12), u.ct),
])
def test_cube_arithmetic_radd(ndcube_2d_ln_lt_units, value):
    cube_quantity = u.Quantity(ndcube_2d_ln_lt_units.data, ndcube_2d_ln_lt_units.unit)
    new_cube = value + ndcube_2d_ln_lt_units
    check_arithmetic_value_and_units(new_cube, value + cube_quantity)


@pytest.mark.parametrize('value', [
    10 * u.ct,
    u.Quantity([10], u.ct),
    u.Quantity(np.random.rand(12), u.ct),
    u.Quantity(np.random.rand(10, 12), u.ct),
])
def test_cube_arithmetic_subtract(ndcube_2d_ln_lt_units, value):
    cube_quantity = u.Quantity(ndcube_2d_ln_lt_units.data, ndcube_2d_ln_lt_units.unit)
    new_cube = ndcube_2d_ln_lt_units - value
    check_arithmetic_value_and_units(new_cube, cube_quantity - value)


@pytest.mark.parametrize('value', [
    10 * u.ct,
    u.Quantity([10], u.ct),
    u.Quantity(np.random.rand(12), u.ct),
    u.Quantity(np.random.rand(10, 12), u.ct),
])
def test_cube_arithmetic_rsubtract(ndcube_2d_ln_lt_units, value):
    cube_quantity = u.Quantity(ndcube_2d_ln_lt_units.data, ndcube_2d_ln_lt_units.unit)
    new_cube = value - ndcube_2d_ln_lt_units
    check_arithmetic_value_and_units(new_cube, value - cube_quantity)


@pytest.mark.parametrize('value', [
    10 * u.ct,
    u.Quantity([10], u.ct),
    u.Quantity(np.random.rand(12), u.ct),
    u.Quantity(np.random.rand(10, 12), u.ct),
    10.0,
    np.random.rand(12),
    np.random.rand(10, 12),
])
def test_cube_arithmetic_multiply(ndcube_2d_ln_lt_units, value):
    cube_quantity = u.Quantity(ndcube_2d_ln_lt_units.data, ndcube_2d_ln_lt_units.unit)
    new_cube = ndcube_2d_ln_lt_units * value
    check_arithmetic_value_and_units(new_cube, cube_quantity * value)
    # TODO: test that uncertainties scale correctly


@pytest.mark.parametrize(("ndc", "value", "expected_kwargs"),
                        [
                            ("ndcube_2d_ln_lt_no_unit_no_unc_no_mask_2", NDData(np.ones((2, 3)),
                                                                                wcs=None),
                             {"data": np.array([[0, 1, 2], [3, 4, 5]])} # neither of the two has uncertainty or mask or unit.
                            ),
                            ("ndcube_2d_ln_lt_no_unit_no_unc_no_mask_2", NDData(np.ones((2, 3)),
                                                                                wcs=None,
                                                                                uncertainty=StdDevUncertainty(np.ones((2, 3))*0.1),
                                                                                mask=np.ones((2, 3), dtype=bool),
                                                                                unit=u.ct),
                             {"data": np.array([[0, 1, 2], [3, 4, 5]]),
                              "uncertainty": astropy.nddata.StdDevUncertainty(np.ones((2, 3))*0.1),
                              "mask": np.ones((2, 3), dtype=bool),
                              "unit": u.ct
                             }# ndc has no mask no uncertainty no unit, but nddata has all.
                            ),
                            ("ndcube_2d_ln_lt_unit_unc_mask", NDData(np.ones((2, 3)),
                                                                     wcs=None,
                                                                     uncertainty=StdDevUncertainty(np.ones((2, 3))*0.1),
                                                                     mask=np.ones((2, 3), dtype=bool),
                                                                     unit=u.ct),
                             {"unit": u.ct**2,
                             "data": np.array([[0, 1, 2], [3, 4, 5]]),
                             "uncertainty": astropy.nddata.StdDevUncertainty(np.array([[0.1118034, 0.1118034, 0.1118034],
                                                                    [0.1118034, 0.1118034, 0.1118034]])),
                             "mask": np.ones((2, 3), dtype=bool)} # both of them have uncertainty and mask and unit.
                            )
                        ],
                        indirect=("ndc",))
def test_cube_arithmetic_multiply_ndcube_nddata(ndc, value, expected_kwargs, wcs_2d_lt_ln):
    output_cube = ndc * value  # perform the multiplication

    expected_kwargs["wcs"] = wcs_2d_lt_ln
    expected_cube = NDCube(**expected_kwargs)

    # Assert output cube is same as expected cube
    assert_cubes_equal(output_cube, expected_cube)


@pytest.mark.parametrize('value', [
    10 * u.ct,
    u.Quantity([10], u.ct),
    u.Quantity(np.random.rand(12), u.ct),
    u.Quantity(np.random.rand(10, 12), u.ct),
    10.0,
    np.random.rand(12),
    np.random.rand(10, 12),
])
def test_cube_arithmetic_rmultiply(ndcube_2d_ln_lt_units, value):
    cube_quantity = u.Quantity(ndcube_2d_ln_lt_units.data, ndcube_2d_ln_lt_units.unit)
    new_cube = value * ndcube_2d_ln_lt_units
    check_arithmetic_value_and_units(new_cube, value * cube_quantity)


@pytest.mark.parametrize('value', [
    10 * u.ct,
    u.Quantity([10], u.ct),
    u.Quantity([2], u.s),
    u.Quantity(np.random.rand(12), u.ct),
    u.Quantity(np.random.rand(10, 12), u.ct),
    10.0,
    np.random.rand(12),
    np.random.rand(10, 12),
])
def test_cube_arithmetic_divide(ndcube_2d_ln_lt_units, value):
    cube_quantity = u.Quantity(ndcube_2d_ln_lt_units.data, ndcube_2d_ln_lt_units.unit)
    new_cube = ndcube_2d_ln_lt_units / value
    check_arithmetic_value_and_units(new_cube, cube_quantity / value)

@pytest.mark.parametrize('value', [1, 2, -1])
def test_cube_arithmetic_rdivide(ndcube_2d_ln_lt_units, value):
    cube_quantity = u.Quantity(ndcube_2d_ln_lt_units.data, ndcube_2d_ln_lt_units.unit)
    with np.errstate(divide='ignore'):
        new_cube =  value / ndcube_2d_ln_lt_units
        check_arithmetic_value_and_units(new_cube,  value / cube_quantity)

@pytest.mark.parametrize('value', [1, 2, -1])
def test_cube_arithmetic_rdivide_uncertainty(ndcube_4d_unit_uncertainty, value):
    cube_quantity = u.Quantity(ndcube_4d_unit_uncertainty.data, ndcube_4d_unit_uncertainty.unit)
    with pytest.warns(NDCubeUserWarning, match="UnknownUncertainty does not support uncertainty propagation with correlation. Setting uncertainties to None."):
        with np.errstate(divide='ignore'):
            new_cube =  value / ndcube_4d_unit_uncertainty
            check_arithmetic_value_and_units(new_cube,  value / cube_quantity)

def test_cube_arithmetic_neg(ndcube_2d_ln_lt_units):
    check_arithmetic_value_and_units(
        -ndcube_2d_ln_lt_units,
        u.Quantity(-ndcube_2d_ln_lt_units.data, ndcube_2d_ln_lt_units.unit),
    )


def test_add_unitless_cube_typeerror(ndcube_2d_ln_lt_units):
    with pytest.raises(TypeError):
        _ = ndcube_2d_ln_lt_units + 10.0


def test_cube_arithmetic_add_notimplementederror(ndcube_2d_ln_lt_units):
    with pytest.raises(TypeError):
        _ = ndcube_2d_ln_lt_units + ndcube_2d_ln_lt_units


def test_cube_arithmetic_multiply_notimplementederror(ndcube_2d_ln_lt_units):
    with pytest.raises(TypeError):
        _ = ndcube_2d_ln_lt_units * ndcube_2d_ln_lt_units



@pytest.mark.parametrize('power', [2, -2, 10, 0.5])
def test_cube_arithmetic_power(ndcube_2d_ln_lt, power):
    cube_quantity = u.Quantity(ndcube_2d_ln_lt.data, ndcube_2d_ln_lt.unit)
    with np.errstate(divide='ignore'):
        new_cube = ndcube_2d_ln_lt ** power
        check_arithmetic_value_and_units(new_cube, cube_quantity**power)


@pytest.mark.parametrize('power', [2, -2, 10, 0.5])
def test_cube_arithmetic_power_unknown_uncertainty(ndcube_4d_unit_uncertainty, power):
    cube_quantity = u.Quantity(ndcube_4d_unit_uncertainty.data, ndcube_4d_unit_uncertainty.unit)
    with pytest.warns(NDCubeUserWarning, match="UnknownUncertainty does not support uncertainty propagation with correlation. Setting uncertainties to None."):
        with np.errstate(divide='ignore'):
            new_cube = ndcube_4d_unit_uncertainty ** power
            check_arithmetic_value_and_units(new_cube, cube_quantity**power)


@pytest.mark.parametrize('power', [2, -2, 10, 0.5])
def test_cube_arithmetic_power_std_uncertainty(ndcube_2d_ln_lt_uncert, power):
    cube_quantity = u.Quantity(ndcube_2d_ln_lt_uncert.data, ndcube_2d_ln_lt_uncert.unit)
    with pytest.warns(NDCubeUserWarning, match=r"<class 'astropy.nddata.nduncertainty.StdDevUncertainty'> does not support propagation of uncertainties for power. Setting uncertainties to None."):
        with np.errstate(divide='ignore'):
            new_cube = ndcube_2d_ln_lt_uncert ** power
            check_arithmetic_value_and_units(new_cube, cube_quantity**power)


@pytest.mark.parametrize('new_unit', [u.mJ, 'mJ'])
def test_to(ndcube_1d_l, new_unit):
    cube = ndcube_1d_l
    expected_factor = 1000
    output = cube.to(new_unit)
    assert np.allclose(output.data, cube.data * expected_factor)
    assert np.allclose(output.uncertainty.array, cube.uncertainty.array * expected_factor)
    assert output.unit == u.Unit(new_unit)


def test_to_dask(ndcube_2d_dask):
    output = ndcube_2d_dask.to(u.mJ)
    dask_type = dask.array.core.Array
    assert isinstance(output.data, dask_type)
    assert isinstance(output.uncertainty.array, dask_type)
    assert isinstance(output.mask, dask_type)


def test_squeeze(ndcube_4d_ln_l_t_lt):
    assert np.array_equal(ndcube_4d_ln_l_t_lt.squeeze().shape, ndcube_4d_ln_l_t_lt.shape)
    assert np.array_equal(ndcube_4d_ln_l_t_lt[:,:,0,:].shape, ndcube_4d_ln_l_t_lt[:,:,0:1,:].squeeze().shape)
    assert np.array_equal(ndcube_4d_ln_l_t_lt[:,:,0,:].shape, ndcube_4d_ln_l_t_lt[:,:,0:1,:].squeeze(2).shape)
    assert np.array_equal(ndcube_4d_ln_l_t_lt[:,0,0,:].shape, ndcube_4d_ln_l_t_lt[:,0:1,0:1,:].squeeze([1,2]).shape)
    assert np.array_equal(ndcube_4d_ln_l_t_lt[:,0:1,0,:].shape, ndcube_4d_ln_l_t_lt[:,0:1,0:1,:].squeeze(2).shape)


def test_squeeze_error(ndcube_4d_ln_l_t_lt):
    same = ndcube_4d_ln_l_t_lt.squeeze()[0:1,:,:,:]
    with pytest.raises(ValueError, match="Cannot select any axis to squeeze out, as none of them has size equal to one."):
        same.squeeze([0,1])
    with pytest.raises(ValueError, match="All axes are of length 1, therefore we will not squeeze NDCube to become a scalar. Use `axis=` keyword to specify a subset of axes to squeeze."):
        same[0:1,0:1,0:1,0:1].squeeze()


def test_ndcube_quantity(ndcube_2d_ln_lt_units):
    cube = ndcube_2d_ln_lt_units
    expected = u.Quantity(cube.data, cube.unit)
    np.testing.assert_array_equal(cube.quantity, expected)


def test_data_setter(ndcube_4d_ln_l_t_lt):
    cube = ndcube_4d_ln_l_t_lt
    assert isinstance(cube.data, np.ndarray)

    new_data = np.zeros_like(cube.data)
    cube.data = new_data
    assert cube.data is new_data

    dask_array = dask.array.zeros_like(cube.data)
    cube.data = dask_array
    assert cube.data is dask_array


def test_invalid_data_setter(ndcube_4d_ln_l_t_lt):
    cube = ndcube_4d_ln_l_t_lt

    with pytest.raises(TypeError, match="set data with an array-like"):
        cube.data = None

    with pytest.raises(TypeError, match="set data with an array-like"):
        cube.data = np.zeros((100,100))

    with pytest.raises(TypeError, match="set data with an array-like"):
        cube.data = 10


def test_quantity_data_setter(ndcube_2d_ln_lt_units):
    cube = ndcube_2d_ln_lt_units
    assert cube.unit

    new_data = np.zeros_like(cube.data) * cube.unit
    cube.data = new_data

    assert isinstance(cube.data, np.ndarray)
    np.testing.assert_allclose(cube.data, new_data.value)

    new_data = np.zeros_like(cube.data) * u.Jy
    with pytest.raises(u.UnitsError, match=f"Unable to set data with unit {u.Jy}"):
        cube.data = new_data


def test_quantity_no_unit_data_setter(ndcube_4d_ln_l_t_lt):
    cube = ndcube_4d_ln_l_t_lt

    new_data = np.zeros_like(cube.data) * u.Jy
    with pytest.raises(u.UnitsError, match=f"Unable to set data with unit {u.Jy}.* current unit of None"):
        cube.data = new_data


def test_set_data_mask(ndcube_4d_mask):
    cube = ndcube_4d_mask

    assert isinstance(cube.mask, np.ndarray)

    new_data = np.ones_like(cube.data)
    new_mask = np.zeros_like(cube.mask)
    masked_array = np.ma.MaskedArray(new_data, new_mask)

    with pytest.raises(TypeError, match="Can not set the .data .* with a numpy masked array"):
        cube.data = masked_array


@pytest.mark.parametrize(
    ("ndc", "fill_value", "uncertainty_fill_value", "unmask", "expected_cube"),
    [
        ("ndcube_2d_ln_lt_mask_uncert_unit_one_maskele_true", 1.0, 0.1, False, "ndcube_2d_ln_lt_mask_uncert_unit_one_maskele_true_expected_unmask_false"),  # when it changes the cube in place: its data, uncertainty; it does not unmask the mask.
        ("ndcube_2d_ln_lt_mask_uncert_unit_one_maskele_true", 1.0, 0.1, True, "ndcube_2d_ln_lt_mask_uncert_unit_one_maskele_true_expected_unmask_true"),
        ("ndcube_2d_ln_lt_mask_uncert_unit_one_maskele_true", 1.0 * u.ct, 0.1 * u.ct, False, "ndcube_2d_ln_lt_mask_uncert_unit_one_maskele_true_expected_unmask_false"), # fill_value has a unit

        ("ndcube_2d_ln_lt_mask_uncert_unit_mask_true", 1.0, 0.1, False, "ndcube_2d_ln_lt_mask_uncert_unit_mask_true_expected_unmask_false"),  # when it changes the cube in place: its data, uncertainty; it does not unmask the mask.
        ("ndcube_2d_ln_lt_mask_uncert_unit_mask_true", 1.0, 0.1, True, "ndcube_2d_ln_lt_mask_uncert_unit_mask_true_expected_unmask_true"),
        ("ndcube_2d_ln_lt_mask_uncert_unit_mask_true", 1.0 * u.ct, 0.1* u.ct, False, "ndcube_2d_ln_lt_mask_uncert_unit_mask_true_expected_unmask_false"), # fill_value has a unit
        # TODO: test unit not aligned??

        ("ndcube_2d_ln_lt_mask_uncert_unit_mask_false", 1.0, 0.1 * u.ct, False, "ndcube_2d_ln_lt_mask_uncert_unit_mask_false") # no change.

        # TODO: are there more test cases needed?
    ],
    indirect=("ndc", "expected_cube")
)
def test_fill_masked_fill_in_place_true(ndc, fill_value, uncertainty_fill_value, unmask, expected_cube):
    # when the fill_masked method is applied on the fixture argument, it should
    # give me the correct data value and type, uncertainty, mask, unit.

    # original cube: [[0,1,2],[3,4,5]],
    # original mask: scenario 1, [[T,F,F],[F,F,F]]; scenario 2, T; scenario 3, None.
    # expected cube: [[1,1,2],[3,4,5]]; [[1,1,1], [1,1,1]]; [[0,1,2],[3,4,5]]
    # expected mask: when unmask is T, becomes all false, when unmask is F, stays the same.

    # perform the fill_masked method on the fixture, using parametrized as parameters.
    ndc.fill_masked(fill_value, unmask=unmask, uncertainty_fill_value=uncertainty_fill_value, fill_in_place=True)
    helpers.assert_cubes_equal(ndc, expected_cube, check_uncertainty_values=True)


@pytest.mark.parametrize(
    ("ndc", "fill_value", "uncertainty_fill_value", "unmask", "expected_cube"),
    [
        ("ndcube_2d_ln_lt_mask_uncert_unit_one_maskele_true", 1.0, 0.1, False, "ndcube_2d_ln_lt_mask_uncert_unit_one_maskele_true_expected_unmask_false"),  # when it changes the cube in place: its data, uncertainty; it does not unmask the mask.
        ("ndcube_2d_ln_lt_mask_uncert_unit_one_maskele_true", 1.0, 0.1, True, "ndcube_2d_ln_lt_mask_uncert_unit_one_maskele_true_expected_unmask_true"),
        ("ndcube_2d_ln_lt_mask_uncert_unit_one_maskele_true", 1.0 * u.ct, 0.1* u.ct, False, "ndcube_2d_ln_lt_mask_uncert_unit_one_maskele_true_expected_unmask_false"), # fill_value has a unit

        ("ndcube_2d_ln_lt_mask_uncert_unit_mask_true", 1.0, 0.1, False, "ndcube_2d_ln_lt_mask_uncert_unit_mask_true_expected_unmask_false"),  # when it changes the cube in place: its data, uncertainty; it does not unmask the mask.
        ("ndcube_2d_ln_lt_mask_uncert_unit_mask_true", 1.0, 0.1, True, "ndcube_2d_ln_lt_mask_uncert_unit_mask_true_expected_unmask_true"),
        ("ndcube_2d_ln_lt_mask_uncert_unit_mask_true", 1.0 * u.ct, 0.1* u.ct, False, "ndcube_2d_ln_lt_mask_uncert_unit_mask_true_expected_unmask_false"), # fill_value has a unit
        #TODO: test unit not aligned??

        ("ndcube_2d_ln_lt_mask_uncert_unit_mask_false", 1.0, 0.1 * u.ct, False, "ndcube_2d_ln_lt_mask_uncert_unit_mask_false") # no change.

        # TODO: are there more test cases needed? yes: when uncertainty fill is not None but ndc's uncertainty is None.
    ],
    indirect=("ndc", "expected_cube")
)
def test_fill_masked_fill_in_place_false(ndc, fill_value, uncertainty_fill_value, unmask, expected_cube):
    # compare the expected cube with the cube saved in the new place

    # perform the fill_masked method on the fixture, using parametrized as parameters.
    filled_cube = ndc.fill_masked(fill_value, uncertainty_fill_value, unmask, fill_in_place=False)
    helpers.assert_cubes_equal(filled_cube, expected_cube, check_uncertainty_values=True)

@pytest.mark.parametrize(
    ("ndc", "fill_value", "uncertainty_fill_value", "unmask"),
    [
        # cube has no uncertainty but uncertainty_fill_value has an uncertainty
        ("ndcube_2d_ln_lt_mask", 1.0, 0.1, False),
        ("ndcube_2d_ln_lt_mask", 1.0, 0.1 * u.ct, True),
    ],
    indirect=("ndc",)
)
def test_fill_masked_ndc_uncertainty_none(ndc, fill_value, uncertainty_fill_value, unmask):
    assert ndc.uncertainty is None
    with pytest.raises(TypeError,match="Cannot fill uncertainty as uncertainty is None."):
        ndc.fill_masked(
            fill_value,
            unmask=unmask,
            uncertainty_fill_value=uncertainty_fill_value,
            fill_in_place=True
        )
