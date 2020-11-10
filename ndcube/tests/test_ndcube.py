import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.wcs.wcsapi import BaseHighLevelWCS, BaseLowLevelWCS
from astropy.wcs.wcsapi.wrappers import SlicedLowLevelWCS

from ndcube import NDCube
from ndcube.tests import helpers


def generate_data(shape):
    data = np.arange(np.product(shape))
    return data.reshape(shape)


def test_wcs_object(all_ndcubes):
    assert isinstance(all_ndcubes.wcs.low_level_wcs, BaseLowLevelWCS)
    assert isinstance(all_ndcubes.wcs, BaseHighLevelWCS)


@pytest.mark.parametrize("ndc, item",
                         (
                             ("ndcube_3d_ln_lt_l", np.s_[:, :, 0]),
                             ("ndcube_3d_ln_lt_l", np.s_[..., 0]),
                             ("ndcube_3d_ln_lt_l", np.s_[1:2, 1:2, 0]),
                             ("ndcube_3d_ln_lt_l", np.s_[..., 0]),
                             ("ndcube_3d_ln_lt_l", np.s_[:, :, 0]),
                             ("ndcube_3d_ln_lt_l", np.s_[1:2, 1:2, 0]),
                             ("ndcube_4d_ln_lt_l_t", np.s_[:, :, 0, 0]),
                             ("ndcube_4d_ln_lt_l_t", np.s_[..., 0, 0]),
                             ("ndcube_4d_ln_lt_l_t", np.s_[1:2, 1:2, 1, 1]),
                         ),
                         indirect=("ndc",))
def test_slicing_ln_lt(ndc, item):
    sndc = ndc[item]
    assert len(sndc.dimensions) == 2
    assert set(sndc.wcs.world_axis_physical_types) == {"custom:pos.helioprojective.lat", "custom:pos.helioprojective.lon"}
    if sndc.uncertainty is not None:
        assert np.allclose(sndc.data, sndc.uncertainty.array)
    if sndc.mask is not None:
        assert np.allclose(sndc.data > 0, sndc.mask)

    if ndc.extra_coords and ndc.extra_coords.keys():
        ec = sndc.extra_coords
        assert set(ec.keys()) == {"time", "hello", "bye"}

    wcs = sndc.wcs
    assert isinstance(wcs, BaseHighLevelWCS)
    assert isinstance(wcs.low_level_wcs, SlicedLowLevelWCS)
    assert wcs.pixel_n_dim == 2
    assert wcs.world_n_dim == 2
    assert np.allclose(wcs.array_shape, sndc.data.shape)
    assert np.allclose(sndc.wcs.axis_correlation_matrix, np.ones(2, dtype=bool))


@pytest.mark.parametrize("ndc, item",
                         (
                             ("ndcube_3d_ln_lt_l", np.s_[0, 0, :]),
                             ("ndcube_3d_ln_lt_l", np.s_[0, 0, ...]),
                             ("ndcube_3d_ln_lt_l", np.s_[1, 1, 1:2]),
                             ("ndcube_3d_ln_lt_l", np.s_[0, 0, :]),
                             ("ndcube_3d_ln_lt_l", np.s_[0, 0, ...]),
                             ("ndcube_3d_ln_lt_l", np.s_[1, 1, 1:2]),
                             ("ndcube_4d_ln_lt_l_t", np.s_[0, 0, :, 0]),
                             ("ndcube_4d_ln_lt_l_t", np.s_[0, 0, ..., 0]),
                             ("ndcube_4d_ln_lt_l_t", np.s_[1, 1, 1:2, 1]),
                         ),
                         indirect=("ndc",))
def test_slicing_wave(ndc, item):
    sndc = ndc[item]
    assert len(sndc.dimensions) == 1
    assert set(sndc.wcs.world_axis_physical_types) == {"em.wl"}
    if sndc.uncertainty is not None:
        assert np.allclose(sndc.data, sndc.uncertainty.array)
    if sndc.mask is not None:
        assert np.allclose(sndc.data > 0, sndc.mask)

    if ndc.extra_coords and ndc.extra_coords.keys():
        ec = sndc.extra_coords
        assert set(ec.keys()) == {"bye", "hello", "time"}

    wcs = sndc.wcs
    assert isinstance(wcs, BaseHighLevelWCS)
    assert isinstance(wcs.low_level_wcs, SlicedLowLevelWCS)
    assert wcs.pixel_n_dim == 1
    assert wcs.world_n_dim == 1
    assert np.allclose(wcs.array_shape, sndc.data.shape)
    assert np.allclose(sndc.wcs.axis_correlation_matrix, np.ones(1, dtype=bool))


@pytest.mark.parametrize("ndc, item",
                         (
                             ("ndcube_3d_ln_lt_l", np.s_[0, :, :]),
                             ("ndcube_3d_ln_lt_l", np.s_[0, ...]),
                             ("ndcube_3d_ln_lt_l", np.s_[1, 1:2]),
                             ("ndcube_3d_ln_lt_l", np.s_[0, :, :]),
                             ("ndcube_3d_ln_lt_l", np.s_[0, ...]),
                             ("ndcube_3d_ln_lt_l", np.s_[1, :, 1:2]),
                             ("ndcube_4d_ln_lt_l_t", np.s_[0, :, :, 0]),
                             ("ndcube_4d_ln_lt_l_t", np.s_[0, ..., 0]),
                             ("ndcube_4d_ln_lt_l_t", np.s_[1, 1:2, 1:2, 1]),
                         ),
                         indirect=("ndc",))
def test_slicing_split_celestial(ndc, item):
    sndc = ndc[item]
    assert len(sndc.dimensions) == 2
    if sndc.uncertainty is not None:
        assert np.allclose(sndc.data, sndc.uncertainty.array)
    if sndc.mask is not None:
        assert np.allclose(sndc.data > 0, sndc.mask)

    if ndc.extra_coords and ndc.extra_coords.keys():
        ec = sndc.extra_coords
        assert set(ec.keys()) == {"hello", "bye", "time"}

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


@pytest.mark.parametrize("axes", ([-1], [2], ["em"]))
def test_axis_world_coords_single(axes, ndcube_3d_ln_lt_l):
    coords = ndcube_3d_ln_lt_l.axis_world_coords_values(*axes)
    assert u.allclose(coords, [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09]*u.m)


@pytest.mark.parametrize("axes", ([-1], [2], ["em"]))
def test_axis_world_coords_single_edges(axes, ndcube_3d_ln_lt_l):
    coords = ndcube_3d_ln_lt_l.axis_world_coords_values(*axes, edges=True)
    assert u.allclose(coords, [1.01e-09, 1.03e-09, 1.05e-09, 1.07e-09, 1.09e-09]*u.m)


@pytest.mark.parametrize("ndc, item",
                         (
                             ("ndcube_3d_ln_lt_l", np.s_[0, 0, :]),
                             ("ndcube_3d_ln_lt_l", np.s_[0, 0, ...]),
                             ("ndcube_3d_ln_lt_l", np.s_[0, 0, :]),
                             ("ndcube_3d_ln_lt_l", np.s_[0, 0, ...]),
                         ),
                         indirect=("ndc",))
def test_axis_world_coords_sliced_all_3d(ndc, item):
    coords = ndc[item].axis_world_coords_values()
    assert u.allclose(coords, [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09] * u.m)


@pytest.mark.parametrize("ndc, item",
                         (
                             ("ndcube_4d_ln_lt_l_t", np.s_[0, 0, :, 0]),
                             ("ndcube_4d_ln_lt_l_t", np.s_[0, 0, ..., 0]),
                         ),
                         indirect=("ndc",))
def test_axis_world_coords_sliced_all_4d(ndc, item):
    coords = ndc[item].axis_world_coords_values()

    expected = [2.0e-11, 4.0e-11, 6.0e-11, 8.0e-11, 1.0e-10,
                1.2e-10, 1.4e-10, 1.6e-10, 1.8e-10, 2.0e-10] * u.m

    assert u.allclose(coords, expected)


@pytest.mark.xfail
def test_axis_world_coords_all(ndcube_3d_ln_lt_l):
    coords = ndcube_3d_ln_lt_l.axis_world_coord()
    assert len(coords) == 2
    assert isinstance(coords[0], SkyCoord)

    assert u.allclose(coords[0].Tx, [[0.60002173, 0.59999127, 0.5999608],
                                     [1., 1., 1.]] * u.deg)
    assert u.allclose(coords[0].Ty, [[1.26915033e-05, 4.99987815e-01, 9.99962939e-01],
                                     [1.26918126e-05, 5.00000000e-01, 9.99987308e-01]] * u.deg)
    assert isinstance(coords[1], u.Quantity)
    assert u.allclose(coords[1], [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09] * u.m)


def test_axis_world_coords_values_all(ndcube_3d_ln_lt_l):
    coords = ndcube_3d_ln_lt_l.axis_world_coords_values()
    assert len(coords) == 3
    assert all(isinstance(c, u.Quantity) for c in coords)

    assert u.allclose(coords[0], [[0.00277778, 0.00277778, 0.00277778],
                                  [0.00555556, 0.00555556, 0.00555556]] * u.deg)
    assert u.allclose(coords[1], [[-0.00555556, -0.00416667, -0.00277778],
                                  [-0.00555556, -0.00416667, -0.00277778]] * u.deg)
    assert u.allclose(coords[2], [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09] * u.m)


def test_array_axis_physical_types(ndcube_4d_ln_lt_l_t):
    expected = [
            ('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat'),
            ('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat'),
            ('em.wl',), ('time',)]
    output = ndcube_4d_ln_lt_l_t.array_axis_physical_types
    for i in range(len(expected)):
        assert all([physical_type in expected[i] for physical_type in output[i]])


def test_crop(ndcube_4d_ln_lt_l_t):
    intervals = ndcube_4d_ln_lt_l_t.wcs.array_index_to_world([1, 2], [0, 1], [0, 1], [0, 2])
    expected = ndcube_4d_ln_lt_l_t[1:3, 0:2, 0:2, 0:3]
    output = ndcube_4d_ln_lt_l_t.crop(*intervals)
    helpers.assert_cubes_equal(output, expected)


def test_crop_with_nones(ndcube_4d_ln_lt_l_t):
    intervals = [None] * 4
    intervals[0] = ndcube_4d_ln_lt_l_t.wcs.array_index_to_world([1, 2], [0, 1], [0, 1], [0, 2])[0]
    expected = ndcube_4d_ln_lt_l_t[:, :, :, 0:3]
    output = ndcube_4d_ln_lt_l_t.crop(*intervals)
    helpers.assert_cubes_equal(output, expected)


def test_crop_1d_independent(ndcube_4d_ln_lt_l_t):
    cube_1d = ndcube_4d_ln_lt_l_t[0, 0, :, 0]
    wl_range = SpectralCoord([3e-11, 4.5e-11], unit=u.m)
    expected = cube_1d[0:2]
    output = cube_1d.crop(wl_range)
    helpers.assert_cubes_equal(output, expected)


def test_crop_1d_dependent(ndcube_4d_ln_lt_l_t):
    cube_1d = ndcube_4d_ln_lt_l_t[0, :, 0, 0]
    sky_range = cube_1d.wcs.array_index_to_world([0, 1])
    expected = cube_1d[0:2]
    output = cube_1d.crop(sky_range)
    helpers.assert_cubes_equal(output, expected)


def test_crop_by_values(ndcube_4d_ln_lt_l_t):
    time_range = [0.4, 1.2] * u.min
    wl_range = [2e-11, 4e-11] * u.m
    lon_range = [359.99, 359.996] * u.deg
    lat_range = [0.00556, 0.01112] * u.deg
    expected = ndcube_4d_ln_lt_l_t[1:3, 0:2, 0:2, 0:3]
    output = ndcube_4d_ln_lt_l_t.crop_by_values(time_range, wl_range, lat_range, lon_range)
    helpers.assert_cubes_equal(output, expected)


def test_crop_by_values_with_nones(ndcube_4d_ln_lt_l_t):
    intervals = [None] * 4
    intervals[0] = [0.5, 1.1] * u.min
    expected = ndcube_4d_ln_lt_l_t[:, :, :, 0:3]
    output = ndcube_4d_ln_lt_l_t.crop_by_values(*intervals)
    helpers.assert_cubes_equal(output, expected)


def test_crop_by_values_all_nones(ndcube_4d_ln_lt_l_t):
    intervals = [None] * 4
    output = ndcube_4d_ln_lt_l_t.crop_by_values(*intervals)
    helpers.assert_cubes_equal(output, ndcube_4d_ln_lt_l_t)


def test_crop_by_values_indexerror(ndcube_4d_ln_lt_l_t):
    time_range = [0.5, 1.1] * u.min
    wl_range = [-3e-11, -2.5e-11] * u.m
    lat_range = [0.6, 0.75] * u.deg
    lon_range = [1, 1.5] * u.deg
    with pytest.raises(IndexError):
        ndcube_4d_ln_lt_l_t.crop_by_values(time_range, wl_range, lat_range, lon_range)


def test_crop_1d_dependent_2(ndcube_4d_ln_lt_l_t):
    cube_1d = ndcube_4d_ln_lt_l_t[0, :, 0, 0]
    lon_range = [359.99, 359.996] * u.deg
    lat_range = [0.00556, 0.01112] * u.deg
    expected = cube_1d[0:2]
    output = cube_1d.crop_by_values(lat_range, lon_range)
    helpers.assert_cubes_equal(output, expected)
