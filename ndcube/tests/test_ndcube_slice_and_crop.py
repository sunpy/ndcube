from textwrap import dedent

import numpy as np
import pytest

import astropy.units as u
import astropy.wcs
from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.io import fits
from astropy.time import Time
from astropy.units import UnitsError
from astropy.wcs import WCS
from astropy.wcs.utils import wcs_to_celestial_frame
from astropy.wcs.wcsapi import BaseHighLevelWCS, HighLevelWCSWrapper
from astropy.wcs.wcsapi.wrappers import SlicedLowLevelWCS

from ndcube import ExtraCoords, NDCube, NDMeta
from ndcube.tests import helpers


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


@pytest.mark.parametrize(("ndc","item","expected_shape"),
                         [
                             ("ndcube_4d_ln_l_t_lt", np.s_[..., 1], (5, 10, 12)),
                             ("ndcube_4d_ln_l_t_lt", np.s_[..., 1:, 1], (5, 10, 11)),
                             ("ndcube_4d_ln_l_t_lt", np.s_[1, ...], (10, 12, 8)),
                             ("ndcube_4d_ln_l_t_lt", np.s_[1, 1:, ...], (9, 12, 8)),
                             ("ndcube_4d_ln_l_t_lt", np.s_[1, ..., 1:], (10, 12, 7)),
                             ("ndcube_4d_ln_l_t_lt", np.s_[1, 1:, ..., 1:], (9, 12, 7)),
                         ],
                         indirect=("ndc",))
def test_ellipsis_usage(ndc, item, expected_shape):
    sliced_cube = ndc[item]
    assert sliced_cube.data.shape == expected_shape

def test_ellipsis_error(ndcube_4d_ln_l_t_lt):
    with pytest.raises(IndexError, match="An index can only have a single ellipsis"):
        ndcube_4d_ln_l_t_lt[..., ..., 1]


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


def test_crop_length_1_input(ndcube_2d_ln_lt):
    cube = ndcube_2d_ln_lt
    frame = astropy.wcs.utils.wcs_to_celestial_frame(cube.wcs)
    lower_corner = SkyCoord(Tx=[0359.99667], Ty=[-0.0011111111], unit="deg", frame=frame)
    upper_corner = SkyCoord(Tx=[[0.0044444444]], Ty=[[0.0011111111]], unit="deg", frame=frame)
    cropped_by_shaped = cube.crop(lower_corner, upper_corner)
    cropped_by_scalars = cube.crop((lower_corner.squeeze(),), (upper_corner.squeeze(),))
    helpers.assert_cubes_equal(cropped_by_shaped, cropped_by_scalars)


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


def test_crop_by_extra_coords_using_combined_wcs(ndcube_3d_ln_lt_l_ec_time):
    cube = ndcube_3d_ln_lt_l_ec_time
    # ['spectral_0', 'celestial_0', 'temporal_1', 'PIXEL_1']
    lower_corner = (Time("2000-01-01T15:00:00", scale="utc", format="fits"), None, None)
    upper_corner = (Time("2000-01-01T20:00:00", scale="utc", format="fits"), None, None)
    output = cube.crop(lower_corner, upper_corner, wcs=HighLevelWCSWrapper(cube.extra_coords.cube_wcs))
    expected = cube[0]
    helpers.assert_cubes_equal(output, expected)


def test_crop_by_extra_coords_values_using_combined_wcs(ndcube_3d_ln_lt_l_ec_time):
    cube = ndcube_3d_ln_lt_l_ec_time
    lower_corner = (3 * 60 * 60 * u.s, None, None)
    upper_corner = (8 * 60 * 60 * u.s, None, None)
    output = cube.crop_by_values(lower_corner, upper_corner, wcs=HighLevelWCSWrapper(cube.extra_coords.cube_wcs))
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


def test_crop_1d():
    # This use case revealed a bug so has been added as a test.
    # Create NDCube.
    wcs = astropy.wcs.WCS(naxis=1)
    wcs.wcs.ctype = 'WAVE',
    wcs.wcs.cunit = 'nm',
    wcs.wcs.cdelt = 4,
    wcs.wcs.crpix = 1,
    wcs.wcs.crval = 3,
    cube = NDCube(np.arange(200), wcs=wcs)

    expected = cube[1:4]

    output = cube.crop((7*u.nm,), (15*u.nm,))

    helpers.assert_cubes_equal(output, expected)


@pytest.mark.filterwarnings("ignore::Warning")
@pytest.mark.parametrize(("points", "expected_slice", "crop_by_values", "keepdims"),
                         [
                             (((15*u.m,), (45*u.m,)), np.s_[1:4], False, False), # A range starting and ending at different pixel edges
                             (((15*u.m,), (45*u.m,)), np.s_[1:4], True, False),
                             (((15*u.m,)), np.s_[1:2], False, True), # A range starting and ending on same pixel edge.
                             (((15*u.m,)), np.s_[1:2], True, True),
                             (((5*u.m,)), np.s_[0:1], False, True), # A range starting and ending at the exact start of the cube extent.
                             (((5*u.m,)), np.s_[0:1], True, True),
                             (((104*u.m,)), np.s_[9:10], False, True), # A range starting and ending slightly below the end of cube extent.
                             (((104*u.m,)), np.s_[9:10], True, True),
                             (((1*u.m,), (40*u.m,)), np.s_[:4], False, False), # A range starting below cube extent.
                             (((1*u.m,), (40*u.m,)), np.s_[:4], True, False),
                             (((15*u.m,), (200*u.m,)), np.s_[1:], False, False), # A range ending above cube extent.
                             (((15*u.m,), (200*u.m,)), np.s_[1:], True, False),
                         ])
def test_crop_at_pixel_edges(points, expected_slice, crop_by_values, keepdims):
    wcs = astropy.wcs.WCS(naxis=1)
    wcs.wcs.ctype = 'WAVE',
    wcs.wcs.cunit = 'm',
    wcs.wcs.cdelt = 10,
    wcs.wcs.crpix = 1,
    wcs.wcs.crval = 10,
    cube = NDCube(np.arange(10), wcs=wcs)

    expected = cube[expected_slice]

    output = cube.crop_by_values(*points, keepdims=keepdims) if crop_by_values else cube.crop(*points, keepdims=keepdims)

    helpers.assert_cubes_equal(output, expected)


@pytest.mark.filterwarnings("ignore::Warning")
@pytest.mark.parametrize("points",
                         [
                             ((1*u.m,),),
                             ((105*u.m,),), # Exactly at the end of the cube extent.
                             ((200*u.m,),),
                         ])
def test_crop_all_points_beyond_cube_extent_error(points):
    wcs = astropy.wcs.WCS(naxis=1)
    wcs.wcs.ctype = 'WAVE',
    wcs.wcs.cunit = 'm',
    wcs.wcs.cdelt = 10,
    wcs.wcs.crpix = 1,
    wcs.wcs.crval = 10,
    cube = NDCube(np.arange(10), wcs=wcs)

    with pytest.raises(ValueError, match="are outside the range of the NDCube being cropped"):
        cube.crop(*points, keepdims=True)
