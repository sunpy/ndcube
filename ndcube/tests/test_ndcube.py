import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.wcs import WCS
from astropy.wcs.wcsapi import BaseHighLevelWCS, BaseLowLevelWCS
from astropy.wcs.wcsapi.wrappers import SlicedLowLevelWCS

from ndcube import NDCube
from ndcube.tests import helpers


def generate_data(shape):
    data = np.arange(np.product(shape))
    return data.reshape(shape)


@pytest.fixture
def wcs_4d_ln_lt_l_t():
    ht = {'CTYPE4': 'HPLN-TAN', 'CUNIT4': 'deg', 'CDELT4': 0.4, 'CRPIX4': 2, 'CRVAL4': 1, 'NAXIS4': 2,
          'CTYPE3': 'HPLT-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.5, 'CRPIX3': 0, 'CRVAL3': 0, 'NAXIS3': 3,
          'CTYPE2': 'WAVE    ', 'CUNIT2': 'Angstrom', 'CDELT2': 0.2, 'CRPIX2': 0, 'CRVAL2': 10,
          'NAXIS2': 4,
          'CTYPE1': 'TIME    ', 'CUNIT1': 'min', 'CDELT1': 0.4, 'CRPIX1': 0, 'CRVAL1': 0, 'NAXIS1': 5,
          'DATEREF':"2020-01-01T00:00:00"}
    return WCS(header=ht)


@pytest.fixture
def wcs_3d_l_lt_ln():
    hm = {'CTYPE1': 'WAVE    ', 'CUNIT1': 'Angstrom', 'CDELT1': 0.2, 'CRPIX1': 0, 'CRVAL1': 10,
          'NAXIS1': 4,
          'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5,
          'NAXIS2': 3,
          'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 2}
    return WCS(header=hm)


@pytest.fixture
def wcs_3d_ln_lt_t_rotated():
    h_rotated = {'CTYPE1': 'HPLN-TAN', 'CUNIT1': 'arcsec', 'CDELT1': 0.4, 'CRPIX1': 0,
                 'CRVAL1': 0, 'NAXIS1': 5,
                 'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'arcsec', 'CDELT2': 0.5, 'CRPIX2': 0,
                 'CRVAL2': 0, 'NAXIS2': 5,
                 'CTYPE3': 'TIME    ', 'CUNIT3': 'seconds', 'CDELT3': 3, 'CRPIX3': 0,
                 'CRVAL3': 0, 'NAXIS3': 2, 'DATEREF':"2020-01-01T00:00:00",
                 'PC1_1': 0.714963912964, 'PC1_2': -0.699137151241, 'PC1_3': 0.0,
                 'PC2_1': 0.699137151241, 'PC2_2': 0.714963912964, 'PC2_3': 0.0,
                 'PC3_1': 0.0, 'PC3_2': 0.0, 'PC3_3': 1.0}
    return WCS(header=h_rotated)


@pytest.fixture
def simple_extra_coords():
    data = generate_data((2, 3, 4))
    return [('time', 0, u.Quantity(range(data.shape[0]), unit=u.pix)),
            ('hello', 1, u.Quantity(range(data.shape[1]), unit=u.pix)),
            ('bye', 2, u.Quantity(range(data.shape[2]), unit=u.pix))]


@pytest.fixture
def simple_ndcube(wcs_3d_l_lt_ln):
    data = generate_data((2, 3, 4))
    return NDCube(
        data,
        wcs_3d_l_lt_ln,
    )


@pytest.fixture
def ndcube_3d_1(wcs_3d_l_lt_ln, simple_extra_coords):
    data = generate_data((2, 3, 4))
    mask = data > 0
    return NDCube(
        data,
        wcs_3d_l_lt_ln,
        mask = mask,
        uncertainty=data,
        extra_coords=simple_extra_coords
    )


@pytest.fixture
def ndcube_4d_1(wcs_4d_ln_lt_l_t, simple_extra_coords):
    data = generate_data((2, 3, 4, 5))
    mask = data > 0
    return NDCube(
        data,
        wcs_4d_ln_lt_l_t,
        mask = mask,
        uncertainty=data,
        extra_coords=simple_extra_coords
    )


@pytest.fixture
def ndcube_3d_rotated(wcs_3d_ln_lt_t_rotated, simple_extra_coords):
    data_rotated = np.array([[[1, 2, 3, 4, 6], [2, 4, 5, 3, 1], [0, -1, 2, 4, 2], [3, 5, 1, 2, 0]],
                             [[2, 4, 5, 1, 3], [1, 5, 2, 2, 4], [2, 3, 4, 0, 5], [0, 1, 2, 3, 4]]])
    mask_rotated = data_rotated >= 0
    return NDCube(
        data_rotated,
        wcs_3d_ln_lt_t_rotated,
        mask=mask_rotated,
        uncertainty=data_rotated,
        extra_coords=simple_extra_coords
    )


@pytest.fixture(params=["simple_ndcube", "ndcube_3d_1", "ndcube_4d_1", "ndcube_3d_rotated"])
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


def test_wcs_object(all_ndcubes):
    assert isinstance(all_ndcubes.wcs.low_level_wcs, BaseLowLevelWCS)
    assert isinstance(all_ndcubes.wcs, BaseHighLevelWCS)


@pytest.mark.parametrize("ndc, item",
                         (
                             ("simple_ndcube", np.s_[:, :, 0]),
                             ("simple_ndcube", np.s_[..., 0]),
                             ("simple_ndcube", np.s_[1:2, 1:2, 0]),
                             ("ndcube_3d_1", np.s_[..., 0]),
                             ("ndcube_3d_1", np.s_[:, :, 0]),
                             ("ndcube_3d_1", np.s_[1:2, 1:2, 0]),
                             ("ndcube_4d_1", np.s_[:, :, 0, 0]),
                             ("ndcube_4d_1", np.s_[..., 0, 0]),
                             ("ndcube_4d_1", np.s_[1:2, 1:2, 1, 1]),
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
                             ("simple_ndcube", np.s_[0, 0, :]),
                             ("simple_ndcube", np.s_[0, 0, ...]),
                             ("simple_ndcube", np.s_[1, 1, 1:2]),
                             ("ndcube_3d_1", np.s_[0, 0, :]),
                             ("ndcube_3d_1", np.s_[0, 0, ...]),
                             ("ndcube_3d_1", np.s_[1, 1, 1:2]),
                             ("ndcube_4d_1", np.s_[0, 0, :, 0]),
                             ("ndcube_4d_1", np.s_[0, 0, ..., 0]),
                             ("ndcube_4d_1", np.s_[1, 1, 1:2, 1]),
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
                             ("simple_ndcube", np.s_[0, :, :]),
                             ("simple_ndcube", np.s_[0, ...]),
                             ("simple_ndcube", np.s_[1, 1:2]),
                             ("ndcube_3d_1", np.s_[0, :, :]),
                             ("ndcube_3d_1", np.s_[0, ...]),
                             ("ndcube_3d_1", np.s_[1, :, 1:2]),
                             ("ndcube_4d_1", np.s_[0, :, :, 0]),
                             ("ndcube_4d_1", np.s_[0, ..., 0]),
                             ("ndcube_4d_1", np.s_[1, 1:2, 1:2, 1]),
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
def test_axis_world_coords_single(axes, simple_ndcube):
    coords = simple_ndcube.axis_world_coords_values(*axes)
    assert u.allclose(coords, [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09]*u.m)


@pytest.mark.parametrize("axes", ([-1], [2], ["em"]))
def test_axis_world_coords_single_edges(axes, simple_ndcube):
    coords = simple_ndcube.axis_world_coords_values(*axes, edges=True)
    assert u.allclose(coords, [1.01e-09, 1.03e-09, 1.05e-09, 1.07e-09, 1.09e-09]*u.m)


@pytest.mark.parametrize("ndc, item",
                         (
                             ("simple_ndcube", np.s_[0, 0, :]),
                             ("simple_ndcube", np.s_[0, 0, ...]),
                             ("ndcube_3d_1", np.s_[0, 0, :]),
                             ("ndcube_3d_1", np.s_[0, 0, ...]),
                             ("ndcube_4d_1", np.s_[0, 0, :, 0]),
                             ("ndcube_4d_1", np.s_[0, 0, ..., 0]),
                         ),
                         indirect=("ndc",))
def test_axis_world_coords_sliced_all(ndc, item):
    coords = ndc[item].axis_world_coords_values()
    assert u.allclose(coords, [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09]*u.m)


@pytest.mark.xfail
def test_axis_world_coords_all(simple_ndcube):
    coords = simple_ndcube.axis_world_coord()
    assert len(coords) == 2
    assert isinstance(coords[0], SkyCoord)

    assert u.allclose(coords[0].Tx, [[0.60002173, 0.59999127, 0.5999608],
                                     [1., 1., 1.]] * u.deg)
    assert u.allclose(coords[0].Ty, [[1.26915033e-05, 4.99987815e-01, 9.99962939e-01],
                                     [1.26918126e-05, 5.00000000e-01, 9.99987308e-01]] * u.deg)
    assert isinstance(coords[1], u.Quantity)
    assert u.allclose(coords[1], [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09] * u.m)


def test_axis_world_coords_values_all(simple_ndcube):
    coords = simple_ndcube.axis_world_coords_values()
    assert len(coords) == 3
    assert all(isinstance(c, u.Quantity) for c in coords)

    assert u.allclose(coords[0], [[0.60002173, 0.59999127, 0.5999608],
                                  [1., 1., 1.]] * u.deg)
    assert u.allclose(coords[1], [[1.26915033e-05, 4.99987815e-01, 9.99962939e-01],
                                  [1.26918126e-05, 5.00000000e-01, 9.99987308e-01]] * u.deg)
    assert u.allclose(coords[2], [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09] * u.m)


def test_array_axis_physical_types(ndcube_4d_1):
    expected = [
            ('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat'),
            ('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat'),
            ('em.wl',), ('time',)]
    output = ndcube_4d_1.array_axis_physical_types
    for i in range(len(expected)):
        assert all([physical_type in expected[i] for physical_type in output[i]])


def test_crop(ndcube_4d_1):
    intervals = ndcube_4d_1.wcs.array_index_to_world([1, 2], [0, 1], [0, 1], [0, 2])
    expected = ndcube_4d_1[1:3, 0:2, 0:2, 0:3]
    output = ndcube_4d_1.crop(*intervals)
    helpers.assert_cubes_equal(output, expected)


def test_crop_with_nones(ndcube_4d_1):
    intervals = [None] * 4
    intervals[0] = ndcube_4d_1.wcs.array_index_to_world([1, 2], [0, 1], [0, 1], [0, 2])[0]
    expected = ndcube_4d_1[:, :, :, 0:3]
    output = ndcube_4d_1.crop(*intervals)
    helpers.assert_cubes_equal(output, expected)


def test_crop_1d_independent(ndcube_4d_1):
    cube_1d = ndcube_4d_1[0, 0, :, 0]
    wl_range = SpectralCoord([1.02e-9, 1.04e-9], unit=u.m)
    expected = cube_1d[0:2]
    output = cube_1d.crop(wl_range)
    helpers.assert_cubes_equal(output, expected)


def test_crop_1d_dependent(ndcube_4d_1):
    cube_1d = ndcube_4d_1[0, :, 0, 0]
    sky_range = cube_1d.wcs.array_index_to_world([0, 1])
    expected = cube_1d[0:2]
    output = cube_1d.crop(sky_range)
    helpers.assert_cubes_equal(output, expected)


def test_crop_by_values(ndcube_4d_1):
    time_range = [0.5, 1.1] * u.min
    wl_range = [1.02e-9, 1.04e-9] * u.m
    lat_range = [0.6, 0.75] * u.deg
    lon_range = [1, 1.5]*u.deg
    expected = ndcube_4d_1[1:3, 0:2, 0:2, 0:3]
    output = ndcube_4d_1.crop_by_values(time_range, wl_range, lat_range, lon_range)
    helpers.assert_cubes_equal(output, expected)


def test_crop_by_values_with_nones(ndcube_4d_1):
    intervals = [None] * 4
    intervals[0] = [0.5, 1.1] * u.min
    expected = ndcube_4d_1[:, :, :, 0:3]
    print(ndcube_4d_1.dimensions)
    output = ndcube_4d_1.crop_by_values(*intervals)
    helpers.assert_cubes_equal(output, expected)


def test_crop_by_values_all_nones(ndcube_4d_1):
    intervals = [None] * 4
    output = ndcube_4d_1.crop_by_values(*intervals)
    helpers.assert_cubes_equal(output, ndcube_4d_1)


def test_crop_by_values_indexerror(ndcube_4d_1):
    time_range = [0.5, 1.1] * u.min
    wl_range = [-3e-11, -2.5e-11] * u.m
    lat_range = [0.6, 0.75] * u.deg
    lon_range = [1, 1.5]*u.deg
    with pytest.raises(IndexError):
        output = ndcube_4d_1.crop_by_values(time_range, wl_range, lat_range, lon_range)


def test_crop_1d_dependent(ndcube_4d_1):
    cube_1d = ndcube_4d_1[0, :, 0, 0]
    lat_range = [0.6, 0.75] * u.deg
    lon_range = [1, 1]*u.deg
    expected = cube_1d[0:2]
    output = cube_1d.crop_by_values(lat_range, lon_range)
    helpers.assert_cubes_equal(output, expected)
