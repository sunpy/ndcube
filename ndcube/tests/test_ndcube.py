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


def test_array_axis_physical_types(ndcube_3d_ln_lt_l):
    expected = [
        ('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat', 'custom:PIXEL'),
        ('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat', 'custom:PIXEL'),
        ('em.wl', 'custom:PIXEL')]
    output = ndcube_3d_ln_lt_l.array_axis_physical_types
    for i in range(len(expected)):
        assert all(physical_type in expected[i] for physical_type in output[i])


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
    with pytest.raises(TypeError, match="Adding objects requires both have a unit or neither has a unit."):
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
    expected_uncertainty = ndc.uncertainty.propagate(
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


# The case when both NDData and NDCube have uncertainty, unit. Also:
# 1, NDCube has mask but NDData does not;
# 2, Both NDCube and NDData have masks.
@pytest.mark.parametrize('value', [
    NDData(np.ones((10, 12)),
           wcs=None,
           uncertainty=StdDevUncertainty(np.ones((10, 12)) * 0.1)),

    NDData(np.ones((10, 12)) * 2,
           wcs=None,
           uncertainty=StdDevUncertainty(np.ones((10, 12)) * 0.05),
           mask=np.ones((10, 12), dtype=bool))
])
def test_arithmetic_add_cube_unit_mask_nddata_unc_unit_mask(ndcube_2d_ln_lt_mask, value):
    with pytest.raises(TypeError, match='Please use the add method.'):
        ndcube_2d_ln_lt_mask + value


# Test the three different with-mask scenarios for the add method.
# 1, both have masks. To test: data, combined mask, uncertainty
@pytest.mark.parametrize(
    ("value", "handle_mask"),
    [(NDData(np.ones((2, 3)),
            wcs=None,
            uncertainty=StdDevUncertainty(np.ones((2, 3)) * 0.05),
            mask=np.ones((2, 3), dtype=bool)),
      np.logical_and)]
)
def test_arithmetic_add_both_mask(ndcube_2d_ln_lt_mask2, value, handle_mask):
    output_cube = ndcube_2d_ln_lt_mask2.add(value, handle_mask)  # perform the addition

    # Construct expected cube
    expected_data = ndcube_2d_ln_lt_mask2.data + value.data
    expected_uncertainty = ndcube_2d_ln_lt_mask2.uncertainty
    expected_mask = np.array([[False, True, True],
                              [True, True, True]])
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
    output_cube = ndcube_2d_ln_lt_mask2.add(value)  # perform the addition

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
    output_cube = ndcube_2d_ln_lt_nomask.add(value)  # perform the addition

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
