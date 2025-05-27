import re
import copy

import dask.array
import numpy as np
import pytest
from specutils import Spectrum1D

import astropy.units as u
import astropy.wcs
from astropy.coordinates import SpectralCoord 
from astropy.nddata import UnknownUncertainty
from astropy.time import Time
from ndcube import ExtraCoords#, NDCube, NDMeta
from ndcube.tests import helpers
from ndcube.utils.exceptions import NDCubeUserWarning


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


############### REBIN ############################

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
