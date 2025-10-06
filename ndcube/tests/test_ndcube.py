from inspect import signature

import dask.array
import numpy as np
import pytest

import astropy.nddata
import astropy.units as u
import astropy.wcs
from astropy.wcs.wcsapi import BaseHighLevelWCS
from astropy.wcs.wcsapi.wrappers import SlicedLowLevelWCS

from ndcube import NDCube
from ndcube.tests import helpers


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


def test_wcs_type_after_init(ndcube_3d_ln_lt_l, wcs_3d_l_lt_ln):
    # Generate a low level WCS
    slices = np.s_[:, :, 0]
    low_level_wcs = SlicedLowLevelWCS(wcs_3d_l_lt_ln, slices)
    # Generate an NDCube using the low level WCS
    cube = NDCube(ndcube_3d_ln_lt_l.data[slices], low_level_wcs)
    # Check the WCS has been converted to high level but NDCube init.
    assert isinstance(cube.wcs, BaseHighLevelWCS)


def test_plot_docstring():
    cube = NDCube([], astropy.wcs.WCS())

    assert cube.plot.__doc__ == cube.plotter.plot.__doc__
    assert signature(cube.plot) == signature(cube.plotter.plot)
# This function is used in the arithmetic tests below


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


def test_to_nddata_no_wcs(ndcube_2d_ln_lt):
    ndc = ndcube_2d_ln_lt
    output = ndc.to_nddata(wcs=None)
    assert type(output) is astropy.nddata.NDData
    assert output.wcs is None
    assert (output.data == ndc.data).all()
