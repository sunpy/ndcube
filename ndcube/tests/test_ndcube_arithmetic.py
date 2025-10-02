import numpy as np
import pytest

import astropy.units as u
import astropy.wcs
from astropy.nddata import NDData, StdDevUncertainty

from ndcube import NDCube
from ndcube.tests.helpers import assert_cubes_equal
from ndcube.utils.exceptions import NDCubeUserWarning


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


@pytest.mark.parametrize(("ndc", "value", "expected_kwargs"),
                        [(
                          "ndcube_2d_ln_lt_no_unit_no_unc_no_mask_2",
                          NDData(np.ones((2, 3)), wcs=None),
                          {"data": np.array([[1, 2, 3], [4, 5, 6]])}
                         ),
                         (
                          "ndcube_2d_ln_lt_no_unit_no_unc_no_mask_2",
                          NDData(np.ones((2, 3)),
                                 wcs=None,
                                 uncertainty=StdDevUncertainty(np.ones((2, 3))*0.1),
                                 mask=np.array([[True, False, False], [False, True, False]])),
                          {"data": np.array([[1, 2, 3], [4, 5, 6]]),
                           "uncertainty": astropy.nddata.StdDevUncertainty(np.ones((2, 3))*0.1),
                           "mask": np.array([[True, False, False], [False, True, False]])}
                         ), # ndc has no mask no uncertainty no unit, but nddata has all.
                         (
                          "ndcube_2d_ln_lt_unit_unc_mask",
                          NDData(np.ones((2, 3)), wcs=None, unit=u.ct),
                          {"data": np.array([[1, 2, 3], [4, 5, 6]]),
                           "uncertainty": astropy.nddata.StdDevUncertainty(np.array([[0, 0.05, 0.1],
                                                                                     [0.15, 0.2, 0.25]])),
                           "mask": np.array([[False, True, True], [False, True, True]])}
                         ), # ndc has mask and uncertainty unit, but nddata doesn't.
                         (
                          "ndcube_2d_ln_lt_unit_unc_mask",
                          NDData(np.ones((2, 3)),
                                 wcs=None,
                                 uncertainty=StdDevUncertainty(np.ones((2, 3))*0.1),
                                 mask=np.array([[True, False, False], [False, True, False]]),
                                 unit=u.ct),
                          {"unit": u.ct,
                           "data": np.array([[1, 2, 3], [4, 5, 6]]),
                           "uncertainty": astropy.nddata.StdDevUncertainty(np.array([[0.1       , 0.1118034 , 0.14142136],
                                                                                     [0.18027756, 0.2236068 , 0.26925824]])),
                           "mask": np.array([[True, True, True], [False, True, True]])}
                         ) # both of them have uncertainty and mask and unit.
                        ],
                        indirect=("ndc",))
def test_cube_arithmetic_add_nddata(ndc, value, expected_kwargs, wcs_2d_lt_ln):
    output_cube = ndc + value  # perform the multiplication

    expected_kwargs["wcs"] = wcs_2d_lt_ln
    expected_cube = NDCube(**expected_kwargs)

    # Assert output cube is same as expected cube
    assert_cubes_equal(output_cube, expected_cube, check_uncertainty_values=True)


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


@pytest.mark.parametrize(("ndc", "value", "expected_kwargs"),
                        [(
                          "ndcube_2d_ln_lt_no_unit_no_unc_no_mask_2",
                          NDData(np.ones((2, 3)), wcs=None),
                          {"data": np.array([[-1, 0, 1], [2, 3, 4]])}
                         ),
                         (
                          "ndcube_2d_ln_lt_no_unit_no_unc_no_mask_2",
                          NDData(np.ones((2, 3)),
                                 wcs=None,
                                 uncertainty=StdDevUncertainty(np.ones((2, 3))*0.1),
                                 mask=np.array([[True, False, False], [False, True, False]])),
                          {"data": np.array([[-1, 0, 1], [2, 3, 4]]),
                           "uncertainty": astropy.nddata.StdDevUncertainty(np.ones((2, 3))*0.1),
                           "mask": np.array([[True, False, False], [False, True, False]])}
                         ), # ndc has no mask no uncertainty no unit, but nddata has all.
                         (
                          "ndcube_2d_ln_lt_unit_unc_mask",
                          NDData(np.ones((2, 3)), wcs=None, unit=u.ct),
                          {"data": np.array([[-1, 0, 1], [2, 3, 4]]),
                           "uncertainty": astropy.nddata.StdDevUncertainty(np.array([[0, 0.05, 0.1],
                                                                                     [0.15, 0.2, 0.25]])),
                           "mask": np.array([[False, True, True], [False, True, True]])}
                         ), # ndc has mask and uncertainty unit, but nddata doesn't.
                         (
                          "ndcube_2d_ln_lt_unit_unc_mask",
                          NDData(np.ones((2, 3)),
                                 wcs=None,
                                 uncertainty=StdDevUncertainty(np.ones((2, 3))*0.1),
                                 mask=np.array([[True, False, False], [False, True, False]]),
                                 unit=u.ct),
                          {"unit": u.ct,
                           "data": np.array([[-1, 0, 1], [2, 3, 4]]),
                           "uncertainty": astropy.nddata.StdDevUncertainty(np.array([[0.1       , 0.1118034 , 0.14142136],
                                                                                     [0.18027756, 0.2236068 , 0.26925824]])),
                           "mask": np.array([[True, True, True], [False, True, True]])}
                         ) # both of them have uncertainty and mask and unit.
                        ],
                        indirect=("ndc",))
def test_cube_arithmetic_subtract_nddata(ndc, value, expected_kwargs, wcs_2d_lt_ln):
    output_cube = ndc - value  # perform the subtraction

    expected_kwargs["wcs"] = wcs_2d_lt_ln
    expected_cube = NDCube(**expected_kwargs)

    # Assert output cube is same as expected cube
    assert_cubes_equal(output_cube, expected_cube, check_uncertainty_values=True)


@pytest.mark.parametrize("value",
                        [
                         NDData(np.ones((8, 4)), wcs=None, unit=u.J)
                        ])
def test_cube_dask_arithmetic_subtract_nddata(ndcube_2d_dask, value):
    ndc = ndcube_2d_dask
    output_cube = ndc - value
    assert type(output_cube.data) is type(ndc.data)


@pytest.mark.parametrize("value",
                        [
                         NDData(np.ones((8, 4)), wcs=None, unit=u.J)
                        ])
def test_cube_dask_arithmetic_subtract_nddata(ndcube_2d_dask, value):
    ndc = ndcube_2d_dask
    output_cube = ndc - value
    assert type(output_cube.data) is type(ndc.data)


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
                        [(
                          "ndcube_2d_ln_lt_no_unit_no_unc_no_mask_2",
                          NDData(np.ones((2, 3)) + 1, wcs=None),
                          {"data": np.array([[0, 2, 4], [6, 8, 10]])}
                         ),
                         (
                          "ndcube_2d_ln_lt_no_unit_no_unc_no_mask_2",
                          NDData(np.ones((2, 3)) + 1,
                                 wcs=None,
                                 uncertainty=StdDevUncertainty((np.ones((2, 3)) + 1) * 0.1),
                                 mask=np.array([[True, False, False], [False, True, False]]),
                                 unit=u.ct),
                          {"data": np.array([[0, 2, 4], [6, 8, 10]]),
                           "uncertainty": astropy.nddata.StdDevUncertainty((np.ones((2, 3)) + 1) * 0.1),
                           "mask": np.array([[True, False, False], [False, True, False]]),
                           "unit": u.ct} # ndc has no mask no uncertainty no unit, but nddata has all.
                         ),
                         (
                          "ndcube_2d_ln_lt_unit_unc_mask",
                          NDData(np.ones((2, 3)) * 2, wcs=None),
                          {"data": np.array([[0, 2, 4], [6, 8, 10]]),
                           "uncertainty": astropy.nddata.StdDevUncertainty(np.array([[0, 0.05, 0.1],
                                                                                     [0.15, 0.2, 0.25]])),
                           "mask": np.array([[False, True, True], [False, True, True]])}
                         ), # ndc has mask and uncertainty unit, but nddata doesn't.
                         (
                         "ndcube_2d_ln_lt_unit_unc_mask",
                          NDData(np.ones((2, 3)) + 1,
                                 wcs=None,
                                 uncertainty=StdDevUncertainty((np.ones((2, 3)) + 1) * 0.1),
                                 mask=np.array([[True, False, False], [False, True, False]]),
                                 unit=u.ct),
                          {"unit": u.ct**2,
                           "data": np.array([[0, 2, 4], [6, 8, 10]]),
                           "uncertainty": astropy.nddata.StdDevUncertainty(np.array([[0.        , 0.2236068 , 0.4472136 ],
                                                                                     [0.67082039, 0.89442719, 1.11803399]])),
                           "mask": np.array([[True, True, True], [False, True, True]])}
                         ) # both of them have uncertainty and mask and unit.
                        ],
                        indirect=("ndc",))
def test_cube_arithmetic_multiply_nddata(ndc, value, expected_kwargs, wcs_2d_lt_ln):
    output_cube = ndc * value  # perform the multiplication

    expected_kwargs["wcs"] = wcs_2d_lt_ln
    expected_cube = NDCube(**expected_kwargs)

    # Assert output cube is same as expected cube
    assert_cubes_equal(output_cube, expected_cube, check_uncertainty_values=True)


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


@pytest.mark.parametrize(("ndc", "value", "expected_kwargs"),
                        [(
                          "ndcube_2d_ln_lt_no_unit_no_unc_no_mask_2",
                          NDData(np.ones((2, 3)) + 1, wcs=None),
                          {"data": np.array([[0, 0.5, 1], [1.5, 2, 2.5]])},
                         ),
                         (
                          "ndcube_2d_ln_lt_no_unit_no_unc_no_mask_2",
                          NDData(np.ones((2, 3)) + 1,
                                 wcs=None,
                                 uncertainty=StdDevUncertainty((np.ones((2, 3)) + 1) * 0.1),
                                 mask=np.array([[True, False, False], [False, True, False]]),
                                 unit=u.ct),
                           {"data": np.array([[0, 0.5, 1], [1.5, 2, 2.5]]),
                           "uncertainty": astropy.nddata.StdDevUncertainty((np.ones((2, 3)) + 1) * 0.1),
                           "mask": np.array([[True, False, False], [False, True, False]]),
                           "unit": u.dimensionless_unscaled / u.ct} # ndc has no mask no uncertainty no unit, but nddata has all.
                         ),
                         (
                          "ndcube_2d_ln_lt_unit_unc_mask",
                          NDData(np.ones((2, 3)) * 2, wcs=None),
                           {"data": np.array([[0, 0.5, 1], [1.5, 2, 2.5]]),
                           "uncertainty": astropy.nddata.StdDevUncertainty(np.array([[0, 0.05, 0.1],
                                                                                     [0.15, 0.2, 0.25]])),
                           "mask": np.array([[False, True, True], [False, True, True]])}
                         ), # ndc has mask, uncertainty and unit, but nddata doesn't.
                         (
                         "ndcube_2d_ln_lt_unit_unc_mask",
                          NDData(np.ones((2, 3)) + 1,
                                 wcs=None,
                                 uncertainty=StdDevUncertainty((np.ones((2, 3)) + 1) * 0.1),
                                 mask=np.array([[True, False, False], [False, True, False]]),
                                 unit=u.ct),
                          {"unit": u.dimensionless_unscaled,
                           "data": np.array([[0, 0.5, 1], [1.5, 2, 2.5]]),
                           "uncertainty": astropy.nddata.StdDevUncertainty(np.array([[0.       , 0.0559017, 0.1118034],
                                                                                     [0.1677051, 0.2236068, 0.2795085]])),
                           "mask": np.array([[True, True, True], [False, True, True]])}
                         ) # both of them have uncertainty and mask and unit.
                        ],
                        indirect=("ndc",))
def test_cube_arithmetic_divide_nddata(ndc, value, expected_kwargs, wcs_2d_lt_ln):
    output_cube = ndc / value  # perform the division

    expected_kwargs["wcs"] = wcs_2d_lt_ln
    expected_cube = NDCube(**expected_kwargs)

    # Assert output cube is same as expected cube
    assert_cubes_equal(output_cube, expected_cube, check_uncertainty_values=True)


@pytest.mark.parametrize("value",
                        [
                         NDData(np.ones((8, 4)) * 2, wcs=None)
                        ])
def test_cube_dask_arithmetic_divide_nddata(ndcube_2d_dask, value):
    ndc = ndcube_2d_dask
    output_cube = ndc / value
    assert type(output_cube.data) is type(ndc.data)


@pytest.mark.parametrize('value', [1, 2, -1])
def test_cube_arithmetic_rdivide(ndcube_2d_ln_lt_units, value):
    cube_quantity = u.Quantity(ndcube_2d_ln_lt_units.data, ndcube_2d_ln_lt_units.unit)
    with np.errstate(divide='ignore'):
        new_cube =  value / ndcube_2d_ln_lt_units
        check_arithmetic_value_and_units(new_cube,  value / cube_quantity)


@pytest.mark.parametrize('value', [1, 2, -1])
def test_cube_arithmetic_rdivide_uncertainty(ndcube_4d_unit_uncertainty, value):
    cube_quantity = u.Quantity(ndcube_4d_unit_uncertainty.data, ndcube_4d_unit_uncertainty.unit)
    match = "UnknownUncertainty does not support uncertainty propagation with correlation. Setting uncertainties to None."
    with pytest.warns(NDCubeUserWarning, match=match):  # noqa: PT031
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
    match = "UnknownUncertainty does not support uncertainty propagation with correlation. Setting uncertainties to None."
    with pytest.warns(NDCubeUserWarning, match=match):  # noqa: PT031
        with np.errstate(divide='ignore'):
            new_cube = ndcube_4d_unit_uncertainty ** power
            check_arithmetic_value_and_units(new_cube, cube_quantity**power)


@pytest.mark.parametrize('power', [2, -2, 10, 0.5])
def test_cube_arithmetic_power_std_uncertainty(ndcube_2d_ln_lt_uncert, power):
    cube_quantity = u.Quantity(ndcube_2d_ln_lt_uncert.data, ndcube_2d_ln_lt_uncert.unit)
    match = r"<class 'astropy.nddata.nduncertainty.StdDevUncertainty'> does not support propagation of uncertainties for power. Setting uncertainties to None."
    with pytest.warns(NDCubeUserWarning, match=match):  # noqa: PT031
        with np.errstate(divide='ignore'):
            new_cube = ndcube_2d_ln_lt_uncert ** power
            check_arithmetic_value_and_units(new_cube, cube_quantity**power)
