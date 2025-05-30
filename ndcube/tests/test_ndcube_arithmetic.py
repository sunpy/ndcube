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
