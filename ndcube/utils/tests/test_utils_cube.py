
import numpy as np
import pytest

from astropy.nddata import StdDevUncertainty

from ndcube.utils.cube import propagate_rebin_uncertainties


@pytest.fixture
def stacked_pixel_data():
    return np.array([[[0, 2, 4], [12, 14, 16]],
                     [[1, 3, 5], [13, 15, 17]],
                     [[6, 8, 10], [18, 20, 22]],
                     [[7, 9, 11], [19, 21, 23]]], dtype=float) + 1


def test_propagate_rebin_uncertainties_mean(stacked_pixel_data):
    # Build inputs
    data = stacked_pixel_data
    mask = data < 0
    mask[3, 0, 0] = True
    uncertainty = StdDevUncertainty(data * 0.1)

    # Build expected output
    expected = np.sqrt((uncertainty.array**2).sum(axis=0)) / data.shape[0]
    expected[0, 0] = np.sqrt((uncertainty.array[:3, 0, 0]**2).sum()) / 3
    expected = StdDevUncertainty(expected)

    # Run function
    output = propagate_rebin_uncertainties(uncertainty, data, mask,
                                           np.mean, operation_ignores_mask=False)

    assert type(output) is type(expected)
    assert np.allclose(output.array, expected.array)


def test_propagate_rebin_uncertainties_use_masked(stacked_pixel_data):
    # Build inputs
    data = stacked_pixel_data
    mask = data < 0
    mask[3, 0, 0] = True
    uncertainty = StdDevUncertainty(data * 0.1)

    # Build expected output
    expected = np.sqrt((uncertainty.array**2).sum(axis=0)) / data.shape[0]
    expected = StdDevUncertainty(expected)

    # Run function
    output = propagate_rebin_uncertainties(uncertainty, data, mask,
                                           np.mean, operation_ignores_mask=True)

    assert type(output) is type(expected)
    assert np.allclose(output.array, expected.array)


@pytest.mark.parametrize("operation", [np.mean, np.nanmean])
@pytest.mark.parametrize("readonly", [False, True])
def test_propagate_rebin_uncertainties_preserves_inputs(stacked_pixel_data, operation, readonly):
    data = stacked_pixel_data
    mask = np.zeros(data.shape, dtype=bool)
    mask[3, 0, 0] = True
    if operation is np.nanmean:
        data[3, 0, 1] = np.nan
    uncertainty = StdDevUncertainty(data * 0.1)
    originals = [array.copy() for array in (data, uncertainty.array, mask)]
    if readonly:
        for array in (data, uncertainty.array, mask):
            array.flags.writeable = False

    propagate_rebin_uncertainties(uncertainty, data, mask, operation)

    for array, original in zip((data, uncertainty.array, mask), originals):
        np.testing.assert_array_equal(array, original)


def test_propagate_rebin_uncertainties_prod(stacked_pixel_data):
    # Build inputs
    data = stacked_pixel_data
    mask = data < 0
    uncertainty = StdDevUncertainty(data * 0.1)

    # Build expected output
    binned_data = data.prod(axis=0)
    # TODO: Why do I have to divide by a factor 2 here?
    expected = np.sqrt(((uncertainty.array / data)**2).sum(axis=0)) * binned_data / 2
    expected = StdDevUncertainty(expected)

    # Run function
    output = propagate_rebin_uncertainties(uncertainty, data, mask,
                                           np.prod, operation_ignores_mask=False)

    assert type(output) is type(expected)
    assert np.allclose(output.array, expected.array)


def test_propagate_rebin_uncertainties_nan(stacked_pixel_data):
    # Build inputs
    data = stacked_pixel_data
    data[3, 0, 0] = np.nan
    mask = data < 0
    uncertainty = StdDevUncertainty(data * 0.1)

    # Build expected output
    expected = np.sqrt((uncertainty.array**2).sum(axis=0)) / data.shape[0]
    expected[0, 0] = np.sqrt((uncertainty.array[:3, 0, 0]**2).sum()) / 3
    expected = StdDevUncertainty(expected)

    # Run function
    output = propagate_rebin_uncertainties(uncertainty, data, mask,
                                           np.nanmean, operation_ignores_mask=False)

    assert type(output) is type(expected)
    assert np.allclose(output.array, expected.array)

    # Test another code path that does the same thing.
    mask = None
    output = propagate_rebin_uncertainties(uncertainty, data, mask,
                                           np.nanmean, operation_ignores_mask=False)
    assert type(output) is type(expected)
    assert np.allclose(output.array, expected.array)


def test_propagate_rebin_uncertainties_nanmean_without_nans(stacked_pixel_data):
    uncertainty = StdDevUncertainty(stacked_pixel_data * 0.1)
    expected = np.sqrt((uncertainty.array**2).sum(axis=0)) / stacked_pixel_data.shape[0]

    output = propagate_rebin_uncertainties(uncertainty, stacked_pixel_data, None, np.nanmean)

    np.testing.assert_allclose(output.array, expected)
