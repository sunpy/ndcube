"""
Tests to simulate dynamic spectrum WCSes (frequency x time).
"""
import pytest
from numpy.testing import assert_allclose

import astropy.units as u

from ndcube.wcs.wrappers import ResampledLowLevelWCS


def _world_at(cube, time_pixel, freq_pixel):
    return cube.wcs.low_level_wcs.pixel_to_world_values(time_pixel, freq_pixel)


@pytest.mark.parametrize("ndc", [
    "ndcube_gwcs_2d_t_f_linear",
    "ndcube_gwcs_2d_t_f_log",
], indirect=True)
def test_dynspec_array_axis_physical_types(ndc):
    types = ndc.array_axis_physical_types
    assert "em.freq" in types[0]
    assert "time" in types[1]


def test_linear_dynspec_pixel_to_world(ndcube_gwcs_2d_t_f_linear):
    time, freq = ndcube_gwcs_2d_t_f_linear.wcs.low_level_wcs.pixel_to_world_values(3, 2)
    assert_allclose(time, 42.0)
    assert_allclose(freq, 2e6)


def test_linear_dynspec_world_to_pixel(ndcube_gwcs_2d_t_f_linear):
    pix_t, pix_f = ndcube_gwcs_2d_t_f_linear.wcs.low_level_wcs.world_to_pixel_values(28.0, 4e6)
    assert_allclose(pix_t, 2.0)
    assert_allclose(pix_f, 4.0)


@pytest.mark.parametrize(("bin_shape", "expected_shape", "expected_time", "expected_freq"), [
    ((2, 1), (8, 10), 0.0, 0.5e6),
    ((1, 2), (16, 5), 7.0, 0.0),
])
def test_linear_dynspec_rebin_wcs(ndcube_gwcs_2d_t_f_linear, bin_shape,
                                  expected_shape, expected_time, expected_freq):
    rebinned = ndcube_gwcs_2d_t_f_linear.rebin(bin_shape)
    time0, freq0 = rebinned.wcs.low_level_wcs.pixel_to_world_values(0, 0)

    assert rebinned.shape == expected_shape
    assert isinstance(rebinned.wcs.low_level_wcs, ResampledLowLevelWCS)
    assert_allclose(time0, expected_time)
    assert_allclose(freq0, expected_freq)


@pytest.mark.parametrize(("lower_corner", "upper_corner", "expected_shape"), [
    ([None, 3e6 * u.Hz], [None, 7e6 * u.Hz], (5, 10)),
    ([14 * u.s, None], [56 * u.s, None], (16, 4)),
])
def test_linear_dynspec_crop_by_values_shape(ndcube_gwcs_2d_t_f_linear,
                                             lower_corner, upper_corner,
                                             expected_shape):
    cropped = ndcube_gwcs_2d_t_f_linear.crop_by_values(lower_corner, upper_corner)
    assert cropped.shape == expected_shape


def test_log_dynspec_world_axis_units(ndcube_gwcs_2d_t_f_log):
    assert ndcube_gwcs_2d_t_f_log.wcs.world_axis_units == ("s", "Hz")


@pytest.mark.parametrize(("time_pixel", "freq_pixel", "expected_time", "expected_freq"), [
    (0, 0, 0.0, 3.992e6),
    (9, 15, 122.5, 978.572e6),
])
def test_log_dynspec_pixel_to_world_endpoints(ndcube_gwcs_2d_t_f_log,
                                              time_pixel, freq_pixel,
                                              expected_time, expected_freq):
    time, freq = ndcube_gwcs_2d_t_f_log.wcs.low_level_wcs.pixel_to_world_values(
        time_pixel, freq_pixel)
    assert_allclose(time, expected_time)
    assert_allclose(freq, expected_freq, rtol=1e-6)


def test_log_dynspec_world_to_pixel_roundtrip(ndcube_gwcs_2d_t_f_log):
    time, freq = _world_at(ndcube_gwcs_2d_t_f_log, 3, 7)
    pix_t, pix_f = ndcube_gwcs_2d_t_f_log.wcs.low_level_wcs.world_to_pixel_values(
        time, freq)
    assert_allclose(pix_t, 3.0, atol=1e-10)
    assert_allclose(pix_f, 7.0, atol=1e-10)


@pytest.mark.parametrize(("bin_shape", "expected_shape", "axis"), [
    ((2, 1), (8, 10), "freq"),
    ((1, 2), (16, 5), "time"),
])
def test_log_dynspec_rebin_wcs_midpoint(ndcube_gwcs_2d_t_f_log, bin_shape,
                                        expected_shape, axis):
    rebinned = ndcube_gwcs_2d_t_f_log.rebin(bin_shape)
    time0, freq0 = rebinned.wcs.low_level_wcs.pixel_to_world_values(0, 0)

    assert rebinned.shape == expected_shape
    assert isinstance(rebinned.wcs.low_level_wcs, ResampledLowLevelWCS)
    if axis == "freq":
        _, freq_left = _world_at(ndcube_gwcs_2d_t_f_log, 0, 0)
        _, freq_right = _world_at(ndcube_gwcs_2d_t_f_log, 0, 1)
        assert_allclose(freq0, (freq_left + freq_right) / 2, rtol=1e-6)
    else:
        time_left, _ = _world_at(ndcube_gwcs_2d_t_f_log, 0, 0)
        time_right, _ = _world_at(ndcube_gwcs_2d_t_f_log, 1, 0)
        assert_allclose(time0, (time_left + time_right) / 2, rtol=1e-6)


@pytest.mark.parametrize(("lower_corner", "upper_corner", "expected_shape",
                          "axis", "bounds"), [
    ([None, 10e6 * u.Hz], [None, 100e6 * u.Hz], (8, 10), "freq", (10e6, 100e6)),
    ([20 * u.s, None], [80 * u.s, None], (16, 6), "time", (20.0, 80.0)),
])
def test_log_dynspec_crop_by_values_single_axis(ndcube_gwcs_2d_t_f_log,
                                                lower_corner, upper_corner,
                                                expected_shape, axis, bounds):
    cropped = ndcube_gwcs_2d_t_f_log.crop_by_values(lower_corner, upper_corner)
    assert cropped.shape == expected_shape

    if axis == "freq":
        values = [cropped.wcs.low_level_wcs.pixel_to_world_values(0, i)[1]
                  for i in range(cropped.shape[0])]
    else:
        values = [cropped.wcs.low_level_wcs.pixel_to_world_values(i, 0)[0]
                  for i in range(cropped.shape[1])]

    assert values[0] <= bounds[0]
    assert values[-1] >= bounds[1]


def test_log_dynspec_crop_by_freq_and_time(ndcube_gwcs_2d_t_f_log):
    cropped = ndcube_gwcs_2d_t_f_log.crop_by_values(
        [20 * u.s, 10e6 * u.Hz], [80 * u.s, 100e6 * u.Hz])
    assert cropped.shape == (8, 6)
