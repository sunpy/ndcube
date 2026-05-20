"""
Tests to simulate dynamic spectrum WCSes (frequency x time).
"""
from numpy.testing import assert_allclose

import astropy.units as u

from ndcube.wcs.wrappers import ResampledLowLevelWCS


def _world_at(cube, time_pixel, freq_pixel):
    return cube.wcs.low_level_wcs.pixel_to_world_values(time_pixel, freq_pixel)


def test_linear_dynspec_array_axis_physical_types(ndcube_gwcs_2d_t_f_linear):
    types = ndcube_gwcs_2d_t_f_linear.array_axis_physical_types
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


def test_linear_dynspec_rebin_freq_shape(ndcube_gwcs_2d_t_f_linear):
    assert ndcube_gwcs_2d_t_f_linear.rebin((2, 1)).shape == (8, 10)


def test_linear_dynspec_rebin_freq_wcs(ndcube_gwcs_2d_t_f_linear):
    rebinned = ndcube_gwcs_2d_t_f_linear.rebin((2, 1))
    _, freq0 = rebinned.wcs.low_level_wcs.pixel_to_world_values(0, 0)
    assert_allclose(freq0, 0.5e6)


def test_linear_dynspec_rebin_time_shape(ndcube_gwcs_2d_t_f_linear):
    assert ndcube_gwcs_2d_t_f_linear.rebin((1, 2)).shape == (16, 5)


def test_linear_dynspec_rebin_time_wcs(ndcube_gwcs_2d_t_f_linear):
    rebinned = ndcube_gwcs_2d_t_f_linear.rebin((1, 2))
    time0, _ = rebinned.wcs.low_level_wcs.pixel_to_world_values(0, 0)
    assert_allclose(time0, 7.0)


def test_linear_dynspec_rebin_wcs_is_resampled(ndcube_gwcs_2d_t_f_linear):
    assert isinstance(ndcube_gwcs_2d_t_f_linear.rebin((2, 2)).wcs.low_level_wcs,
                      ResampledLowLevelWCS)


def test_linear_dynspec_crop_by_freq_shape(ndcube_gwcs_2d_t_f_linear):
    cropped = ndcube_gwcs_2d_t_f_linear.crop_by_values([None, 3e6 * u.Hz],
                                                        [None, 7e6 * u.Hz])

    assert cropped.shape == (5, 10)


def test_linear_dynspec_crop_by_time_shape(ndcube_gwcs_2d_t_f_linear):
    cropped = ndcube_gwcs_2d_t_f_linear.crop_by_values([14 * u.s, None],
                                                        [56 * u.s, None])
    assert cropped.shape == (16, 4)


def test_log_dynspec_array_axis_physical_types(ndcube_gwcs_2d_t_f_log):
    types = ndcube_gwcs_2d_t_f_log.array_axis_physical_types
    assert "em.freq" in types[0]
    assert "time" in types[1]


def test_log_dynspec_world_axis_units(ndcube_gwcs_2d_t_f_log):
    assert ndcube_gwcs_2d_t_f_log.wcs.world_axis_units == ("s", "Hz")


def test_log_dynspec_pixel_to_world_origin(ndcube_gwcs_2d_t_f_log):
    time, freq = ndcube_gwcs_2d_t_f_log.wcs.low_level_wcs.pixel_to_world_values(0, 0)
    assert_allclose(time, 0.0)
    assert_allclose(freq, 3.992e6, rtol=1e-6)


def test_log_dynspec_pixel_to_world_last(ndcube_gwcs_2d_t_f_log):
    time, freq = ndcube_gwcs_2d_t_f_log.wcs.low_level_wcs.pixel_to_world_values(9, 15)
    assert_allclose(time, 122.5)
    assert_allclose(freq, 978.572e6, rtol=1e-6)


def test_log_dynspec_world_to_pixel_roundtrip(ndcube_gwcs_2d_t_f_log):
    time, freq = _world_at(ndcube_gwcs_2d_t_f_log, 3, 7)
    pix_t, pix_f = ndcube_gwcs_2d_t_f_log.wcs.low_level_wcs.world_to_pixel_values(
        time, freq)
    assert_allclose(pix_t, 3.0, atol=1e-10)
    assert_allclose(pix_f, 7.0, atol=1e-10)


def test_log_dynspec_rebin_freq_shape(ndcube_gwcs_2d_t_f_log):
    assert ndcube_gwcs_2d_t_f_log.rebin((2, 1)).shape == (8, 10)


def test_log_dynspec_rebin_freq_wcs_midpoint(ndcube_gwcs_2d_t_f_log):
    rebinned = ndcube_gwcs_2d_t_f_log.rebin((2, 1))
    _, freq0 = rebinned.wcs.low_level_wcs.pixel_to_world_values(0, 0)
    _, freq_left = _world_at(ndcube_gwcs_2d_t_f_log, 0, 0)
    _, freq_right = _world_at(ndcube_gwcs_2d_t_f_log, 0, 1)
    expected = (freq_left + freq_right) / 2
    assert_allclose(freq0, expected, rtol=1e-6)


def test_log_dynspec_rebin_time_shape(ndcube_gwcs_2d_t_f_log):
    assert ndcube_gwcs_2d_t_f_log.rebin((1, 2)).shape == (16, 5)


def test_log_dynspec_rebin_time_wcs_midpoint(ndcube_gwcs_2d_t_f_log):
    rebinned = ndcube_gwcs_2d_t_f_log.rebin((1, 2))
    time0, _ = rebinned.wcs.low_level_wcs.pixel_to_world_values(0, 0)
    time_left, _ = _world_at(ndcube_gwcs_2d_t_f_log, 0, 0)
    time_right, _ = _world_at(ndcube_gwcs_2d_t_f_log, 1, 0)
    expected = (time_left + time_right) / 2
    assert_allclose(time0, expected, rtol=1e-6)


def test_log_dynspec_rebin_wcs_is_resampled(ndcube_gwcs_2d_t_f_log):
    assert isinstance(ndcube_gwcs_2d_t_f_log.rebin((2, 2)).wcs.low_level_wcs,
                      ResampledLowLevelWCS)


def test_log_dynspec_crop_by_freq_shape(ndcube_gwcs_2d_t_f_log):
    cropped = ndcube_gwcs_2d_t_f_log.crop_by_values(
        [None, 10e6 * u.Hz], [None, 100e6 * u.Hz])
    assert cropped.shape == (8, 10)


def test_log_dynspec_crop_by_freq_bounds(ndcube_gwcs_2d_t_f_log):
    lo, hi = 10e6, 100e6
    cropped = ndcube_gwcs_2d_t_f_log.crop_by_values(
        [None, lo * u.Hz], [None, hi * u.Hz])
    freqs = [cropped.wcs.low_level_wcs.pixel_to_world_values(0, i)[1]
             for i in range(cropped.shape[0])]
    assert freqs[0] <= lo
    assert freqs[-1] >= hi


def test_log_dynspec_crop_by_time_shape(ndcube_gwcs_2d_t_f_log):
    cropped = ndcube_gwcs_2d_t_f_log.crop_by_values(
        [20 * u.s, None], [80 * u.s, None])
    assert cropped.shape == (16, 6)


def test_log_dynspec_crop_by_time_bounds(ndcube_gwcs_2d_t_f_log):
    lo, hi = 20.0, 80.0
    cropped = ndcube_gwcs_2d_t_f_log.crop_by_values(
        [lo * u.s, None], [hi * u.s, None])
    times = [cropped.wcs.low_level_wcs.pixel_to_world_values(i, 0)[0]
             for i in range(cropped.shape[1])]
    assert times[0] <= lo
    assert times[-1] >= hi


def test_log_dynspec_crop_by_freq_and_time(ndcube_gwcs_2d_t_f_log):
    cropped = ndcube_gwcs_2d_t_f_log.crop_by_values(
        [20 * u.s, 10e6 * u.Hz], [80 * u.s, 100e6 * u.Hz])
    assert cropped.shape == (8, 6)
