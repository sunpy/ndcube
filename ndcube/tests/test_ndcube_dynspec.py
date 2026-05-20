"""
Tests for NDCube with dynamic spectrum WCS (frequency x time).

Convention throughout: array shape (n_freq, n_time) so that when plotted
as a 2D image frequency varies along rows (Y axis) and time along columns
(X axis), matching standard radio astronomy dynamic spectrum displays.

WCS pixel axis ordering (reversed from array):
  pixel axis 0 -> time   (array axis 1, X axis)
  pixel axis 1 -> freq   (array axis 0, Y axis)

world_axis_units = ('s', 'Hz') and pixel_to_world_values(p_time, p_freq)
returns (time_s, freq_hz).
"""
import numpy as np
from numpy.testing import assert_allclose

import astropy.units as u

from ndcube.wcs.wrappers import ResampledLowLevelWCS

# Pre-computed from fixture definitions; kept here so tests are self-documenting.
_FREQS_LOG_HZ = np.logspace(np.log10(3.992e6), np.log10(978.572e6), 16)
_TIMES_S = np.array([0.0, 14.0, 27.4, 41.1, 55.2, 67.8, 82.3, 95.9, 109.1, 122.5])


class TestLinearDynspec:
    """Linear Scale gWCS: 14 s/pixel time, 1 MHz/pixel freq."""

    def test_array_axis_physical_types(self, ndcube_gwcs_2d_t_f_linear):
        types = ndcube_gwcs_2d_t_f_linear.array_axis_physical_types
        assert 'em.freq' in types[0]   # array axis 0 = rows = Y axis
        assert 'time'    in types[1]   # array axis 1 = cols = X axis

    def test_pixel_to_world(self, ndcube_gwcs_2d_t_f_linear):
        # pixel_to_world_values(time_pixel, freq_pixel) -> (time_s, freq_hz)
        time, freq = ndcube_gwcs_2d_t_f_linear.wcs.low_level_wcs.pixel_to_world_values(3, 2)
        assert_allclose(time, 42.0)  # 3 * 14 s/pixel
        assert_allclose(freq, 2e6)   # 2 * 1 MHz/pixel

    def test_world_to_pixel(self, ndcube_gwcs_2d_t_f_linear):
        # world_to_pixel_values(time_s, freq_hz) -> (time_pixel, freq_pixel)
        pix_t, pix_f = ndcube_gwcs_2d_t_f_linear.wcs.low_level_wcs.world_to_pixel_values(28.0, 4e6)
        assert_allclose(pix_t, 2.0)
        assert_allclose(pix_f, 4.0)

    def test_rebin_freq_shape(self, ndcube_gwcs_2d_t_f_linear):
        # rebin(2, 1): bin 2 freq rows, keep time cols -> (8, 10)
        assert ndcube_gwcs_2d_t_f_linear.rebin((2, 1)).shape == (8, 10)

    def test_rebin_freq_wcs(self, ndcube_gwcs_2d_t_f_linear):
        rebinned = ndcube_gwcs_2d_t_f_linear.rebin((2, 1))
        _, freq0 = rebinned.wcs.low_level_wcs.pixel_to_world_values(0, 0)
        assert_allclose(freq0, 0.5e6)  # midpoint of freq pixels 0 and 1

    def test_rebin_time_shape(self, ndcube_gwcs_2d_t_f_linear):
        # rebin(1, 2): keep freq rows, bin 2 time cols -> (16, 5)
        assert ndcube_gwcs_2d_t_f_linear.rebin((1, 2)).shape == (16, 5)

    def test_rebin_time_wcs(self, ndcube_gwcs_2d_t_f_linear):
        rebinned = ndcube_gwcs_2d_t_f_linear.rebin((1, 2))
        time0, _ = rebinned.wcs.low_level_wcs.pixel_to_world_values(0, 0)
        assert_allclose(time0, 7.0)  # midpoint of 0 and 14 s

    def test_rebin_wcs_is_resampled(self, ndcube_gwcs_2d_t_f_linear):
        assert isinstance(ndcube_gwcs_2d_t_f_linear.rebin((2, 2)).wcs.low_level_wcs,
                          ResampledLowLevelWCS)

    def test_crop_by_freq_shape(self, ndcube_gwcs_2d_t_f_linear):
        # world order: (time, freq); freq crop reduces rows
        # 3–7 MHz = freq pixels 3,4,5,6,7 -> 5 freq rows
        cropped = ndcube_gwcs_2d_t_f_linear.crop_by_values([None, 3e6 * u.Hz],
                                                            [None, 7e6 * u.Hz])
        assert cropped.shape == (5, 10)

    def test_crop_by_time_shape(self, ndcube_gwcs_2d_t_f_linear):
        # time crop reduces cols
        # 14–56 s = time pixels 1,2,3,4 -> 4 time cols
        cropped = ndcube_gwcs_2d_t_f_linear.crop_by_values([14 * u.s, None],
                                                            [56 * u.s, None])
        assert cropped.shape == (16, 4)


class TestLogDynspec:
    """Log-spaced Tabular1D gWCS: synthetic metric-range frequency, irregular time."""

    def test_array_axis_physical_types(self, ndcube_gwcs_2d_t_f_log):
        types = ndcube_gwcs_2d_t_f_log.array_axis_physical_types
        assert 'em.freq' in types[0]   # array axis 0 = rows = Y axis
        assert 'time'    in types[1]   # array axis 1 = cols = X axis

    def test_world_axis_units(self, ndcube_gwcs_2d_t_f_log):
        units = ndcube_gwcs_2d_t_f_log.wcs.world_axis_units
        assert units[0] == 's'    # world axis 0 = time
        assert units[1] == 'Hz'   # world axis 1 = freq

    def test_pixel_to_world_origin(self, ndcube_gwcs_2d_t_f_log):
        # pixel_to_world_values(time_pixel, freq_pixel) -> (time_s, freq_hz)
        time, freq = ndcube_gwcs_2d_t_f_log.wcs.low_level_wcs.pixel_to_world_values(0, 0)
        assert_allclose(time, _TIMES_S[0])
        assert_allclose(freq, _FREQS_LOG_HZ[0], rtol=1e-6)

    def test_pixel_to_world_last(self, ndcube_gwcs_2d_t_f_log):
        time, freq = ndcube_gwcs_2d_t_f_log.wcs.low_level_wcs.pixel_to_world_values(9, 15)
        assert_allclose(time, _TIMES_S[9])
        assert_allclose(freq, _FREQS_LOG_HZ[15], rtol=1e-6)

    def test_world_to_pixel_roundtrip(self, ndcube_gwcs_2d_t_f_log):
        # world_to_pixel_values(time_s, freq_hz) -> (time_pixel, freq_pixel)
        pix_t, pix_f = ndcube_gwcs_2d_t_f_log.wcs.low_level_wcs.world_to_pixel_values(
            _TIMES_S[3], _FREQS_LOG_HZ[7])
        assert_allclose(pix_t, 3.0, atol=1e-10)
        assert_allclose(pix_f, 7.0, atol=1e-10)

    def test_rebin_freq_shape(self, ndcube_gwcs_2d_t_f_log):
        # rebin(2, 1): bin 2 freq rows -> (8, 10)
        assert ndcube_gwcs_2d_t_f_log.rebin((2, 1)).shape == (8, 10)

    def test_rebin_freq_wcs_midpoint(self, ndcube_gwcs_2d_t_f_log):
        # ResampledLowLevelWCS shifts pixel centres by 0.5; Tabular1D returns
        # the linearly interpolated value at the midpoint of the binned pixels.
        rebinned = ndcube_gwcs_2d_t_f_log.rebin((2, 1))
        _, freq0 = rebinned.wcs.low_level_wcs.pixel_to_world_values(0, 0)
        expected = (_FREQS_LOG_HZ[0] + _FREQS_LOG_HZ[1]) / 2
        assert_allclose(freq0, expected, rtol=1e-6)

    def test_rebin_time_shape(self, ndcube_gwcs_2d_t_f_log):
        # rebin(1, 2): bin 2 time cols -> (16, 5)
        assert ndcube_gwcs_2d_t_f_log.rebin((1, 2)).shape == (16, 5)

    def test_rebin_time_wcs_midpoint(self, ndcube_gwcs_2d_t_f_log):
        rebinned = ndcube_gwcs_2d_t_f_log.rebin((1, 2))
        time0, _ = rebinned.wcs.low_level_wcs.pixel_to_world_values(0, 0)
        expected = (_TIMES_S[0] + _TIMES_S[1]) / 2
        assert_allclose(time0, expected, rtol=1e-6)

    def test_rebin_wcs_is_resampled(self, ndcube_gwcs_2d_t_f_log):
        assert isinstance(ndcube_gwcs_2d_t_f_log.rebin((2, 2)).wcs.low_level_wcs,
                          ResampledLowLevelWCS)

    def test_crop_by_freq_shape(self, ndcube_gwcs_2d_t_f_log):
        # world order: (time, freq); freq crop reduces rows
        # 10–100 MHz selects 8 freq rows (bounding box including boundary pixels)
        cropped = ndcube_gwcs_2d_t_f_log.crop_by_values(
            [None, 10e6 * u.Hz], [None, 100e6 * u.Hz])
        assert cropped.shape == (8, 10)

    def test_crop_by_freq_bounds(self, ndcube_gwcs_2d_t_f_log):
        # Crop returns the smallest pixel bounding box spanning the world range.
        # First and last pixels may fall outside [lo, hi]; the full range is covered.
        lo, hi = 10e6, 100e6
        cropped = ndcube_gwcs_2d_t_f_log.crop_by_values(
            [None, lo * u.Hz], [None, hi * u.Hz])
        freqs = [cropped.wcs.low_level_wcs.pixel_to_world_values(0, i)[1]
                 for i in range(cropped.shape[0])]
        assert freqs[0] <= lo      # first channel at or below lower bound
        assert freqs[-1] >= hi     # last channel at or above upper bound

    def test_crop_by_time_shape(self, ndcube_gwcs_2d_t_f_log):
        # time crop reduces cols; 20–80 s spans 6 time steps (14.0..82.3 s)
        cropped = ndcube_gwcs_2d_t_f_log.crop_by_values(
            [20 * u.s, None], [80 * u.s, None])
        assert cropped.shape == (16, 6)

    def test_crop_by_time_bounds(self, ndcube_gwcs_2d_t_f_log):
        lo, hi = 20.0, 80.0
        cropped = ndcube_gwcs_2d_t_f_log.crop_by_values(
            [lo * u.s, None], [hi * u.s, None])
        times = [cropped.wcs.low_level_wcs.pixel_to_world_values(i, 0)[0]
                 for i in range(cropped.shape[1])]
        assert times[0] <= lo
        assert times[-1] >= hi

    def test_crop_by_freq_and_time(self, ndcube_gwcs_2d_t_f_log):
        # world order (time, freq)
        cropped = ndcube_gwcs_2d_t_f_log.crop_by_values(
            [20 * u.s, 10e6 * u.Hz], [80 * u.s, 100e6 * u.Hz])
        assert cropped.shape == (8, 6)
