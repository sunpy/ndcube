import numpy as np
from astropy.wcs.wcsapi import BaseHighLevelWCS
from astropy.wcs.wcsapi.wrappers.base import BaseWCSWrapper

__all__ = ['ResampledLowLevelWCS']


class ResampledLowLevelWCS(BaseWCSWrapper):
    """
    A wrapper for a low-level WCS object that has down- or
    up-sampled pixel axes.

    Parameters
    ----------
    wcs : `~astropy.wcs.wcsapi.BaseLowLevelWCS`
        The original WCS for which to reorder axes

    factor : int or float or iterable
        The factor by which to increase the pixel size for each pixel
        axis. If a scalar, the same factor is used for all axes.
    """
    def __init__(self, wcs, factor):
        self._wcs = wcs
        if np.isscalar(factor):
            factor = np.array([factor] * self.pixel_n_dim)
        self._factor = factor

    def _top_to_underlying_pixels(self, top_pixels):
        # Convert user-facing pixel indices to the pixel grid of underlying WCS.
        # Additive factor makes sure the centre of the resampled pixel is being used.
        factor_shape = list(self._factor.shape) + [1] * (top_pixels.ndim - 1)
        factor = self._factor.reshape(factor_shape)
        return top_pixels * factor + (factor - 1) / 2

    def _underlying_to_top_pixels(self, underlying_pixels):
        # Convert pixel indices of underlying pixel grid to user-facing grid.
        # Subtractive factor makes sure the correct sub-pixel location is returned.
        factor_shape = list(self._factor.shape) + [1] * (underlying_pixels.ndim - 1)
        factor = self._factor.reshape(factor_shape)
        return (underlying_pixels - (factor - 1) / 2) / factor

    def pixel_to_world_values(self, *pixel_arrays):
        underlying_pixel_arrays = self._top_to_underlying_pixels(np.asarray(pixel_arrays))
        return self._wcs.pixel_to_world_values(*underlying_pixel_arrays)

    def world_to_pixel_values(self, *world_arrays):
        underlying_pixel_arrays = self._wcs.world_to_pixel_values(*world_arrays)
        top_pixel_arrays = self._underlying_to_top_pixels(np.asarray(underlying_pixel_arrays))
        return tuple(array for array in top_pixel_arrays)

    @property
    def pixel_shape(self):
        return tuple(np.asarray(self._wcs.pixel_shape) / self._factor)

    @property
    def pixel_bounds(self):
        if self._wcs.pixel_bounds is None:
            return self._wcs.pixel_bounds
        top_level_bounds = self._underlying_to_top_pixels(np.asarray(self._wcs.pixel_bounds))
        return [tuple(bounds) for bounds in top_level_bounds]
