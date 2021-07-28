import numpy as np
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

    def _top_to_underlying_pixels(top_pixels):
        # Convert user-facing pixel indices to the pixel grid of underlying WCS.
        # Additive factor makes sure the centre of the resampled pixel is being used.
        return top_pixels * self._factor + (self._factor - 1) / 2

    def _underlying_to_top_pixels(underlying_pixels):
        # Convert pixel indices of underlying pixel grid to user-facing grid.
        # Subtractive factor makes sure the correct sub-pixel location is returned.
        return (underlying_pixels - (self._factor - 1) / 2) / self._factor

    def pixel_to_world_values(self, *pixel_arrays):
        underlying_pixel_arrays = self._top_to_underlying_pixels(np.asarray(pixel_arrays))
        return self._wcs.pixel_to_world_values(*underlying_pixel_arrays)

    def world_to_pixel_values(self, *world_arrays):
        underlying_pixel_arrays = self._wcs.world_to_pixel(*world_arrays)
        return self._underyling_to_top_pixels(np.asarray(underlying_pixel_arrays))

    @property
    def pixel_shape(self):
        return tuple(np.asarray(self._wcs.pixel_shape) / self._factor)

    @property
    def pixel_bounds(self):
        if self._wcs.pixel_bounds is None:
            return self._wcs.pixel_bounds
        top_level_bounds = self._underlying_to_top_pixels(np.asarray(self._wcs.pixel_bounds))
        return [tuple(bounds) for bounds in top_level_bounds]
