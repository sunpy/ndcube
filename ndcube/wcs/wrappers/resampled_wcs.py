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

    factor : `int` or `float` or iterable of the same
        The factor by which to increase the pixel size for each pixel
        axis. If a scalar, the same factor is used for all axes.

    offset: `int` or `float` or iterable of the same
        The location on the underlying pixel grid which corresponds
        to zero on the top level pixel grid. If a scalar, the grid will be
        shifted by the same amount in all dimensions.
    """
    def __init__(self, wcs, factor, offset=0):
        self._wcs = wcs
        if np.isscalar(factor):
            factor = [factor] * self.pixel_n_dim
        self._factor = np.array(factor)
        if len(self._factor) !=  self.pixel_n_dim:
            raise ValueError(f"Length of factor must equal number of dimensions {self.pixel_n_dim}.")
        if np.isscalar(offset):
            offset = [offset] * self.pixel_n_dim
        self._offset = np.array(offset)
        if len(self._offset) != self.pixel_n_dim:
            raise ValueError(f"Length of offset must equal number of dimensions {self.pixel_n_dim}.")

    def _top_to_underlying_pixels(self, top_pixels):
        # Convert user-facing pixel indices to the pixel grid of underlying WCS.
        factor = self._pad_dims(self._factor, top_pixels.ndim)
        offset = self._pad_dims(self._offset, top_pixels.ndim)
        return top_pixels * factor + offset

    def _underlying_to_top_pixels(self, underlying_pixels):
        # Convert pixel indices of underlying pixel grid to user-facing grid.
        factor = self._pad_dims(self._factor, underlying_pixels.ndim)
        offset = self._pad_dims(self._offset, underlying_pixels.ndim)
        return (underlying_pixels - offset) / factor

    def _pad_dims(self, arr, ndim):
        # Pad array with trailing degenerate dimensions.
        # This make scaling with pixel arrays easier.
        shape = np.ones(ndim, dtype=int)
        shape[0] = len(arr)
        return arr.reshape(tuple(shape))

    def pixel_to_world_values(self, *pixel_arrays):
        underlying_pixel_arrays = self._top_to_underlying_pixels(np.asarray(pixel_arrays))
        return self._wcs.pixel_to_world_values(*underlying_pixel_arrays)

    def world_to_pixel_values(self, *world_arrays):
        underlying_pixel_arrays = self._wcs.world_to_pixel_values(*world_arrays)
        top_pixel_arrays = self._underlying_to_top_pixels(np.asarray(underlying_pixel_arrays))
        return tuple(array for array in top_pixel_arrays)

    @property
    def pixel_shape(self):
        # Return pixel shape of resampled grid.
        # Where shape is an integer, return an int type as its required for some uses.
        if self._wcs.pixel_shape is None:
            return self._wcs.pixel_shape
        underlying_shape = np.asarray(self._wcs.pixel_shape)
        int_elements = np.isclose(np.mod(underlying_shape, self._factor), 0,
                                  atol=np.finfo(float).resolution)
        pixel_shape = underlying_shape / self._factor
        return tuple(int(np.rint(i)) if is_int else i
                     for i, is_int in zip(pixel_shape, int_elements))

    @property
    def pixel_bounds(self):
        if self._wcs.pixel_bounds is None:
            return self._wcs.pixel_bounds
        top_level_bounds = self._underlying_to_top_pixels(np.asarray(self._wcs.pixel_bounds))
        return [tuple(bounds) for bounds in top_level_bounds]
