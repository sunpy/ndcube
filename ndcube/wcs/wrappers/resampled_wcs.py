
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
        The shift of the lower edge of the 0th pixel (i.e. the pixel
        coordinate -0.5) of the resampled grid relative to the lower
        edge of the 0th pixel in the original underlying pixel grid,
        in units of original pixel widths.  (See the schematic in the
        Notes section for a graphical example.) If a scalar, the grid
        will be shifted by the same amount in all dimensions.

    Notes
    -----
    Below is a schematic of how ResampledLowLevelWCS works. The asterisks show the
    corners of pixels in a grid before resampling, while the dashes and
    pipes show the edges of the resampled pixels. The resampling along the
    x-axis has been performed using a factor of 2 and offset of 1, respectively,
    while the resampling of the y-axis uses a factor of 3 and offset of 2.
    The right column and upper row of numbers along the side and bottom of the
    grids denote the edges and centres of the original pixel grid in the original
    pixel coordinates.  The left column and lower row gives the same locations in
    the pixel coordinates of the resampled grid.  Note that the resampled pixels
    have an (x, y) shape of (2, 3) relative to the original pixel grid.
    Also note, the left/lower edge of the 0th pixel in the resampled grid (i.e. pixel
    coord -0.5) is shifted relative to the left/lower edge of the original 0th pixel,
    and that shift is given by the offset (+1 in the x-axis and +2 along the y-axis),
    which is in units of original pixel widths.

    ::

        resampled  original
        factor=3
        offset=2

          0.5      4.5 *-----------*-----------*-----------*-----------*
                                   |                       |
          2/6       4              |                       |
                                   |                       |
          1/6      3.5 *           *           *           *           *
                                   |                       |
           0        3              |                       |
                                   |                       |
         -1/3      2.5 *           *           *           *           *
                                   |                       |
         -2/6       2              |                       |
                                   |                       |
         -0.5      1.5 *-----------*-----------*-----------*-----------*
                                   |                       |
         -4/6       1              |                       |
                                   |                       |
         -5/6      0.5 *           *           *           *           *
                                   |                       |
          -1        0              |                       |
                                   |                       |
         -1-1/6   -0.5 *           *           *           *           *
                     -0.5    0    0.5    1    1.5    2    2.5    3    3.5  original pixel indices
                      -1   -0.75 -0.5  -0.25   0    0.25  0.5   0.75   1   resampled pixel indices: factor=2, offset=1
    """

    def __init__(self, wcs, factor, offset=0):
        self._wcs = wcs
        if np.isscalar(factor):
            factor = [factor] * self.pixel_n_dim
        self._factor = np.array(factor)
        if len(self._factor) != self.pixel_n_dim:
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
        return (top_pixels + 0.5) * factor - 0.5 + offset

    def _underlying_to_top_pixels(self, underlying_pixels):
        # Convert pixel indices of underlying pixel grid to user-facing grid.
        factor = self._pad_dims(self._factor, underlying_pixels.ndim)
        offset = self._pad_dims(self._offset, underlying_pixels.ndim)
        return (underlying_pixels + 0.5 - offset) / factor - 0.5

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
