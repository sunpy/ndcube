from sunpy.visualization.imageanimator import ImageAnimatorWCS
import matplotlib.pyplot as plt
import sunpy.visualization.wcsaxes_compat as wcsaxes_compat
import astropy.units as u
import sunpycube.wcs_util
import astropy.nddata
import numpy as np

__all__ = ['NDCube', 'Cube2D', 'Cube1D']


class NDCube(astropy.nddata.NDData):
    """docstring for NDCube"""

    def __init__(self, data, uncertainty=None, mask=None, wcs=None, meta=None, unit=None, copy=False, **kwargs):
        if data.ndim is not wcs.naxis:
            _naxis = wcs._naxis
            count = 0
            for i in _naxis:
                if not i is 1:
                    count += 1
            if count is not data.ndim:
                raise ValueError(
                    "The number of dimensions of data and number of axes of wcs donot match.")
        super(NDCube, self).__init__(data, uncertainty=uncertainty, mask=mask,
                                     wcs=wcs, meta=meta, unit=unit, copy=copy, **kwargs)

    def pixel_to_world(self):
        pass

    def world_to_pixel(self):
        pass

    def to_sunpy(self):
        pass

    def dimension(self):
        pass

    def plot(self, *args, **kwargs):
        """
        Plots an interactive visualization of this cube using sliders to move through axes
        plot using in the image.
        Parameters other than data and wcs are passed to ImageAnimatorWCS, which in turn
        passes them to imshow.

        Parameters
        ----------
        image_axes: `list`
            The two axes that make the image.
            Like [-1,-2] this implies cube instance -1 dimension
            will be x-axis and -2 dimension will be y-axis.

        unit_x_axis: `astropy.units.Unit`
            The unit of x axis.

        unit_y_axis: `astropy.units.Unit`
            The unit of y axis.

        axis_ranges: list of physical coordinates for array or None
            If None array indices will be used for all axes.
            If a list it should contain one element for each axis of the numpy array.
            For the image axes a [min, max] pair should be specified which will be
            passed to :func:`matplotlib.pyplot.imshow` as extent.
            For the slider axes a [min, max] pair can be specified or an array the
            same length as the axis which will provide all values for that slider.
            If None is specified for an axis then the array indices will be used
            for that axis.
        """
        i = ImageAnimatorWCS(self.data, wcs=self.wcs, *args, **kwargs)
        return i

    def __getitem__(self, item):
        if item is None or (isinstance(item, tuple) and None in item):
            raise IndexError("None indices not supported")
        data = self.data[item]
        wcs = sunpycube.wcs_util._wcs_slicer(self.wcs, item)
        if self.mask is not None:
            mask = self.mask[item]
        else:
            mask = None
        if data.ndim is 2:
            result = Cube2D(data, wcs=wcs, mask=mask, uncertainty=self.uncertainty,
                            meta=self.meta, unit=self.unit, copy=False)
        elif data.ndim is 1:
            result = Cube1D(data, wcs=wcs, mask=mask, uncertainty=self.uncertainty,
                            meta=self.meta, unit=self.unit, copy=False)
        else:
            result = NDCube(data, wcs=wcs, mask=mask, uncertainty=self.uncertainty,
                            meta=self.meta, unit=self.unit, copy=False)
        return result


class Cube2D(NDCube):
    """docstring for Cube2D"""

    def __init__(self, data, uncertainty=None, mask=None, wcs=None, meta=None, unit=None, copy=False, **kwargs):
        super(Cube2D, self).__init__(data, uncertainty=uncertainty, mask=mask,
                                     wcs=wcs, meta=meta, unit=unit, copy=copy, **kwargs)

    def plot(self, axes=None, axis_data=['x', 'y'], **kwargs):
        """
        Plots an x-y graph at a certain specified wavelength onto the current
        axes. Keyword arguments are passed on to matplotlib.

        Parameters
        ----------
        axes: `astropy.visualization.wcsaxes.core.WCSAxes` or `None`:
            The axes to plot onto. If None the current axes will be used.

        axis_data: `list`.
            The first axis in WCS object will become the first axis of axis_data and
            second axis in WCS object will become the seconf axis of axis_data.
        """
        if axes is None:
            if self.wcs.naxis is not 2:
                slice_list = self.wcs._naxis
                axis_index = []
                for i, ax in enumerate(slice_list):
                    if ax is not 1:
                        axis_index.append(i)
                if len(axis_index) is 2:
                    slice_list[axis_index[0]] = axis_data[0]
                    slice_list[axis_index[1]] = axis_data[1]
                else:
                    raise ValueError("Dimensions of WCS and data don't match")
            axes = wcsaxes_compat.gca_wcs(self.wcs, slices=slice_list)
        plot = axes.imshow(self.data, **kwargs)
        return plot


class Cube1D(NDCube):
    """docstring for Cube1D"""

    def __init__(self, data, uncertainty=None, mask=None, wcs=None, meta=None, unit=None, copy=False, **kwargs):
        super(Cube1D, self).__init__(data, uncertainty=uncertainty, mask=mask,
                                     wcs=wcs, meta=meta, unit=unit, copy=copy, **kwargs)

    def plot(self, unit=None):
        """
        Plots a graph.
        Keyword arguments are passed on to matplotlib.

        Parameters
        ----------
        unit: `astropy.unit.Unit`
        The data is changed to the unit given or the self.unit if not given.
        """
        if self.unit is not None and unit is None:
            if isinstance(self.unit, u.Unit):
                plot = plt.plot(self.data * self.unit)
            else:
                plot = plt.plot(self.data)
        elif unit is not None:
            if isinstance(unit, u.Unit):
                plot = plt.plot(self.data * unit)
        return plot
