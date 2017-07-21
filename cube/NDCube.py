from sunpy.visualization.imageanimator import ImageAnimatorWCS
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
        wcs = self.wcs.wcs_slicer[item]
        mask = None
        if self.mask is not None:
            mask = self.mask[item]
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

    def plot(self):
        pass


class Cube1D(NDCube):
    """docstring for Cube1D"""

    def __init__(self, data, uncertainty=None, mask=None, wcs=None, meta=None, unit=None, copy=False, **kwargs):
        super(Cube1D, self).__init__(data, uncertainty=uncertainty, mask=mask,
                                     wcs=wcs, meta=meta, unit=unit, copy=copy, **kwargs)

    def plot(self):
        pass
