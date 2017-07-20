from sunpy.visualization.imageanimator import ImageAnimatorWCS
import astropy.nddata

__all__ = ['NDCube', 'Cube2D', 'Cube1D']


class NDCube(astropy.nddata.NDData):
    """docstring for NDCube"""

    def __init__(self, data=None, wcs=None, **kwargs):
        super(NDCube, self).__init__(data=data, **kwargs)
        self.axes_wcs = wcs

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
        Plots an interactive visualization of this cube with a slider
        controlling the wavelength axis.
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
        i = ImageAnimatorWCS(self.data, wcs=self.axes_wcs, *args, **kwargs)
        return i

    def __getitem__(self, item):
        pass


class Cube2D(NDCube):
    """docstring for Cube2D"""

    def __init__(self, data=None, wcs=None, **kwargs):
        super(Cube2D, self).__init__(data=data, wcs=wcs, **kwargs)

    def plot(self):
        pass

    def __getitem__(self, item):
        pass


class Cube1D(NDCube):
    """docstring for Cube1D"""

    def __init__(self, data=None, wcs=None, **kwargs):
        super(Cube1D, self).__init__(data=data, wcs=wcs, **kwargs)

    def plot(self):
        pass

    def __getitem__(self, item):
        pass
