import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from sunpy.visualization.imageanimator import ImageAnimatorWCS
import sunpy.visualization.wcsaxes_compat as wcsaxes_compat

__all__ = ['NDCubePlotMixin']

class NDCubeSequencePlotMixin:
    def plot(self, axes=None, plot_as_cube=False, image_axes=None, axis_ranges=None,
             unit_x_axis=None, unit_y_axis=None, data_unit=None, **kwargs):
        """
        Visualizes data in the NDCubeSequence.

        Based on the dimensionality of the sequence and value of image_axes kwarg,
        a Line/Image Animation/Plot is produced.

        Parameters
        ----------
        axes: `astropy.visualization.wcsaxes.core.WCSAxes` or ??? or None.
            The axes to plot onto. If None the current axes will be used.

        plot_as_cube: `bool`
            If the sequence has a common axis, visualize the sequence as a single
            cube where the sequence sub-cubes are sequential along the common axis.
            This will result in the sequence being treated as a cube with N-1 dimensions
            where N is the number of dimensions of the sequence, including the sequence
            dimension.
            Default=False

        image_axes: `int` or iterable of one or two `int`
            Default is images_axes=[-1, -2].  If sequence only has one dimension,
            default is images_axes=0.

        axis_ranges: `list` of physical coordinates for image axes and sliders or `None`
            If None coordinates derived from the WCS objects will be used for all axes.
            If a list, it should contain one element for each axis.  Each element should
            be either an `astropy.units.Quantity` or a `numpy.ndarray` of coordinates for
            each pixel, or a `str` denoting a valid extra coordinate.

        unit_x_axis: `astropy.unit.Unit` or valid unit `str`
            Unit in which X-axis should be displayed.  Only used if corresponding entry in
            axis_ranges is a Quantity or None.

        unit_y_axis: `astropy.unit.Unit` or valid unit `str`
            Unit in which Y-axis should be displayed.  Only used if corresponding entry in
            axis_ranges is a Quantity or None.

        unit: `astropy.unit.Unit` or valid unit `str`
            Unit in which data in a 2D image or animation should be displayed.  Only used if
            visualization is a 2D image or animation, i.e. if image_axis has length 2.
        
        """
        raise NotImplementedError()

    def _plot_1D_sequence(self, axes=None, x_axis_range=None, unit_x_axis=None, unit_y_axis=None):
        """
        Visualizes an NDCubeSequence of scalar NDCubes as a line plot.

        """
        raise NotImplementedError()

    def _plot_2D_sequence_as_1Dline(self, axes=None, x_axis_range=None, unit_x_axis=None, unit_y_axis=None,
                                    **kwargs)):
        """
        Visualizes an NDCubeSequence of 1D NDCubes with a common axis as a line plot.

        Called if plot_as_cube=True.

        """
        raise NotImplementedError()

    def _plot_2D_sequence(self, *args **kwargs):
        """
        Visualizes an NDCubeSequence of 1D NDCubes as a 2D image.

        """
        raise NotImplementedError()

    def _plot_3D_sequence_as_2Dimage(self, *args **kwargs):
        """
        Visualizes an NDCubeSequence of 2D NDCubes with a common axis as a 2D image.

        Called if plot_as_cube=True.

        """
        raise NotImplementedError()

    def _animate_ND_sequence(self, *args **kwargs):
        """
        Visualizes an NDCubeSequence of >2D NDCubes as 2D an animation with N-2 sliders.

        """
        raise NotImplementedError()

    def _animate_ND_sequence_as_Nminus1Danimation(self, *args **kwargs):
        """
        Visualizes a common axis NDCubeSequence of >3D NDCubes as 2D animation with N-3 sliders.

        Called if plot_as_cube=True.

        """
        raise NotImplementedError()
