import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from sunpy.visualization.imageanimator import ImageAnimatorWCS
import sunpy.visualization.wcsaxes_compat as wcsaxes_compat

__all__ = ['_plot_3D_cube', '_plot_2D_cube', '_plot_1D_cube']


def _plot_3D_cube(cube, image_axes=None, unit_x_axis=None, unit_y_axis=None,
                  axis_ranges=None, **kwargs):
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

    axis_ranges: `list` of physical coordinates for array or None
        If None array indices will be used for all axes.
        If a list it should contain one element for each axis of the numpy array.
        For the image axes a [min, max] pair should be specified which will be
        passed to :func:`matplotlib.pyplot.imshow` as extent.
        For the slider axes a [min, max] pair can be specified or an array the
        same length as the axis which will provide all values for that slider.
        If None is specified for an axis then the array indices will be used
        for that axis.
    """
    if not image_axes:
        image_axes = [-1, -2]
    i = ImageAnimatorWCS(cube.data, wcs=cube.wcs, image_axes=image_axes,
                         unit_x_axis=unit_x_axis, unit_y_axis=unit_y_axis,
                         axis_ranges=axis_ranges, **kwargs)
    return i


def _plot_2D_cube(cube, axes=None, image_axes=None, **kwargs):
    """
    Plots a 2D image onto the current
    axes. Keyword arguments are passed on to matplotlib.

    Parameters
    ----------
    axes: `astropy.visualization.wcsaxes.core.WCSAxes` or `None`:
        The axes to plot onto. If None the current axes will be used.

    image_axes: `list`.
        The first axis in WCS object will become the first axis of image_axes and
        second axis in WCS object will become the second axis of image_axes.
        Default: ['x', 'y']
    """
    if not image_axes:
        image_axes = ['x', 'y']
    if axes is None:
        if cube.wcs.naxis is not 2:
            missing_axis = cube.missing_axis
            slice_list = []
            axis_index = []
            index = 0
            for i, bool_ in enumerate(missing_axis):
                if not bool_:
                    slice_list.append(image_axes[index])
                    index += 1
                else:
                    slice_list.append(1)
            if index is not 2:
                raise ValueError("Dimensions of WCS and data don't match")
        axes = wcsaxes_compat.gca_wcs(cube.wcs, slices=slice_list)
    plot = axes.imshow(cube.data, **kwargs)
    return plot


def _plot_1D_cube(cube, unit=None, origin=0):
    """
    Plots a graph.
    Keyword arguments are passed on to matplotlib.

    Parameters
    ----------
    unit: `astropy.unit.Unit`
        The data is changed to the unit given or the cube.unit if not given.
    """
    index_not_one = []
    for i, _bool in enumerate(cube.missing_axis):
        if not _bool:
            index_not_one.append(i)
    if unit is None:
        unit = cube.wcs.wcs.cunit[index_not_one[0]]
    plot = plt.plot(cube.pixel_to_world(
        [u.Quantity(np.arange(cube.data.shape[0]), unit=u.pix)], origin=origin)[0].to(unit),
                    cube.data)
    return plot
