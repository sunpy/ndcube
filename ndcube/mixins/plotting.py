import datetime
from warnings import warn

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import WCSAxes
import sunpy.visualization.wcsaxes_compat as wcsaxes_compat
try:
    from sunpy.visualization.animator import ImageAnimator, ImageAnimatorWCS, LineAnimator
except ImportError:
    from sunpy.visualization.imageanimator import ImageAnimator, ImageAnimatorWCS, LineAnimator

from ndcube import utils
from ndcube.utils.cube import _get_extra_coord_edges
import ndcube.mixins.plotting_utils as utils

__all__ = ['NDCubePlotMixin']


class NDCubePlotMixin:
    """
    Add plotting functionality to a NDCube class.
    """

    def plot(self, axes=None, plot_axes=None, axes_coordinates=None,
             axes_units=None, data_unit=None, **kwargs):
        """
        Visualize the `~ndcube.NDCube`.

        Parameters
        ----------
        axes: `~astropy.visualization.wcsaxes.WCSAxes` or None:, optional
            The axes to plot onto. If None the current axes will be used.

        plot_axes: `list`, optional
            A list of length equal to the number of pixel dimensions. This list
            selects which cube axes are displayed on which plot axes. For a
            image plot this list should contain ``'x'`` and ``'y'`` for the
            plot axes and `None` for all the other elements. For a line plot it
            should only contain ``'x'`` and `None` for all the other elements.

        axes_unit: `list`, optional
            A list of length equal to the number of pixel dimensions specifying
            the units of each axis, or `None` to use the default unit for that
            axis.

        axes_coordinates: `list`, optional
            A list of length equal to the number of pixel dimensions. For each
            axis the value of the list should either be a string giving the
            world axis type or `None` to use the default axis from the WCS.

        data_unit: `astropy.unit.Unit`
            The data is changed to the unit given or the ``NDCube.unit`` if not
            given, used for 1D plots.
        """
        naxis = self.wcs.pixel_n_dim

        # Check kwargs are in consistent formats and set default values if not done so by user.
        plot_axes, axes_coordinates, axes_units = utils.prep_plot_kwargs(
            self, plot_axes, axes_coordinates, axes_units)

        if naxis == 1:
            ax = self._plot_1D_cube(axes, axes_coordinates,
                                    axes_units, data_unit, **kwargs)
        elif len(plot_axes) == 1:
            raise NotImplementedError()
            ax = self._animate_cube_1D(
                plot_axes=plot_axes, axes_coordinates=axes_coordinates,
                axes_units=axes_units, data_unit=data_unit, **kwargs)

        elif naxis == 2:
            ax = self._plot_2D_cube(axes, plot_axes, axes_coordinates,
                                    axes_units, data_unit, **kwargs)
        else:
            raise NotImplementedError()
            ax = self._animate_cube_2D(
                plot_axes=plot_axes, axes_coordinates=axes_coordinates,
                axes_units=axes_units, **kwargs)

        return ax

    def _plot_1D_cube(self, axes=None, axes_coordinates=None, axes_units=None,
                      data_unit=None, **kwargs):
        """
        Plots a graph. Keyword arguments are passed on to matplotlib.

        Parameters
        ----------
        data_unit: `astropy.unit.Unit`
            The data is changed to the unit given or the cube.unit if not given.
        """

        if axes is None:
            axes = wcsaxes_compat.gca_wcs(self.wcs)

        if axes_coordinates is not None and axes_coordinates[0] != self.world_axis_physical_types[0]:
            raise NotImplementedError("We need to support extra_coords here")

        default_ylabel = f"Data"

        # Derive y-axis coordinates, uncertainty and unit from the NDCube's data.
        yerror = self.uncertainty.array if (self.uncertainty is not None) else None
        ydata = self.data

        if self.unit is None:
            if data_unit is not None:
                raise TypeError("Can only set y-axis unit if self.unit is set to a "
                                "compatible unit.")
        else:
            if data_unit is not None:
                ydata = u.Quantity(ydata, unit=self.unit).to_value(data_unit)
                if yerror is not None:
                    yerror = u.Quantity(yerror, self.unit).to_value(data_unit)
            else:
                data_unit = self.unit

            default_ylabel += f" [{data_unit}]"

        # Combine data and uncertainty with mask.
        if self.mask is not None:
            ydata = np.ma.masked_array(ydata, self.mask)

            if yerror is not None:
                yerror = np.ma.masked_array(yerror, self.mask)

        if yerror is not None:
            # We plot against pixel coordinates
            axes.errorbar(np.arange(len(ydata)), ydata, yerr=yerror, **kwargs)
        else:
            axes.plot(ydata, **kwargs)

        axes.set_ylabel(default_ylabel)

        utils.set_wcsaxes_labels_units(axes.coords, self.wcs, axes_units)

        return axes

    def _plot_2D_cube(self, axes=None, plot_axes=None, axes_coordinates=None,
                      axes_units=None, data_unit=None, **kwargs):
        """
        Plots a 2D image onto the current axes. Keyword arguments are passed on
        to matplotlib.

        Parameters
        ----------
        axes: `astropy.visualization.wcsaxes.core.WCSAxes` or `None`:
            The axes to plot onto. If None the current axes will be used.

        plot_axes: `list`.
            The first axis in WCS object will become the first axis of plot_axes and
            second axis in WCS object will become the second axis of plot_axes.
            Default: ['x', 'y']
        """
        if axes is None:
            slices = list(filter(lambda x: x is not None, plot_axes))
            axes = wcsaxes_compat.gca_wcs(self.wcs, slices=slices)

        utils.set_wcsaxes_labels_units(axes.coords, self.wcs, axes_units)

        data = self.data
        if data_unit is not None:
            # If user set data_unit, convert dat to desired unit if self.unit set.
            if self.unit is None:
                raise TypeError("Can only set data_unit if NDCube.unit is set.")
            data = u.Quantity(self.data, unit=self.unit).to_value(data_unit)

        if self.mask is not None:
            data = np.ma.masked_array(data, self.mask)

        # Plot data
        axes.imshow(data, **kwargs)

        return axes

    def _animate_cube_2D(self, plot_axes=None, axes_coordinates=None,
                         axes_units=None, data_unit=None, **kwargs):
        """
        Plots an interactive visualization of this cube using sliders to move
        through axes plot using in the image. Parameters other than data and
        wcs are passed to ImageAnimatorWCS, which in turn passes them to
        imshow.

        Parameters
        ----------
        plot_axes: `list`
            The two axes that make the image.
            Like [-1,-2] this implies cube instance -1 dimension
            will be x-axis and -2 dimension will be y-axis.

        axes_unit: `list` of `astropy.units.Unit`

        axes_coordinates: `list` of physical coordinates for array or None
            If None array indices will be used for all axes.
            If a list it should contain one element for each axis of the numpy array.
            For the image axes a [min, max] pair should be specified which will be
            passed to :func:`matplotlib.pyplot.imshow` as extent.
            For the slider axes a [min, max] pair can be specified or an array the
            same length as the axis which will provide all values for that slider.
            If None is specified for an axis then the array indices will be used
            for that axis.
            The physical coordinates expected by axes_coordinates should be an array of
            pixel_edges.
            A str entry in axes_coordinates signifies that an extra_coord will be used
            for the axis's coordinates.
            The str must be a valid name of an extra_coord that corresponds to the same axis
            to which it is applied in the plot.
        """
        # For convenience in inserting dummy variables later, ensure
        # plot_axes are all positive.
        plot_axes = [i if i >= 0 else self.data.ndim + i for i in plot_axes]
        # If axes kwargs not set by user, set them as list of Nones for
        # each axis for consistent behaviour.
        if axes_coordinates is None:
            axes_coordinates = [None] * self.data.ndim
        if axes_units is None:
            axes_units = [None] * self.data.ndim
        # If data_unit set, convert data to that unit
        if data_unit is None:
            data = self.data
        else:
            data = (self.data * self.unit).to(data_unit).value
        # Combine data values with mask.
        if self.mask is not None:
            data = np.ma.masked_array(data, self.mask)
        # If axes_coordinates not provided generate an ImageAnimatorWCS plot
        # using NDCube's wcs object.
        new_axes_coordinates, new_axes_units, default_labels = \
            self._derive_axes_coordinates(axes_coordinates, axes_units, data.shape, edges=True)

        if (axes_coordinates[plot_axes[0]] is None and
                axes_coordinates[plot_axes[1]] is None):
            # Generate plot
            ax = ImageAnimatorWCS(data, wcs=self.wcs, image_axes=plot_axes,
                                  unit_x_axis=new_axes_units[plot_axes[0]],
                                  unit_y_axis=new_axes_units[plot_axes[1]],
                                  axis_ranges=new_axes_coordinates, **kwargs)

            # Set the labels of the plot
            ax.axes.coords[0].set_axislabel(
                self.wcs.world_axis_physical_types[plot_axes[0]])
            ax.axes.coords[1].set_axislabel(
                self.wcs.world_axis_physical_types[plot_axes[1]])

        # If one of the plot axes is set manually, produce a basic ImageAnimator object.
        else:
            # If axis labels not set by user add to kwargs.
            ax = ImageAnimator(data, image_axes=plot_axes,
                               axis_ranges=new_axes_coordinates, **kwargs)

            # Add the labels of the plot
            ax.axes.set_xlabel(default_labels[plot_axes[0]])
            ax.axes.set_ylabel(default_labels[plot_axes[1]])
        return ax

    def _animate_cube_1D(self, plot_axis_index=-1, axes_coordinates=None,
                         axes_units=None, data_unit=None, **kwargs):
        """
        Animates an axis of a cube as a line plot with sliders for other axes.
        """
        if axes_coordinates is None:
            axes_coordinates = [None] * self.data.ndim
        if axes_units is None:
            axes_units = [None] * self.data.ndim
        # Get real world axis values along axis to be plotted and enter into axes_ranges kwarg.
        if axes_coordinates[plot_axis_index] is None:
            xname = self.world_axis_physical_types[plot_axis_index]
            xdata = self.axis_world_coords(plot_axis_index, edges=True)
        elif isinstance(axes_coordinates[plot_axis_index], str):
            xname = axes_coordinates[plot_axis_index]
            xdata = _get_extra_coord_edges(self.extra_coords[xname]["value"])
        else:
            xname = ""
            xdata = axes_coordinates[plot_axis_index]
        # Change x data to desired units it set by user.
        if isinstance(xdata, u.Quantity):
            if axes_units[plot_axis_index] is None:
                unit_x_axis = xdata.unit
            else:
                unit_x_axis = axes_units[plot_axis_index]
                xdata = xdata.to(unit_x_axis).value
        else:
            if axes_units[plot_axis_index] is not None:
                raise TypeError(INVALID_UNIT_SET_MESSAGE)
            else:
                unit_x_axis = None
        # Put xdata back into axes_coordinates as a masked array.
        if len(xdata.shape) > 1:
            # Since LineAnimator currently only accepts 1-D arrays for the x-axis, collapse xdata
            # to single dimension by taking mean along non-plotting axes.
            index = utils.wcs.get_dependent_data_axes(self.wcs, plot_axis_index)
            reduce_axis = np.where(index == np.array(plot_axis_index))[0]

            index = np.delete(index, reduce_axis)
            # Reduce the data by taking mean
            xdata = np.mean(xdata, axis=tuple(index))
        axes_coordinates[plot_axis_index] = xdata
        # Set default x label
        default_xlabel = f"{xname} [{unit_x_axis}]"
        # Derive y axis data
        if data_unit is None:
            data = self.data
            data_unit = self.unit
        else:
            if self.unit is None:
                raise TypeError("NDCube.unit is None.  Must be an astropy.units.unit or "
                                "valid unit string in order to set data_unit.")
            else:
                data = (self.data * self.unit).to(data_unit).value
        # Combine data with mask
        if self.mask is not None:
            data = np.ma.masked_array(data, self.mask)
        # Set default y label
        default_ylabel = f"Data [{unit_x_axis}]"
        # Initiate line animator object.
        ax = LineAnimator(data, plot_axis_index=plot_axis_index, axis_ranges=axes_coordinates,
                          xlabel=default_xlabel,
                          ylabel=f"Data [{data_unit}]", **kwargs)
        return ax
