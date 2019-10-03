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
            ax = self._plot_1D_cube(self.wcs, axes, axes_coordinates,
                                    axes_units, data_unit, **kwargs)
        elif len(plot_axes) == 1:
            raise NotImplementedError()
            ax = self._animate_cube_1D(
                plot_axes=plot_axes, axes_coordinates=axes_coordinates,
                axes_units=axes_units, data_unit=data_unit, **kwargs)

        elif naxis == 2:
            ax = self._plot_2D_cube(self.wcs, axes, plot_axes, axes_coordinates,
                                    axes_units, data_unit, **kwargs)
        else:
            ax = self._animate_cube_2D(self.wcs,
                plot_axes=plot_axes, axes_coordinates=axes_coordinates,
                axes_units=axes_units, **kwargs)

        return ax

    def _plot_1D_cube(self, wcs, axes=None, axes_coordinates=None, axes_units=None,
                      data_unit=None, **kwargs):
        """
        Plots a graph. Keyword arguments are passed on to matplotlib.

        Parameters
        ----------
        data_unit: `astropy.unit.Unit`
            The data is changed to the unit given or the cube.unit if not given.
        """

        if axes is None:
            axes = wcsaxes_compat.gca_wcs(wcs)

        if axes_coordinates is not None and axes_coordinates[0] != wcs.world_axis_physical_types[::-1][0]:
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

        utils.set_wcsaxes_labels_units(axes.coords, wcs, axes_units)

        return axes

    def _plot_2D_cube(self, wcs, axes=None, plot_axes=None, axes_coordinates=None,
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
            slices = plot_axes[::-1] if plot_axes is not None else None
            axes = wcsaxes_compat.gca_wcs(wcs, slices=slices)

        utils.set_wcsaxes_labels_units(axes.coords, wcs, axes_units)

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

    def _animate_cube_2D(self, wcs, plot_axes=None, axes_coordinates=None,
                         axes_units=None, data_unit=None, **kwargs):
        if axes_coordinates is not None and axes_coordinates[0] != wcs.world_axis_physical_types[::-1][0]:
            raise NotImplementedError("We need to support extra_coords here")

        image_axes = [plot_axes[::-1].index("x"), plot_axes[::-1].index("y")]

        # If data_unit set, convert data to that unit
        if data_unit is None:
            data = self.data
        else:
            data = u.Quantity(self.data, unit=self.unit).to_value(data_unit)

        # Combine data values with mask.
        if self.mask is not None:
            data = np.ma.masked_array(data, self.mask)

        # TODO: Deal with axis_ranges for slider axes, should take functions
        # TODO: Add labels to plot, probably needs callback or something

        # Generate plot
        x_unit = y_unit = None
        if axes_units is not None:
            x_unit = axes_units[image_axes[0]]
            y_unit = axes_units[image_axes[1]]

        ax = ImageAnimatorWCS(data, wcs=wcs, image_axes=image_axes,
                              unit_x_axis=x_unit,
                              unit_y_axis=y_unit,
                              **kwargs)

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
