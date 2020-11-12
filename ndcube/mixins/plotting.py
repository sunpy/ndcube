import warnings

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.utils.exceptions import AstropyUserWarning
from astropy.visualization.wcsaxes import WCSAxes
from sunpy.visualization.animator import ArrayAnimatorWCS

from . import plotting_utils as utils

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
            A list of length equal to the number of world dimensions specifying
            the units of each axis, or `None` to use the default unit for that
            axis.

        axes_coordinates: `list`, optional
            A list of length equal to the number of pixel dimensions. For each
            axis the value of the list should either be a string giving the
            world axis type or `None` to use the default axis from the WCS.

        data_unit: `astropy.unit.Unit`
            The data is changed to the unit given or the ``NDCube.unit`` if not
            given.
        """
        naxis = self.wcs.pixel_n_dim

        if not axes_coordinates:
            axes_coordinates = [...]
            plot_wcs = self.wcs.low_level_wcs
        else:
            plot_wcs = self.combined_wcs.low_level_wcs

        # Check kwargs are in consistent formats and set default values if not done so by user.
        plot_axes, axes_coordinates, axes_units = utils.prep_plot_kwargs(
            len(self.dimensions), plot_wcs, plot_axes, axes_coordinates, axes_units)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyUserWarning)
            if naxis == 1:
                ax = self._plot_1D_cube(plot_wcs, axes, axes_coordinates,
                                        axes_units, data_unit, **kwargs)

            elif naxis == 2:
                ax = self._plot_2D_cube(plot_wcs, axes, plot_axes, axes_coordinates,
                                        axes_units, data_unit, **kwargs)
            else:
                ax = self._animate_cube(plot_wcs, plot_axes=plot_axes,
                                        axes_coordinates=axes_coordinates,
                                        axes_units=axes_units, **kwargs)

        return ax

    def _not_visible_coords(self, axes, axes_coordinates):
        """
        Based on an axes object and axes_coords, work out which coords should not be visible.
        """
        visible_coords = set(item[1] for item in axes.coords._aliases.items() if item[0] in axes_coordinates)
        return set(axes.coords._aliases.values()).difference(visible_coords)

    def _apply_axes_coordinates(self, axes, axes_coordinates):
        """
        Hide ticks and labels for non-visible axes based on axes_coordinates.
        """
        for coord_index in self._not_visible_coords(axes, axes_coordinates):
            axes.coords[coord_index].set_ticks_visible(False)
            axes.coords[coord_index].set_ticklabel_visible(False)

    def _plot_1D_cube(self, wcs, axes=None, axes_coordinates=None, axes_units=None,
                      data_unit=None, **kwargs):
        if axes is None:
            axes = plt.subplot(projection=wcs)

        self._apply_axes_coordinates(axes, axes_coordinates)

        default_ylabel = "Data"

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

        utils.set_wcsaxes_format_units(axes.coords, wcs, axes_units)

        return axes

    def _plot_2D_cube(self, wcs, axes=None, plot_axes=None, axes_coordinates=None,
                      axes_units=None, data_unit=None, **kwargs):
        if axes is None:
            axes = plt.subplot(projection=wcs, slices=plot_axes)

        utils.set_wcsaxes_format_units(axes.coords, wcs, axes_units)

        self._apply_axes_coordinates(axes, axes_coordinates)

        data = self.data
        if data_unit is not None:
            # If user set data_unit, convert dat to desired unit if self.unit set.
            if self.unit is None:
                raise TypeError("Can only set data_unit if NDCube.unit is set.")
            data = u.Quantity(self.data, unit=self.unit).to_value(data_unit)

        if self.mask is not None:
            data = np.ma.masked_array(data, self.mask)

        if plot_axes.index('x') > plot_axes.index('y'):
            data = data.T

        # Plot data
        im = axes.imshow(data, **kwargs)

        # Set current axes/image if pyplot is being used (makes colorbar work)
        for i in plt.get_fignums():
            if axes in plt.figure(i).axes:
                plt.sca(axes)
                plt.sci(im)

        return axes

    def _animate_cube(self, wcs, plot_axes=None, axes_coordinates=None,
                      axes_units=None, data_unit=None, **kwargs):

        # If data_unit set, convert data to that unit
        if data_unit is None:
            data = self.data
        else:
            data = u.Quantity(self.data, unit=self.unit).to_value(data_unit)

        # Combine data values with mask.
        if self.mask is not None:
            data = np.ma.masked_array(data, self.mask)

        coord_params = {}
        if axes_units is not None:
            for axis_unit, coord_name in zip(axes_units, wcs.world_axis_physical_types):
                coord_params[coord_name] = {'format_unit': axis_unit}

        # TODO: Add support for transposing the array.
        if 'y' in plot_axes and plot_axes.index('y') < plot_axes.index('x'):
            warnings.warn(
                "Animating a NDCube does not support transposing the array. The world axes "
                "may not display as expected because the array will not be transposed.",
                UserWarning
            )
        plot_axes = [p if p is not None else 0 for p in plot_axes]
        ax = ArrayAnimatorWCS(data, wcs, plot_axes, coord_params=coord_params, **kwargs)

        # We need to modify the visible axes after the axes object has been created.
        # This call affects only the initial draw
        self._apply_axes_coordinates(ax.axes, axes_coordinates)

        # This changes the parameters for future iterations
        for hidden in self._not_visible_coords(ax.axes, axes_coordinates):
            if hidden in ax.coord_params:
                param = ax.coord_params[hidden]
            else:
                param = {}

            param['ticks'] = False
            ax.coord_params[hidden] = param

        return ax

    def _as_mpl_axes(self):
        """
        Compatibility hook for Matplotlib and WCSAxes.
        This functionality requires the WCSAxes package to work. The reason
        we include this here is that it allows users to use WCSAxes without
        having to explicitly import WCSAxes
        With this method, one can do::
            fig = plt.figure()  # doctest: +SKIP
            ax = plt.subplot(projection=my_ndcube)  # doctest: +SKIP
        and this will generate a plot with the correct WCS coordinates on the
        axes. See https://wcsaxes.readthedocs.io for more information.
        """
        kwargs = {'wcs': self.wcs}
        n_dim = len(self.dimensions)
        if n_dim > 2:
            kwargs['slices'] = ['x', 'y'] + [None] * (ndim - 2)
        return WCSAxes, kwargs
