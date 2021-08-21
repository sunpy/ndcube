import warnings

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.utils.exceptions import AstropyUserWarning
from astropy.visualization.wcsaxes import WCSAxes

from . import plotting_utils as utils
from .base import BasePlotter

__all__ = ['MatplotlibPlotter']


class MatplotlibPlotter(BasePlotter):
    """
    Provide visualization methods for NDCube which use `matplotlib`.
    """

    def plot(self, axes=None, plot_axes=None, axes_coordinates=None,
             axes_units=None, data_unit=None, wcs=None, **kwargs):
        """
        Visualize the `~ndcube.NDCube`.

        Parameters
        ----------
        axes: `~astropy.visualization.wcsaxes.WCSAxes` or None:, optional
            The axes to plot onto. If None the current axes will be used.

        plot_axes: `list`, optional
            A list of length equal to the number of pixel dimensions in array axis order.
            This list selects which cube axes are displayed on which plot axes.
            For an image plot this list should contain ``'x'`` and ``'y'`` for the
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

        wcs: `astropy.wcs.wcsapi.BaseHighLevelWCS`
            The WCS object to define the coordinates of the plot axes.

        kwargs :
            Additional keyword arguments are given to the underlying plotting infrastructure
            which depends on the dimensionality of the data and whether 1 or 2 plot_axes are
            defined:
            - Animations: `sunpy.visualization.animator.ArrayAnimatorWCS`
            - Static 2-D images: `matplotllib.pyplot.imshow`
            - Static 1-D line plots: `matplotllib.pyplot.plot`
        """
        naxis = self._ndcube.wcs.pixel_n_dim

        if not axes_coordinates:
            axes_coordinates = [...]
            plot_wcs = self._ndcube.wcs.low_level_wcs
        else:
            plot_wcs = self._ndcube.combined_wcs.low_level_wcs
        if wcs is not None:
            plot_wcs = wcs.low_level_wcs

        # Check kwargs are in consistent formats and set default values if not done so by user.
        plot_axes, axes_coordinates, axes_units = utils.prep_plot_kwargs(
            len(self._ndcube.dimensions), plot_wcs, plot_axes, axes_coordinates, axes_units)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyUserWarning)
            if naxis == 1:
                ax = self._plot_1D_cube(plot_wcs, axes, axes_coordinates,
                                        axes_units, data_unit, **kwargs)

            elif naxis == 2 and 'y' in plot_axes:
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
        yerror = self._ndcube.uncertainty.array if (self._ndcube.uncertainty is not None) else None
        ydata = self._ndcube.data

        if self._ndcube.unit is None:
            if data_unit is not None:
                raise TypeError("Can only set y-axis unit if self._ndcube.unit is set to a "
                                "compatible unit.")
        else:
            if data_unit is not None:
                ydata = u.Quantity(ydata, unit=self._ndcube.unit).to_value(data_unit)
                if yerror is not None:
                    yerror = u.Quantity(yerror, self._ndcube.unit).to_value(data_unit)
            else:
                data_unit = self._ndcube.unit

            default_ylabel += f" [{data_unit}]"

        # Combine data and uncertainty with mask.
        if self._ndcube.mask is not None:
            ydata = np.ma.masked_array(ydata, self._ndcube.mask)

            if yerror is not None:
                yerror = np.ma.masked_array(yerror, self._ndcube.mask)

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

        data = self._ndcube.data
        if data_unit is not None:
            # If user set data_unit, convert dat to desired unit if self._ndcube.unit set.
            if self._ndcube.unit is None:
                raise TypeError("Can only set data_unit if NDCube.unit is set.")
            data = u.Quantity(self._ndcube.data, unit=self._ndcube.unit).to_value(data_unit)

        if self._ndcube.mask is not None:
            data = np.ma.masked_array(data, self._ndcube.mask)

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

        try:
            from sunpy.visualization.animator import ArrayAnimatorWCS  # isort:skip
        except ImportError:
            raise ImportError("Sunpy is required for animated cube plots. "
                              "Either install sunpy or slice your cube down "
                              "to 2D before calling plot.")

        # Derive inputs for animation object and instantiate.
        data, wcs, plot_axes, coord_params = self._prep_animate_args(wcs, plot_axes,
                                                                     axes_units, data_unit)
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
        kwargs = {'wcs': self._ndcube.wcs}
        n_dim = len(self._ndcube.dimensions)
        if n_dim > 2:
            kwargs['slices'] = ['x', 'y'] + [None] * (ndim - 2)
        return WCSAxes, kwargs

    def _prep_animate_args(self, wcs, plot_axes, axes_units, data_unit):
        # If data_unit set, convert data to that unit
        if data_unit is None:
            data = self._ndcube.data
        else:
            data = u.Quantity(self._ndcube.data, unit=self._ndcube.unit, copy=False).to_value(data_unit)

        # Combine data values with mask.
        if self._ndcube.mask is not None:
            data = np.ma.masked_array(data, self._ndcube.mask)

        coord_params = {}
        if axes_units is not None:
            for axis_unit, coord_name in zip(axes_units, world_axis_physical_types):
                coord_params[coord_name] = {'format_unit': axis_unit}

        # TODO: Add support for transposing the array.
        if 'y' in plot_axes and plot_axes.index('y') < plot_axes.index('x'):
            warnings.warn(
                "Animating a NDCube does not support transposing the array. The world axes "
                "may not display as expected because the array will not be transposed.",
                UserWarning
            )
        plot_axes = [p if p is not None else 0 for p in plot_axes]

        return data, wcs, plot_axes, coord_params
