from warnings import warn

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from sunpy.visualization.imageanimator import ImageAnimatorWCS
import sunpy.visualization.wcsaxes_compat as wcsaxes_compat

from ndcube.mixins.sequence_plotting import _prep_axes_kwargs, _derive_1D_coordinates_and_units, _determine_sequence_units, _make_1D_sequence_plot

__all__ = ['NDCubePlotMixin']


class NDCubePlotMixin:
    """
    Add plotting functionality to a NDCube class.
    """

    def plot(self, axes=None, plot_axis_indices=[-1, -2], axes_coordinates=None,
             axes_units=None, data_unit=None, **kwargs):
        """
        Plots an interactive visualization of this cube with a slider
        controlling the wavelength axis for data having dimensions greater than 2.
        Plots an x-y graph onto the current axes for 2D or 1D data. Keyword arguments are passed
        on to matplotlib.
        Parameters other than data and wcs are passed to ImageAnimatorWCS, which in turn
        passes them to imshow for data greater than 2D.

        Parameters
        ----------
        plot_axis_indices: `list`
            The two axes that make the image.
            Like [-1,-2] this implies cube instance -1 dimension
            will be x-axis and -2 dimension will be y-axis.

        axes: `astropy.visualization.wcsaxes.core.WCSAxes` or None:
            The axes to plot onto. If None the current axes will be used.

        axes_unit: `list` of `astropy.units.Unit`

        data_unit: `astropy.unit.Unit`
            The data is changed to the unit given or the cube.unit if not given, for 1D plots.

        axes_coordinates: list of physical coordinates for array or None
            If None array indices will be used for all axes.
            If a list it should contain one element for each axis of the numpy array.
            For the image axes a [min, max] pair should be specified which will be
            passed to :func:`matplotlib.pyplot.imshow` as extent.
            For the slider axes a [min, max] pair can be specified or an array the
            same length as the axis which will provide all values for that slider.
            If None is specified for an axis then the array indices will be used
            for that axis.

        """
        # If old API is used, convert to new API.
        plot_axis_indices, axes_coordiantes, axes_units, data_unit, kwargs = _support_101_plot_API(
            plot_axis_indices, axes_coordinates, axes_units, data_unit, kwargs)
        # Check kwargs are in consistent formats and set default values if not done so by user.
        naxis = len(self.dimensions)
        plot_axis_indices, axes_coordinates, axes_units = _prep_axes_kwargs(
            naxis, plot_axis_indices, axes_coordinates, axes_units)
        if self.data.ndim is 1:
            plot = self._plot_1D_cube(data_unit=data_unit, origin=origin)
        elif self.data.ndim is 2:
            plot = self._plot_2D_cube(axes=axes, plot_axis_indices=plot_axis_indices, **kwargs)
        else:
            plot = self._plot_3D_cube(plot_axis_indices=plot_axis_indices,
                                      axes_coordinates=axes_coordinates, axes_units=axes_units,
                                      **kwargs)
        return plot

    def _plot_1D_cube(self, axes=None, axes_coordinates=None, axes_units=None, data_unit=None,
                      **kwargs):
        """
        Plots a graph.
        Keyword arguments are passed on to matplotlib.

        Parameters
        ----------
        data_unit: `astropy.unit.Unit`
            The data is changed to the unit given or the cube.unit if not given.

        """
        # Derive x-axis coordinates and unit from inputs.
        x_axis_coordinates, unit_x_axis = _derive_1D_coordinates_and_units(axes_coordinates,
                                                                           axes_units)
        if x_axis_coordinates is None:
            # Default is to derive x coords and defaul xlabel from WCS object.
            default_xlabel = self.world_axis_physical_types[0]
            x_axis_coordinates = "{0} [{1}]".format(self.axis_world_coords(), unit_x_axis)
        elif isinstance(x_axis_coordinates, str):
            # User has entered a str as x coords, get that extra coord.
            default_xlabel = "{0} [{1}]".format(x_axis_coordinates, unit_x_axis)
            x_axis_coordinates = self.extra_coords[x_axis_coordinates]["value"]
        # Else user must have set the x-values manually.
        # If a unit has been set for the x-axis, try to convert x coords to that unit.
        if isinstance(unit_x_axis, (u.UnitBase, str)):
            if isinstance(x_axis_coordinates, u.Quantity):
                default_xlabel = " [{0}]".format(unit_x_axis)
                x_axis_coordinates = x_axis_coordinates.to(unit_x_axis)
            else:
                raise TypeError("Can only set unit for x axis if x-axis coordinates is input as "
                                "None, an astropy Quantity or the name of an extra coord that "
                                "is an astropy Quantity.")
        # Derive y-axis coordinates, uncertainty and unit from the NDCube's data.
        if self.unit is None:
            if data_unit is not None:
                raise ValueError("Can only set y-axis unit if self.unit is set to a "
                                 "compatible unit.")
            else:
                ydata = self.data
                yerror = self.uncertainty.array
        else:
            if data_unit is None:
                data_unit = self.unit
                ydata = self.data
                if self.uncertainty is None:
                    yerror = None
                else:
                    yerror = self.uncertainty.array
            else:
                ydata = (self.data * self.unit).to(data_unit).value
                if self.uncertainty is None:
                    yerror = None
                else:
                    yerror = (self.uncertainty.array * self.unit).to(data_unit).value
        # Combine data and uncertainty with mask.
        ydata = np.ma.masked_array(ydata, self.mask)
        if yerror is not None:
            yerror = np.ma.masked_array(yerror, self.mask)
        # Create plot
        fig, ax = _make_1D_sequence_plot(xdata, ydata, yerror, unit_y_axis, default_xlabel, kwargs)
        return ax

    def _plot_2D_cube(self, axes=None, plot_axis_indices=None, **kwargs):
        """
        Plots a 2D image onto the current
        axes. Keyword arguments are passed on to matplotlib.

        Parameters
        ----------
        axes: `astropy.visualization.wcsaxes.core.WCSAxes` or `None`:
            The axes to plot onto. If None the current axes will be used.

        plot_axis_indices: `list`.
            The first axis in WCS object will become the first axis of plot_axis_indices and
            second axis in WCS object will become the second axis of plot_axis_indices.
            Default: ['x', 'y']
        """
        if plot_axis_indices is None:
            axis_data = ['x', 'y']
        else:
            axis_data = ['x' for i in range(2)]
            axis_data[plot_axis_indices[1]] = 'y'
            axis_data = axis_data[::-1]
        if axes is None:
            if self.wcs.naxis is not 2:
                missing_axis = self.missing_axis
                slice_list = []
                index = 0
                for i, bool_ in enumerate(missing_axis):
                    if not bool_:
                        slice_list.append(axis_data[index])
                        index += 1
                    else:
                        slice_list.append(1)
                if index is not 2:
                    raise ValueError("Dimensions of WCS and data don't match")
            axes = wcsaxes_compat.gca_wcs(self.wcs, slices=slice_list)
        plot = axes.imshow(self.data, **kwargs)
        return plot

    def _plot_3D_cube(self, plot_axis_indices=None, axes_units=None,
                      axes_coordinates=None, **kwargs):
        """
        Plots an interactive visualization of this cube using sliders to move through axes
        plot using in the image.
        Parameters other than data and wcs are passed to ImageAnimatorWCS, which in turn
        passes them to imshow.

        Parameters
        ----------
        plot_axis_indices: `list`
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
        """
        if plot_axis_indices is None:
            plot_axis_indices = [-1, -2]
        if axes_units is None:
            axes_units = [None, None]
        i = ImageAnimatorWCS(self.data, wcs=self.wcs, image_axes=plot_axis_indices,
                             unit_x_axis=axes_units[0], unit_y_axis=axes_units[1],
                             axis_ranges=axes_coordinates, **kwargs)
        return i


def _support_101_plot_API(plot_axis_indices, axes_coordinates, axes_units, data_unit, kwargs):
    """Check if user has used old API and convert it to new API."""
    # Get old API variable values.
    image_axes = kwargs.pop("image_axes", None)
    axis_ranges = kwargs.pop("axis_ranges", None)
    unit_x_axis = kwargs.pop("unit_x_axis", None)
    unit_y_axis = kwargs.pop("unit_y_axis", None)
    unit = kwargs.pop("unit", None)
    # Check if conflicting new and old API values have been set.
    # If not, set new API using old API and raise deprecation warning.
    if image_axes is not None:
        variable_names = ("image_axes", "plot_axis_indices")
        _raise_101_API_deprecation_warning(*variable_names)
        if plot_axis_indices is None:
            plot_axis_indices = image_axes
        else:
            _raise_API_error(*variable_names)
    if axis_ranges is not None:
        variable_names = ("axis_ranges", "axes_coordinates")
        _raise_101_API_deprecation_warning(*variable_names)
        if axes_coordinates is None:
            axes_coordinates = axis_ranges
        else:
            _raise_API_error(*variable_names)
    if (unit_x_axis is not None or unit_y_axis is not None) and axes_units is not None:
        _raise_API_error("unit_x_axis and/or unit_y_axis", "axes_units")
    if axes_units is None:
        variable_names = ("unit_x_axis and unit_y_axis", "axes_units")
        if unit_x_axis is not None:
            _raise_101_API_deprecation_warning(*variable_names)
            if len(plot_axis_indices) == 1:
                axes_units = unit_x_axis
            elif len(plot_axis_indices) == 2:
                if unit_y_axis is None:
                    axes_units = [unit_x_axis, None]
                else:
                    axes_units = [unit_x_axis, unit_y_axis]
            else:
                raise ValueError("Length of image_axes must be less than 3.")
        else:
            if unit_y_axis is not None:
                _raise_101_API_deprecation_warning(*variable_names)
                axes_units = [None, unit_y_axis]
    if unit is not None:
        variable_names = ("unit", "data_unit")
        _raise_101_API_deprecation_warning(*variable_names)
        if data_unit is None:
            data_unit = unit
        else:
            _raise_API_error(*variable_names)
    # Return values of new API
    return plot_axis_indices, axes_coordinates, axes_units, data_unit, kwargs


def _raise_API_error(old_name, new_name):
    raise ValueError(
        "Conflicting inputs: {0} (old API) cannot be set if {1} (new API) is set".format(
            old_name, new_name))

def _raise_101_API_deprecation_warning(old_name, new_name):
    warn("{0} is deprecated and will not be supported in version 2.0.  It will be replaced by {1}.  See docstring.".format(old_name, new_name), DeprecationWarning)
