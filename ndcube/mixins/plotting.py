import datetime
from warnings import warn

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u
import sunpy.visualization.wcsaxes_compat as wcsaxes_compat
try:
    from sunpy.visualization.animator import ImageAnimator, ImageAnimatorWCS, LineAnimator
except ImportError:
    from sunpy.visualization.imageanimator import ImageAnimator, ImageAnimatorWCS, LineAnimator

from ndcube import utils
from ndcube.utils.cube import _get_extra_coord_edges
from ndcube.mixins import sequence_plotting

__all__ = ['NDCubePlotMixin']

INVALID_UNIT_SET_MESSAGE = "Can only set unit for axis if corresponding coordinates in " + \
  "axes_coordinates are set to None, an astropy Quantity or the name of an extra coord that " + \
  "is an astropy Quantity."


class NDCubePlotMixin:
    """
    Add plotting functionality to a NDCube class.
    """

    def plot(self, axes=None, plot_axis_indices=None, axes_coordinates=None,
             axes_units=None, data_unit=None, **kwargs):
        """
        Plots an interactive visualization of this cube with a slider
        controlling the wavelength axis for data having dimensions greater than 2.
        Plots an x-y graph onto the current axes for 2D or 1D data.
        Keyword arguments are passed on to matplotlib.
        Parameters other than data and wcs are passed to ImageAnimatorWCS,
        which in turn passes them to imshow for data greater than 2D.

        Parameters
        ----------
        plot_axis_indices: `list`
            The two axes that make the image.
            Default=[-1,-2].  This implies cube instance -1 dimension
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
            The physical coordinates expected by axes_coordinates should be an array of
            pixel_edges.
            A str entry in axes_coordinates signifies that an extra_coord will be used for the axis's coordinates.
            The str must be a valid name of an extra_coord that corresponds to the same axis to which it is applied in the plot.



        """
        # If old API is used, convert to new API.
        plot_axis_indices, axes_coordinates, axes_units, data_unit, kwargs = _support_101_plot_API(
            plot_axis_indices, axes_coordinates, axes_units, data_unit, kwargs)
        # Check kwargs are in consistent formats and set default values if not done so by user.
        naxis = len(self.dimensions)
        plot_axis_indices, axes_coordinates, axes_units = sequence_plotting._prep_axes_kwargs(
            naxis, plot_axis_indices, axes_coordinates, axes_units)
        if naxis is 1:
            ax = self._plot_1D_cube(axes, axes_coordinates,
                                    axes_units, data_unit, **kwargs)
        else:
            if len(plot_axis_indices) == 1:
                ax = self._animate_cube_1D(
                    plot_axis_index=plot_axis_indices[0], axes_coordinates=axes_coordinates,
                    axes_units=axes_units, data_unit=data_unit, **kwargs)
            else:
                if naxis == 2:
                    ax = self._plot_2D_cube(axes, plot_axis_indices, axes_coordinates,
                                            axes_units, data_unit, **kwargs)
                else:
                    ax = self._plot_3D_cube(
                        plot_axis_indices=plot_axis_indices, axes_coordinates=axes_coordinates,
                        axes_units=axes_units, **kwargs)
        return ax

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
        x_axis_coordinates, unit_x_axis = sequence_plotting._derive_1D_coordinates_and_units(
            axes_coordinates, axes_units)
        if x_axis_coordinates is None:
            # Default is to derive x coords and defaul xlabel from WCS object.
            xname = self.world_axis_physical_types[0]
            xdata = self.axis_world_coords()
        elif isinstance(x_axis_coordinates, str):
            # User has entered a str as x coords, get that extra coord.
            xname = x_axis_coordinates
            xdata = self.extra_coords[x_axis_coordinates]["value"]
        else:
            # Else user must have set the x-values manually.
            xname = ""
            xdata = x_axis_coordinates
        # If a unit has been set for the x-axis, try to convert x coords to that unit.
        if isinstance(xdata, u.Quantity):
            if unit_x_axis is None:
                unit_x_axis = xdata.unit
                xdata = xdata.value
            else:
                xdata = xdata.to(unit_x_axis).value
        else:
            if unit_x_axis is not None:
                raise TypeError(INVALID_UNIT_SET_MESSAGE)
        # Define default x axis label.
        default_xlabel = "{0} [{1}]".format(xname, unit_x_axis)
        # Combine data and uncertainty with mask.
        xdata = np.ma.masked_array(xdata, self.mask)
        # Derive y-axis coordinates, uncertainty and unit from the NDCube's data.
        if self.unit is None:
            if data_unit is not None:
                raise TypeError("Can only set y-axis unit if self.unit is set to a "
                                "compatible unit.")
            else:
                ydata = self.data
                if self.uncertainty is None:
                    yerror = None
                else:
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
        fig, ax = sequence_plotting._make_1D_sequence_plot(xdata, ydata, yerror,
                                                           data_unit, default_xlabel, kwargs)
        return ax

    def _plot_2D_cube(self, axes=None, plot_axis_indices=None, axes_coordinates=None,
                      axes_units=None, data_unit=None, **kwargs):
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
        # Set default values of kwargs if not set.
        if axes_coordinates is None:
            axes_coordinates = [None, None]
        if axes_units is None:
            axes_units = [None, None]
        # Set which cube dimensions are on the x an y axes.
        axis_data = ['x', 'x']
        axis_data[plot_axis_indices[1]] = 'y'
        axis_data = axis_data[::-1]
        # Determine data to be plotted
        if data_unit is None:
            data = self.data
        else:
            # If user set data_unit, convert dat to desired unit if self.unit set.
            if self.unit is None:
                raise TypeError("Can only set data_unit if NDCube.unit is set.")
            else:
                data = (self.data * self.unit).to(data_unit).value
        # Combine data with mask
        data = np.ma.masked_array(data, self.mask)
        if axes is None:
            try:
                axes_coord_check = axes_coordinates == [None, None]
            except:
                axes_coord_check = False
            if axes_coord_check:
                # Build slice list for WCS for initializing WCSAxes object.
                if self.wcs.naxis != 2:
                    slice_list = []
                    index = 0
                    for bool_ in self.missing_axes:
                        if not bool_:
                            slice_list.append(axis_data[index])
                            index += 1
                        else:
                            slice_list.append(1)
                    if index != 2:
                        raise ValueError("Dimensions of WCS and data don't match")
                    ax = wcsaxes_compat.gca_wcs(self.wcs, slices=tuple(slice_list))
                else:
                    ax = wcsaxes_compat.gca_wcs(self.wcs)
                # Set axis labels
                x_wcs_axis = utils.cube.data_axis_to_wcs_axis(plot_axis_indices[0],
                                                              self.missing_axes)
                ax.set_xlabel("{0} [{1}]".format(
                    self.world_axis_physical_types[plot_axis_indices[0]],
                    self.wcs.wcs.cunit[x_wcs_axis]))
                y_wcs_axis = utils.cube.data_axis_to_wcs_axis(plot_axis_indices[1],
                                                              self.missing_axes)
                ax.set_ylabel("{0} [{1}]".format(
                    self.world_axis_physical_types[plot_axis_indices[1]],
                    self.wcs.wcs.cunit[y_wcs_axis]))
                # Plot data
                ax.imshow(data, **kwargs)
            else:
                # Else manually set axes x and y values based on user's input for axes_coordinates.
                new_axes_coordinates, new_axis_units, default_labels = \
                  self._derive_axes_coordinates(axes_coordinates, axes_units, data.shape)
                # Initialize axes object and set values along axis.
                fig, ax = plt.subplots(1, 1)
                # Since we can't assume the x-axis will be uniform, create NonUniformImage
                # axes and add it to the axes object.
                if plot_axis_indices[0] < plot_axis_indices[1]:
                    data = data.transpose()
                im_ax = mpl.image.NonUniformImage(
                    ax, extent=(new_axes_coordinates[plot_axis_indices[0]][0],
                                new_axes_coordinates[plot_axis_indices[0]][-1],
                                new_axes_coordinates[plot_axis_indices[1]][0],
                                new_axes_coordinates[plot_axis_indices[1]][-1]), **kwargs)
                im_ax.set_data(new_axes_coordinates[plot_axis_indices[0]],
                               new_axes_coordinates[plot_axis_indices[1]], data)
                ax.add_image(im_ax)
                # Set the limits, labels, etc. of the axes.
                xlim = kwargs.pop("xlim", (new_axes_coordinates[plot_axis_indices[0]][0],
                                           new_axes_coordinates[plot_axis_indices[0]][-1]))
                ax.set_xlim(xlim)
                ylim = kwargs.pop("xlim", (new_axes_coordinates[plot_axis_indices[1]][0],
                                           new_axes_coordinates[plot_axis_indices[1]][-1]))
                ax.set_ylim(ylim)
                xlabel = kwargs.pop("xlabel", default_labels[plot_axis_indices[0]])
                ylabel = kwargs.pop("ylabel", default_labels[plot_axis_indices[1]])
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
        return ax

    def _plot_3D_cube(self, plot_axis_indices=None, axes_coordinates=None,
                      axes_units=None, data_unit=None, **kwargs):
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
            The physical coordinates expected by axes_coordinates should be an array of
            pixel_edges.
            A str entry in axes_coordinates signifies that an extra_coord will be used for the axis's coordinates.
            The str must be a valid name of an extra_coord that corresponds to the same axis to which it is applied in the plot.

        """
        # For convenience in inserting dummy variables later, ensure
        # plot_axis_indices are all positive.
        plot_axis_indices = [i if i >= 0 else self.data.ndim + i for i in plot_axis_indices]
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
        data = np.ma.masked_array(data, self.mask)
        # If axes_coordinates not provided generate an ImageAnimatorWCS plot
        # using NDCube's wcs object.
        new_axes_coordinates, new_axes_units, default_labels = \
              self._derive_axes_coordinates(axes_coordinates, axes_units, data.shape, edges=True)

        if (axes_coordinates[plot_axis_indices[0]] is None and
                axes_coordinates[plot_axis_indices[1]] is None):

            # If there are missing axes in WCS object, add corresponding dummy axes to data.
            if data.ndim < self.wcs.naxis:
                new_shape = list(data.shape)
                for i in np.arange(self.wcs.naxis)[self.missing_axes[::-1]]:
                    new_shape.insert(i, 1)
                    # Also insert dummy coordinates and units.
                    new_axes_coordinates.insert(i, None)
                    new_axes_units.insert(i, None)
                    # Iterate plot_axis_indices if neccessary
                    for j, pai in enumerate(plot_axis_indices):
                        if pai >= i:
                            plot_axis_indices[j] = plot_axis_indices[j] + 1
                # Reshape data
                data = data.reshape(new_shape)
            # Generate plot
            ax = ImageAnimatorWCS(data, wcs=self.wcs, image_axes=plot_axis_indices,
                                  unit_x_axis=new_axes_units[plot_axis_indices[0]],
                                  unit_y_axis=new_axes_units[plot_axis_indices[1]],
                                  axis_ranges=new_axes_coordinates, **kwargs)

            # Set the labels of the plot
            ax.axes.coords[0].set_axislabel(self.wcs.world_axis_physical_types[plot_axis_indices[0]])
            ax.axes.coords[1].set_axislabel(self.wcs.world_axis_physical_types[plot_axis_indices[1]])

        # If one of the plot axes is set manually, produce a basic ImageAnimator object.
        else:
            # If axis labels not set by user add to kwargs.
            ax = ImageAnimator(data, image_axes=plot_axis_indices,
                               axis_ranges=new_axes_coordinates, **kwargs)

            # Add the labels of the plot
            ax.axes.set_xlabel(default_labels[plot_axis_indices[0]])
            ax.axes.set_ylabel(default_labels[plot_axis_indices[1]])
        return ax

    def _animate_cube_1D(self, plot_axis_index=-1, axes_coordinates=None,
                         axes_units=None, data_unit=None, **kwargs):
        """Animates an axis of a cube as a line plot with sliders for other axes."""
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
            index = utils.wcs.get_dependent_data_axes(self.wcs, plot_axis_index, self.missing_axes)
            reduce_axis = np.where(index == np.array(plot_axis_index))[0]

            index = np.delete(index, reduce_axis)
            # Reduce the data by taking mean
            xdata = np.mean(xdata, axis=tuple(index))
        axes_coordinates[plot_axis_index] = xdata
        # Set default x label
        default_xlabel = "{0} [{1}]".format(xname, unit_x_axis)
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
        #data = np.ma.masked_array(data, self.mask)
        # Set default y label
        default_ylabel = "Data [{0}]".format(unit_x_axis)
        # Initiate line animator object.
        ax = LineAnimator(data, plot_axis_index=plot_axis_index, axis_ranges=axes_coordinates,
                          xlabel=default_xlabel,
                          ylabel="Data [{0}]".format(data_unit), **kwargs)
        return ax

    def _derive_axes_coordinates(self, axes_coordinates, axes_units, data_shape, edges=False):
        new_axes_coordinates = []
        new_axes_units = []
        default_labels = []
        default_label_text = ""
        for i, axis_coordinate in enumerate(axes_coordinates):
            # If axis coordinate is None, derive axis values from WCS.
            if axis_coordinate is None:

                # If the new_axis_coordinate is not independent, i.e. dimension is >2D
                # and not equal to dimension of data, then the new_axis_coordinate must
                # be reduced to a 1D ndarray by taking the mean along all non-plotting axes.
                new_axis_coordinate = self.axis_world_coords(i, edges=edges)
                axis_label_text = self.world_axis_physical_types[i]
                # If the shape of the data is not 1, or all the axes are not dependent
                if new_axis_coordinate.ndim != 1 and new_axis_coordinate.ndim != len(data_shape):
                    index = utils.wcs.get_dependent_data_axes(self.wcs, i, self.missing_axes)
                    reduce_axis = np.where(index == np.array([i]))[0]

                    index = np.delete(index, reduce_axis)
                    # Reduce the data by taking mean
                    new_axis_coordinate = np.mean(new_axis_coordinate, axis=tuple(index))

            elif isinstance(axis_coordinate, str):
                # If axis coordinate is a string, derive axis values from
                # corresponding extra coord.
                # Calculate edge value if required
                new_axis_coordinate = _get_extra_coord_edges(self.extra_coords[axis_coordinate]["value"]) if edges else \
                                        self.extra_coords[axis_coordinate]["value"]
                axis_label_text = axis_coordinate
            else:
                # Else user must have manually set the axis coordinates.
                new_axis_coordinate = axis_coordinate
                axis_label_text = default_label_text
            # If axis coordinate is a Quantity, convert to unit supplied by user.
            if isinstance(new_axis_coordinate, u.Quantity):
                if axes_units[i] is None:
                    new_axis_unit = new_axis_coordinate.unit
                    new_axis_coordinate = new_axis_coordinate.value
                else:
                    new_axis_unit = axes_units[i]
                    new_axis_coordinate = new_axis_coordinate.to(new_axis_unit).value
            elif isinstance(new_axis_coordinate[0], datetime.datetime):
                axis_label_text = "{0}/sec since {1}".format(
                    axis_label_text, new_axis_coordinate[0])
                new_axis_coordinate = np.array([(t-new_axis_coordinate[0]).total_seconds()
                                                for t in new_axis_coordinate])
                new_axis_unit = u.s
            else:
                if axes_units[i] is None:
                    new_axis_unit = None
                else:
                    raise TypeError(INVALID_UNIT_SET_MESSAGE)

            # Derive default axis label
            if type(new_axis_coordinate) is datetime.datetime:
                if axis_label_text == default_label_text:
                    default_label = "{0}".format(new_axis_coordinate.strftime("%Y/%m/%d %H:%M"))
                else:
                    default_label = "{0} [{1}]".format(
                        axis_label_text, new_axis_coordinate.strftime("%Y/%m/%d %H:%M"))
            else:
                default_label = "{0} [{1}]".format(axis_label_text, new_axis_unit)
            # Append new coordinates, units and labels to output list.
            new_axes_coordinates.append(new_axis_coordinate)
            new_axes_units.append(new_axis_unit)
            default_labels.append(default_label)
        return new_axes_coordinates, new_axes_units, default_labels


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
