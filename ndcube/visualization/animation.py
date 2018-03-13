import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u
from sunpy.visualization.imageanimator import ImageAnimatorWCS

from ndcube import utils


class ImageAnimatorNDCubeSequence(ImageAnimatorWCS):
    """
    Animates N-dimensional data with the associated astropy WCS object.

    The following keyboard shortcuts are defined in the viewer:

    left': previous step on active slider
    right': next step on active slider
    top': change the active slider up one
    bottom': change the active slider down one
    'p': play/pause active slider

    This viewer can have user defined buttons added by specifying the labels
    and functions called when those buttons are clicked as keyword arguments.

    Parameters
    ----------
    seq: `ndcube.datacube.CubeSequence`
        The list of cubes.

    image_axes: `list`
        The two axes that make the image

    fig: `matplotlib.figure.Figure`
        Figure to use

    axis_ranges: list of physical coordinates for array or None
        If None array indices will be used for all axes.
        If a list it should contain one element for each axis of the numpy array.
        For the image axes a [min, max] pair should be specified which will be
        passed to :func:`matplotlib.pyplot.imshow` as extent.
        For the slider axes a [min, max] pair can be specified or an array the
        same length as the axis which will provide all values for that slider.
        If None is specified for an axis then the array indices will be used
        for that axis.

    interval: `int`
        Animation interval in ms

    colorbar: `bool`
        Plot colorbar

    button_labels: `list`
        List of strings to label buttons

    button_func: `list`
        List of functions to map to the buttons

    unit_x_axis: `astropy.units.Unit`
        The unit of x axis.

    unit_y_axis: `astropy.units.Unit`
        The unit of y axis.

    Extra keywords are passed to imshow.

    """
    def __init__(self, seq, wcs=None, **kwargs):
        if seq._common_axis is not None:
            raise ValueError("Common axis can't set set to use this class. "
                             "Use ImageAnimatorCommonAxisNDCubeSequence.")
        if wcs is None:
            wcs = seq[0].wcs
        self.sequence = seq.data
        self.cumul_cube_lengths = np.cumsum(np.ones(len(self.sequence)))
        data_concat = np.stack([cube.data for cube in seq.data])
        # Add dimensions of length 1 of concatenated data array
        # shape for an missing axes.
        if seq[0].wcs.naxis != len(seq.dimensions) - 1:
            new_shape = list(data_concat.shape)
            for i in np.arange(seq[0].wcs.naxis)[seq[0].missing_axis[::-1]]:
                new_shape.insert(i+1, 1)
            data_concat = data_concat.reshape(new_shape)
        # Add dummy axis to WCS object to represent sequence axis.
        new_wcs = utils.wcs.append_sequence_axis_to_wcs(wcs)

        super(ImageAnimatorNDCubeSequence, self).__init__(data_concat, wcs=new_wcs, **kwargs)


class ImageAnimatorCommonAxisNDCubeSequence(ImageAnimatorWCS):
    """
    Animates N-dimensional data with the associated astropy WCS object.

    The following keyboard shortcuts are defined in the viewer:

    left': previous step on active slider
    right': next step on active slider
    top': change the active slider up one
    bottom': change the active slider down one
    'p': play/pause active slider

    This viewer can have user defined buttons added by specifying the labels
    and functions called when those buttons are clicked as keyword arguments.

    Parameters
    ----------
    seq: `ndcube.datacube.CubeSequence`
        The list of cubes.

    image_axes: `list`
        The two axes that make the image

    fig: `matplotlib.figure.Figure`
        Figure to use

    axis_ranges: list of physical coordinates for array or None
        If None array indices will be used for all axes.
        If a list it should contain one element for each axis of the numpy array.
        For the image axes a [min, max] pair should be specified which will be
        passed to :func:`matplotlib.pyplot.imshow` as extent.
        For the slider axes a [min, max] pair can be specified or an array the
        same length as the axis which will provide all values for that slider.
        If None is specified for an axis then the array indices will be used
        for that axis.

    interval: `int`
        Animation interval in ms

    colorbar: `bool`
        Plot colorbar

    button_labels: `list`
        List of strings to label buttons

    button_func: `list`
        List of functions to map to the buttons

    unit_x_axis: `astropy.units.Unit`
        The unit of x axis.

    unit_y_axis: `astropy.units.Unit`
        The unit of y axis.

    Extra keywords are passed to imshow.

    """
    def __init__(self, seq, wcs=None, **kwargs):
        if seq._common_axis is None:
            raise ValueError("Common axis must be set to use this class. "
                             "Use ImageAnimatorNDCubeSequence.")
        if wcs is None:
            wcs = seq[0].wcs
        self.sequence = seq.data
        self.cumul_cube_lengths = np.cumsum(np.array(
            [c.dimensions[0].value for c in self.sequence], dtype=int))
        data_concat = np.concatenate([cube.data for cube in seq.data], axis=seq._common_axis)
        # Add dimensions of length 1 of concatenated data array
        # shape for an missing axes.
        if seq[0].wcs.naxis != len(seq.dimensions) - 1:
            new_shape = list(data_concat.shape)
            for i in np.arange(seq[0].wcs.naxis)[seq[0].missing_axis[::-1]]:
                new_shape.insert(i, 1)
            data_concat = data_concat.reshape(new_shape)

        super(ImageAnimatorCommonAxisNDCubeSequence, self).__init__(
            data_concat, wcs=wcs, **kwargs)

    def update_plot(self, val, im, slider):
        val = int(val)
        ax_ind = self.slider_axes[slider.slider_ind]
        ind = np.argmin(np.abs(self.axis_ranges[ax_ind] - val))
        self.frame_slice[ax_ind] = ind
        list_slices_wcsaxes = list(self.slices_wcsaxes)
        sequence_index, cube_index = utils.sequence._convert_cube_like_index_to_sequence_indices(
            val, self.cumul_cube_lengths)
        list_slices_wcsaxes[self.wcs.naxis-ax_ind-1] = cube_index
        self.slices_wcsaxes = list_slices_wcsaxes
        if val != slider.cval:
            self.axes.reset_wcs(
                wcs=self.sequence[sequence_index].wcs, slices=self.slices_wcsaxes)
            self._set_unit_in_axis(self.axes)
            im.set_array(self.data[self.frame_slice])
            slider.cval = val


def _plot_2D_sequence_without_common_axis(cubesequence, image_axes=[-1, -2], data_unit=None,
                                          cube_axis_unit=None, **kwargs):
    """
    Plots an NDCubeSequence of 1D NDCubes without a common axis as an image.

    **kwargs are fed into matplotlib.image.NonUniformImage.

    Parameters
    ----------
    cubesequence: `ndcube.NDCubeSequence`
       NDCubeSequence instance to be plotted.

    image_axes: `list`
        The first axis in WCS object will become the first axis of image_axes and
        second axis in WCS object will become the second axis of image_axes.
        Default: ['x', 'y']

    data_unit: `astropy.units.Unit` or unit `str`
        The unit the data in the sequence should be displayed in.  Can only be set it
        the unit attributes of all sub-cubes is set to a compatible unit.
        Default is data unit of first sub-cube.

    cube_axis_unit: `astropy.units.Unit` or unit `str`
        The unit in which to display the axis corresponding to the cube axis
        (i.e. not the sequence axis.). Default is WCS unit of axis of 0th sub-cube.

    """
    image_axes = [len(cubesequence.dimensions)+i if i < 0 else i for i in image_axes]
    # Check that the unit attribute is set of all cubes and
    # derive unit_y_axis if not set.
    sequence_units, data_unit = _determine_sequence_units(cubesequence.data, data_unit)
    # If all cubes have unit set, create a data quantity from cube's data.
    if sequence_units is not None:
        data = np.stack([(cube.data * sequence_units[i]).to(data_unit).value
                         for i, cube in enumerate(cubesequence.data)])
    else:
        data = np.stack([cube.data for i, cube in enumerate(cubesequence.data)])
    # Transpose data if user-defined images_axes require it.
    if image_axes[0] < image_axes[1]:
        data = data.transpose()
    # Derive the x and y axes.
    if cube_axis_unit is None:
        cube_axis_unit = np.array(cubesequence[0].wcs.wcs.cunit)[
            np.invert(cubesequence[0].missing_axis)][0]
    cube_axis = cubesequence[0].axis_world_coords().to(cube_axis_unit)
    sequence_axis = np.arange(len(cubesequence.data))
    axes_values = [sequence_axis, cube_axis.value]
    axes_labels = ["{0} [None]".format(cubesequence.world_axis_physical_types[0]),
                   "{0} [{1}]".format(cubesequence.world_axis_physical_types[1], cube_axis_unit)]
    # Plot image.
    # Create figure and axes objects.
    fig, ax = plt.subplots(1, 1)
    # Since we can't assume the x-axis will be uniform, create NonUniformImage
    # axes and add it to the axes object.
    im_ax = mpl.image.NonUniformImage(
        ax, extent=(axes_values[image_axes[0]][0], axes_values[image_axes[0]][-1],
                    axes_values[image_axes[1]][0], axes_values[image_axes[1]][-1]),
        **kwargs)
    im_ax.set_data(axes_values[image_axes[0]], axes_values[image_axes[1]], data)
    ax.add_image(im_ax)
    # Set the limits, labels, etc. of the axes.
    ax.set_xlim((axes_values[image_axes[0]][0], axes_values[image_axes[0]][-1]))
    ax.set_ylim((axes_values[image_axes[1]][0], axes_values[image_axes[1]][-1]))
    ax.set_xlabel(axes_labels[image_axes[0]])
    ax.set_ylabel(axes_labels[image_axes[1]])

    return ax


def _plot_2D_sequence_with_common_axis(cubesequence, x_axis_extra_coord=None,
                                       unit_x_axis=None, unit_y_axis=None, **kwargs):
    """
    Plots an NDCubeSequence of 1D NDCubes with a common axis as line plot.

    Parameters
    ----------
    cubesequence: `ndcube.NDCubeSequence`
        NDCubeSequence instance to be plotted.
        Each sub-cube must have 1 and only 1 dimension.

    x_axis_extra_coord: `str` (Optional)
        The name of the extra coord along common axis to be used to describe
        the x-axis if preferred to the WCS.
        Default=None, implies the x-axis is derived from the WCS objects in
        the sequence, rather than an extra coord.

    unit_x_axis: `astropy.units.unit` or valid unit `str` (Optional)
        The units into which the x-axis should be displayed.

    unit_y_axis: `astropy.units.unit` or valid unit `str` (Optional)
        The units into which the y-axis should be displayed.  The unit attribute of all
        the sub-cubes must be set to a compatible unit to set this kwarg.

    """
    # Check that the unit attribute is set of all cubes and
    # derive unit_y_axis if not set.
    sequence_units, unit_y_axis = _determine_sequence_units(cubesequence.data, unit_y_axis)
    # If all cubes have unit set, create a y data quantity from cube's data.
    if sequence_units is not None:
        ydata = np.concatenate([(cube.data * sequence_units[i]).to(unit_y_axis).value
                                for i, cube in enumerate(cubesequence.data)])
        yerror = np.concatenate(
            [(cube.uncertainty.array * sequence_units[i]).to(unit_y_axis).value
             for i, cube in enumerate(cubesequence.data)])
    else:
        # If not all cubes have unit set, create a y data array from cube's data.
        ydata = np.concatenate([cube.data for cube in cubesequence.data])
        yerror = np.array([cube.uncertainty for cube in cubesequence.data])
        if all(yerror == None):
            yerror = None
        else:
            if any(yerror == None):
                w = np.where(yerror == None)[0]
                for i in w:
                    yerror[i] = np.zeros(int(cubesequence[i].dimensions.value))
            yerror = np.concatenate(yerror)
    # Define x-axis data.
    xdata, unit_x_axis, default_xlabel = _derive_1D_x_data(
        cubesequence, unit_x_axis, x_axis_extra_coord, sequence_1d=False)
    # Plot data
    fig, ax = _make_1D_sequence_plot(xdata, ydata, yerror, unit_y_axis, default_xlabel, kwargs)
    return ax


def _plot_1D_sequence(cubesequence, x_axis_extra_coord=None,
                      unit_x_axis=None, unit_y_axis=None, **kwargs):
    """
    Plots an NDCubeSequence of scalar NDCubes as line plot.

    A scalar NDCube is one whose NDCube.data is a scalar rather than an array.

    Parameters
    ----------
    cubesequence: `ndcube.NDCubeSequence`
       NDCubeSequence instance to be plotted.

    unit_y_axis: `astropy.units.unit` or valid unit `str`
        The units into which the y-axis should be displayed.  The unit attribute of all
        the sub-cubes must be set to a compatible unit to set this kwarg.

    """
    # Check that the unit attribute is set of all cubes and
    # derive unit_y_axis if not set.
    sequence_units, unit_y_axis = _determine_sequence_units(cubesequence.data, unit_y_axis)
    # If all cubes have unit set, create a data quantity from cube's data.
    if sequence_units is not None:
        ydata = u.Quantity([cube.data * sequence_units[i]
                            for i, cube in enumerate(cubesequence.data)], unit=unit_y_axis).value
        yerror = u.Quantity([cube.uncertainty.array * sequence_units[i]
                             for i, cube in enumerate(cubesequence.data)], unit=unit_y_axis).value
    # If not all cubes have their unit set, create a data array from cube's data.
    else:
        ydata = np.array([cube.data for cube in cubesequence.data])
        yerror = np.array([cube.uncertainty for cube in cubesequence.data])
    if all(yerror == None):
        yerror = None
    # Define x-axis data.
    xdata, unit_x_axis, default_xlabel = _derive_1D_x_data(
        cubesequence, unit_x_axis, x_axis_extra_coord, sequence_1d=True)
    #xdata = np.arange(ydata.size)
    #default_xlabel = cubesequence.world_axis_physical_types[0]
    #return xdata, ydata, yerror, unit_y_axis, default_xlabel, kwargs
    fig, ax = _make_1D_sequence_plot(xdata, ydata, yerror, unit_y_axis, default_xlabel, kwargs)
    return ax


def _determine_sequence_units(cubesequence_data, unit=None):
    """
    Returns units of cubes in sequence and derives data unit if not set.

    If not all cubes have their unit attribute set, an error is raised.

    Parameters
    ----------
    cubesequence_data: `list` of `ndcube.NDCube`
        Taken from NDCubeSequence.data attribute.

    unit: `astropy.units.Unit` or `None`
        If None, an appropriate unit is derived from first cube in sequence.

    Returns
    -------
    sequence_units: `list` of `astropy.units.Unit`
        Unit of each cube.

    unit: `astropy.units.Unit`
        If input unit is not None, then the same as input.  Otherwise it is
        the unit of the first cube in the sequence.

    """
    # Check that the unit attribute is set of all cubes.  If not, unit_y_axis
    try:
        sequence_units = np.array(utils.sequence._get_all_cube_units(cubesequence_data))
    except ValueError:
        sequence_units = None
    # If all cubes have unit set, create a data quantity from cube's data.
    if sequence_units is not None:
        if unit is None:
            unit = sequence_units[0]
    else:
        unit = None
    return sequence_units, unit


def _derive_1D_x_data(cubesequence, unit_x_axis, x_axis_extra_coord, sequence_1d=True):
    # Derive x data from wcs is extra_coord not set.
    if x_axis_extra_coord is None:
        if unit_x_axis is None:
            unit_x_axis = np.asarray(cubesequence[0].wcs.wcs.cunit)[
                np.invert(cubesequence[0].missing_axis)][0]
        xdata = u.Quantity(np.concatenate([cube.axis_world_coords().to(unit_x_axis).value
                                           for cube in cubesequence]), unit=unit_x_axis)
        default_xlabel = "{0} [{1}]".format(cubesequence.cube_like_world_axis_physical_types[0],
                                            unit_x_axis)
    else:
        # Else derive x-axis from extra coord.
        if sequence_1d is True:
            xdata = cubesequence.sequence_axis_extra_coords[x_axis_extra_coord]
        else:
            xdata = cubesequence.common_axis_extra_coords[x_axis_extra_coord]
        if unit_x_axis is None and isinstance(xdata, u.Quantity):
            unit_x_axis = xdata.unit
        default_xlabel = "{0} [{1}]".format(x_axis_extra_coord, unit_x_axis)

    return xdata, unit_x_axis, default_xlabel


def _make_1D_sequence_plot(xdata, ydata, yerror, unit_y_axis, default_xlabel, kwargs):
    # Define plot settings if not set in kwargs.
    xlabel = kwargs.pop("xlabel", default_xlabel)
    ylabel = kwargs.pop("ylabel", "Data [{0}]".format(unit_y_axis))
    title = kwargs.pop("title", "")
    xlim = kwargs.pop("xlim", None)
    ylim = kwargs.pop("ylim", None)
    # Plot data
    fig, ax = plt.subplots(1, 1)
    ax.errorbar(xdata, ydata, yerr=yerror, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return fig, ax
