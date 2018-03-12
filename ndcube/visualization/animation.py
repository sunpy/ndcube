import numpy as np
from sunpy.visualization.imageanimator import ImageAnimatorWCS, LineAnimator

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


class LineAnimatorNDCubeSequence(LineAnimator):
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
    def __init__(self, seq, plot_axis_index=-1, axis_ranges=None, unit_x_axis=None,
                 data_unit=None, xlabel=None, ylabel=None, xlim=None, ylim=None, **kwargs):
        #try:
        #    sequence_units, data_unit = _determine_sequence_units(seq.data, data_unit)
        #except ValueError:
        #    sequence_error = None
        # Form single cube of data from sequence.
        #if sequence_error is None:
        #    data_concat = np.stack([cube.data for i, cube in enumerate(seq.data)])
        #else:
        #    data_concat = np.stack([(cube.data * sequence_units[i]).to(data_unit).value
        #                            for i, cube in enumerate(seq.data)])
        data_concat = np.stack([cube.data for i, cube in enumerate(seq.data)])
        # Ensure plot_axis_index is represented in the positive convention.
        if plot_axis_index < 0:
            plot_axis_index = len(seq.dimensions) + plot_axis_index
        # Calculate the x-axis values if axis_ranges not supplied.
        if axis_ranges is None:
            axis_ranges = [None] * len(seq.dimensions)
            if plot_axis_index == 0:
                axis_ranges[plot_axis_index] = np.arange(len(seq.data))
            else:
                cube_plot_axis_index = plot_axis_index - 1
                # Define unit of x-axis if not supplied by user.
                if unit_x_axis is None:
                    wcs_plot_axis_index = utils.cube.data_axis_to_wcs_axis(
                        cube_plot_axis_index, seq[0].missing_axis)
                    unit_x_axis = np.asarray(
                        seq[0].wcs.wcs.cunit)[np.invert(seq[0].missing_axis)][wcs_plot_axis_index]
                # Get x-axis values from each cube and combine into a single
                # array for axis_ranges kwargs.
                x_axis_coords = _get_non_common_axis_x_axis_coords(seq.data, cube_plot_axis_index)
                axis_ranges[plot_axis_index] = np.stack(x_axis_coords)
            # Set axis labels and limits, etc.
            if xlabel is None:
                xlabel = "{0} [{1}]".format(seq.world_axis_physical_types[plot_axis_index],
                                            unit_x_axis)
            if ylabel is None:
                ylabel = "Data [{0}]".format(data_unit)

        super(LineAnimatorNDCubeSequence, self).__init__(
            data_concat, plot_axis_index=plot_axis_index, axis_ranges=axis_ranges,
            xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, **kwargs)


class LineAnimatorCubeLikeNDCubeSequence(LineAnimator):
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
    def __init__(self, seq, plot_axis_index=-1, axis_ranges=None, unit_x_axis=None,
                 data_unit=None, xlabel=None, ylabel=None, xlim=None, ylim=None, **kwargs):
        #try:
        #    sequence_units, data_unit = _determine_sequence_units(seq.data, data_unit)
        #except ValueError:
        #    sequence_error = None
        # Form single cube of data from sequence.
        #if sequence_error is None:
        #    data_concat = np.concatenate([cube.data for i, cube in enumerate(seq.data)],
        #                                 axis=seq._common_axis))
        #else:
        #    data_concat = np.concatenate([(cube.data * sequence_units[i]).to(data_unit).value
        #                                  for i, cube in enumerate(seq.data)],
        #                                 axis=seq._common_axis)
        data_concat = np.concatenate([cube.data for i, cube in enumerate(seq.data)],
                                     axis=seq._common_axis)
        # Ensure plot_axis_index is represented in the positive convention.
        if plot_axis_index < 0:
            plot_axis_index = len(seq.cube_like_dimensions) + plot_axis_index
        # Calculate the x-axis values if axis_ranges not supplied.
        if axis_ranges is None:
            axis_ranges = [None] * len(seq.cube_like_dimensions)
            # Define unit of x-axis if not supplied by user.
            if unit_x_axis is None:
                wcs_plot_axis_index = utils.cube.data_axis_to_wcs_axis(
                    plot_axis_index, seq[0].missing_axis)
                unit_x_axis = np.asarray(
                    seq[0].wcs.wcs.cunit)[np.invert(seq[0].missing_axis)][wcs_plot_axis_index]
            if plot_axis_index == seq._common_axis:
                # Determine whether common axis is dependent.
                x_axis_coords = np.concatenate(
                    [cube.axis_world_coords(plot_axis_index).to(unit_x_axis).value
                    for cube in seq.data], axis=plot_axis_index)
                dependent_axes = utils.wcs.get_dependent_data_axes(
                    seq[0].wcs, plot_axis_index, seq[0].missing_axis)
                if len(dependent_axes) > 1:
                    independent_axes = list(range(data_concat.ndim))
                    for i in list(dependent_axes)[::-1]:
                        independent_axes.pop(i)
                    # Expand dimensionality of x_axis_cube_coords using np.tile
                    tile_shape = tuple(list(np.array(
                        data_concat.shape)[independent_axes]) + [1]*len(dependent_axes))
                    x_axis_coords = np.tile(x_axis_cube_coords, tile_shape)
                    # Since np.tile puts original array's dimensions as last,
                    # reshape x_axis_cube_coords to cube's shape.
                    x_axis_coords = x_axis_coords.reshape(seq.cube_like_dimensions.value)
            else:
                # Get x-axis values from each cube and combine into a single
                # array for axis_ranges kwargs.
                x_axis_coords = _get_non_common_axis_x_axis_coords(seq.data, plot_axis_index)
                axis_ranges[plot_axis_index] = np.concatenate(x_axis_coords, axis=seq._common_axis)
            # Set axis labels and limits, etc.
            if xlabel is None:
                xlabel = "{0} [{1}]".format(
                    seq.cube_like_world_axis_physical_types[plot_axis_index], unit_x_axis)
            if ylabel is None:
                ylabel = "Data [{0}]".format(data_unit)

        super(LineAnimatorCubeLikeNDCubeSequence, self).__init__(
            data_concat, plot_axis_index=plot_axis_index, axis_ranges=axis_ranges,
            xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, **kwargs)


def _get_non_common_axis_x_axis_coords(seq_data, plot_axis_index):
    """Get coords of an axis from NDCubes and combine into single array."""
    x_axis_coords = []
    for i, cube in enumerate(seq_data):
        # Get the x-axis coordinates for each cube.
        #x_axis_cube_coords = cube.axis_world_coords(plot_axis_index).to(unit_x_axis).value
        x_axis_cube_coords = cube.axis_world_coords(plot_axis_index).value
        # If the returned x-values have fewer dimensions than the cube,
        # repeat the x-values through the higher dimensions.
        if x_axis_cube_coords.shape != cube.data.shape:
            # Get sequence axes dependent and independent of plot_axis_index.
            dependent_axes = utils.wcs.get_dependent_data_axes(
                cube.wcs, plot_axis_index, cube.missing_axis)
            independent_axes = list(range(len(cube.dimensions)))
            for i in list(dependent_axes)[::-1]:
                independent_axes.pop(i)
            # Expand dimensionality of x_axis_cube_coords using np.tile
            tile_shape = tuple(list(np.array(
                cube.data.shape)[independent_axes]) + [1]*len(dependent_axes))
            x_axis_cube_coords = np.tile(x_axis_cube_coords, tile_shape)
            # Since np.tile puts original array's dimensions as last,
            # reshape x_axis_cube_coords to cube's shape.
            x_axis_cube_coords = x_axis_cube_coords.reshape(cube.data.shape)
        x_axis_coords.append(x_axis_cube_coords)
    return x_axis_coords
