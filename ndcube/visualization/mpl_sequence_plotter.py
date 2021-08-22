import warnings

import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs.wcsapi import BaseLowLevelWCS

try:
    from sunpy.visualization.animator import ArrayAnimatorWCS
except ImportError:
    raise ImportError(
        "Sunpy is required to animate NDCubeSequences. "
        "Either install sunpy or extract data from sequence and visualize manually.")

from .base import BasePlotter
from .plotting_utils import prep_plot_kwargs

__all__ = ['MatplotlibSequencePlotter']

BAD_DIMS_ERROR_MESSAGE = "NDCubeSequence must contain 1-D NDCubes to use this visualizer."


class MatplotlibSequencePlotter(BasePlotter):
    """
    Provide visualization methods for NDCubeSequence which use `matplotlib`.

    This plotter delegates much of the visualization to the `ndcube.NDCube.plot`
    which is assumed to employ the `~ndcube.visualization.mpl_plotter.MatplotlibPlotter`.
    """
    def __init__(self, sequence=None):
        super().__init__(ndcube=sequence)
        self._sequence = self._ndcube

    def plot(self, sequence_axis_coords=None, sequence_axis_unit=None, **kwargs):
        """
        Visualize the `~ndcube.NDCubeSequence`.

        Parameters
        ----------
        sequence_axis_coords: `str` or array-like (optional)
            The real world value of each step along the sequene axis.
            If `str`, the values are taken from `ndcube.NDCubeSequence.sequence_axis_coords`.

        sequence_axis_unit: `str` or `astropy.units.Unit` (optional)
            The unit in which to display the sequence_axis_coords.
        """
        seq_dims = self._sequence.dimensions
        if len(seq_dims) == 2 and seq_dims[1] < 2:
            return self._plot_line(sequence_axis_coords, sequence_axis_unit, **kwargs)
        else:
            return self.animate(sequence_axis_coords, sequence_axis_unit, **kwargs)

    def animate(self, sequence_axis_coords=None, sequence_axis_unit=None, **kwargs):
        """
        Animate the `~ndcube.NDCubeSequence` with the sequence axis as a slider.

        **kwargs are passed to
        `ndcube.visualization.mpl_plotter.MatplotlibPlotter.plot` and therefore only
        apply to cube axes, not the sequence axis.
        See that method's docstring for definition of **kwargs.

        Parameters
        ----------
        sequence_axis_coords: `str` optional
            The name of the coordinate in `~ndcube.NDCubeSequence.sequence_axis_coords`
            to be used as the slider pixel values.
            If None, array indices will be used.

        sequence_axis_units: `astropy.units.Unit` or `str` (optional)
            The unit in which the sequence_axis_coordinates should be displayed.
            If None, the default unit will be used.
        """
        return SequenceAnimator(self._ndcube, sequence_axis_coords=None, sequence_axis_unit=None, **kwargs)

    def imshow(self, axes=None, transpose=False, **kwargs):
        seq_dims = self._get_sequence_shape()
        if (seq_dims[1:] > 1).sum() > 1:
            raise ValueError(BAD_DIMS_ERROR_MESSAGE)
        if axes is None:
            axes = plt.subplot()
        data, data_unit, uncertainty = self._stack_data(transpose=transpose)
        data = np.squeeze(data)
        if transpose:
            data = data.T
        axes.imshow(data, **kwargs)
        return axes

    def plot_line(self, axes=None, sequence_axis_coords=None, sequence_axis_unit=None, **kwargs):
        seq_dims = self._get_sequence_shape()
        if (seq_dims[1:] != 1).sum() > 1:
            raise ValueError(BAD_DIMS_ERROR_MESSAGE)
        if axes is None:
            axes = plt.subplot()
        # Define y values from data in sequence.
        y, yunit, yerror = self._stack_data()
        y = np.squeeze(y)
        yerror = np.squeeze(yerror)
        # Define x values.
        if sequence_axis_coords:
            x = self._sequence.sequence_axis_coords[sequence_axis_coords]
            if sequence_axis_unit:
                x = x.to(sequence_axis_unit)
            xunit = x.unit
        else:
            x = np.arange(len(y))
            xunit = None
        # Plot values.
        if yerror is None:
            axes.plot(x, y, **kwargs)
        else:
            axes.errorbar(x, y, yerror, **kwargs)
        # Set default labels.
        axes.set_ylabel(f"Data [{yunit}]")
        axes.set_xlabel(f"{sequence_axis_coords} [{xunit}]")

        return axes

    def _get_sequence_shape(self):
        try:
            seq_dims = np.array([int(d.value) for d in self._sequence.dimensions])
        except TypeError:
            raise ValueError("All cubes in sequence must have same shape.")
        return seq_dims

    def _stack_data(self):
        """Generates a stacked array from a sequence of cubes."""
        # Collect data from cubes in sequence.
        dims = tuple(int(d.value) for d in self._sequence.dimensions)
        data = np.zeros(dims)
        uncerts = np.zeros(dims)
        masks = np.zeros(dims, dtype=bool)
        data_unit = None
        uncert_unit = None
        data_unit_present = False
        data_unit_absent = False
        uncert_unit_present = False
        uncert_unit_absent = False
        for i, cube in enumerate(self._sequence.data):
            # Extract data values, converting to a consistent unit if the cube includes a unit.
            if not data_unit and cube.unit:
                data_unit = cube.unit
            data[i] = cube.data
            if cube.unit:
                data[i] = (data[i] * cube.unit).to_value(data_unit)
                data_unit_present = True
            else:
                data_unit_absent = True
            # Extract uncertainty values, converting to appropriate unit if one is present.
            if cube.uncertainty:
                if not uncert_unit and cube.uncertainty.unit:
                    uncert_unit = cube.uncertainty.unit
                uncerts[i] = cube.uncertainty.array
                if uncert_unit:
                    uncerts[i] = (uncerts[i] * cube.uncertainty.unit).to_value(uncert_unit)
                    uncert_unit_present = True
                else:
                    uncert_unit_absent = True
            # Extract masks.
            if isinstance(cube.mask, (bool, type(None))):
                mask = np.zeros(dims[1:], dtype=bool)
                mask[:] = cube.mask
                masks[i] = mask
            else:
                masks[i] = cube.mask
        # If some cubes contain units and others don't, raise a warning that values
        # may not be correctly scaled.
        if data_unit_present and data_unit_absent:
            warnings.warn("Some cubes in sequence have units and others don't. "
                          "Data values may not be correctly scaled.")
        if uncert_unit_present and uncert_unit_absent:
            warnings.warn("Some cubes in sequence have ucertainty units and others don't. "
                          "Uncertainty values may not be correctly scaled.")
        # Convert uncertainties to same unit as data.
        if uncerts.max() == 0:
            uncerts = None
        elif data_unit and uncert_unit and data_unit != uncert_unit:
            uncerts = (uncerts * uncert_unit).to_value(data_unit)
        # If any values are masked, convert data to masked array.
        # Otherwise leave as a numpy array as they are more efficient.
        if masks.any():
            data = np.ma.masked_array(data, masks)
            uncerts = np.ma.masked_array(uncerts, masks)

        return data, data_unit, uncerts


class SequenceAnimator(ArrayAnimatorWCS):
    """
    Animate an NDCubeSequence of NDCubes with >1 dimension.

    The sequence axis is always set as a sliders axis.
    All kwargs are passed to `ndcube.NDCube.plot`.
    The bulk of the plotting work is performed by `ndcube.NDCube.plot`
    which is assumed to exist and to call a matplotlib-based animator.

    Parameters
    ----------
    sequence: `~ndcube.NDCubeSequence`
        The sequence to animate.

    sequence_axis_coords: `str` or array-like (optional)
        The real world value of each step along the sequene axis.
        If `str`, the values are taken from `ndcube.NDCubeSequence.sequence_axis_coords`.

    sequence_axis_unit: `str` or `astropy.units.Unit` (optional)
        The unit in which to display the sequence_axis_coords.
    """

    def __init__(self, sequence, sequence_axis_coords=None, sequence_axis_unit=None, **kwargs):
        if sequence_axis_coords is not None:
            raise NotImplementedError("Setting sequence_axis_coords not yet supported.")
        if sequence_axis_unit is not None:
            raise NotImplementedError("Setting sequence_axis_unit not yet supported.")

        # Store sequence data
        self._cubes = sequence.data

        #  Process kwargs used by cube plotter.
        plot_axes = kwargs.pop("plot_axes", None)
        axes_coordinates = kwargs.pop("axes_coordinates", None)
        axes_units = kwargs.pop("axes_units", None)
        self._data_unit = kwargs.pop("data_unit", None)
        init_idx = 0
        n_cube_dims = len(self._cubes[init_idx].dimensions)
        init_wcs = self._cubes[init_idx].wcs
        self._plot_axes, self._axes_coordinates, self._axes_units = prep_plot_kwargs(
            n_cube_dims, init_wcs, plot_axes, axes_coordinates, axes_units)

        # Define sequence axis slider properties and add to kwargs.
        base_kwargs = {"slider_functions": [self._sequence_slider_function],
                       "slider_ranges": [[0, len(self._cubes)]]}
        base_kwargs.update(kwargs)

        # Calculate data and wcs for initial animation state and instantiate Animator.
        data, wcs, plot_axes, coord_params = self._cubes[0].plotter._prep_animate_args(
            self._cubes[0].wcs, self._plot_axes, self._axes_units, self._data_unit)
        if not isinstance(wcs, BaseLowLevelWCS):
            wcs = wcs.low_level_wcs
        super().__init__(data, wcs, plot_axes, coord_params=coord_params, **base_kwargs)

    def _sequence_slider_function(self, val, im, slider):
        self._sequence_idx = int(val)
        self.data, self.wcs, plot_axes, coord_params = self._cubes[self._sequence_idx].plotter._prep_animate_args(
            self._cubes[self._sequence_idx].wcs, self._plot_axes, self._axes_units, self._data_unit)
        return self._replot(im)

    def _replot(self, im):
        """
        Replot the image without updating cube sliders.
        """
        self.axes.reset_wcs(wcs=self.wcs, slices=self.slices_wcsaxes)
        im.set_array(self.data_transposed)

        if self.clip_interval is not None:
            vmin, vmax = super()._get_2d_plot_limits()
            im.set_clim(vmin, vmax)
