try:
    from sunpy.visualization.animator import ArrayAnimatorWCS
except ImportError:
    raise ImportError(
        "Sunpy is required to animate NDCubeSequences. "
        "Either install sunpy or extract data from sequence and visualize manually.")

from .plotting_utils import prep_plot_kwargs

__all__ = ['MatplotlibSequencePlotter']


class MatplotlibSequencePlotter():
    """
    Provide visualization methods for NDCubeSequence which use `matplotlib`.

    This plotter delegates much of the visualization to the plot method of the
    `~ndcube.NDCube` instances in the `~ndcube.NDCubeSequence`
    and assumes they employ the `~ndcube.visualization.mpl_plotter.MatplotlibPlotter`.
    """
    def plot(self, sequence_axis_coords=None, sequence_axis_unit=None, **kwargs):
        """
        Visualize the `~ndcube.NDCubeSequence`.
        """
        if len(self._sequence.dimensions) == 2:
            return self._plot_2D_sequence(sequence_axis_coordinates, sequence_axis_unit, **kwargs)
        else:
            return self._animate_sequence(sequence_axis_coordinates, sequence_axis_unit, **kwargs)

    def _plot_2D_sequence(self, sequence_axis_coords=None, sequence_axis_unit=None, **kwargs):
        raise NotImplementedError("Visualizing sequences of 1-D cubes not currently supported.")

    def _animate_sequence(self, sequence_axis_coords=None, sequence_axis_unit=None, **kwargs):
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
        return SequenceAnimator(sequence, sequence_axis_coords=None, sequence_axis_unit=None, **kwargs)


class SequenceAnimator(ArrayAnimatorWCS):
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
        super().__init__(data, wcs, plot_axes, coord_params=coord_params, **base_kwargs)

    def _sequence_slider_function(self, val, im, slider):
        self._sequence_idx = int(val)
        self.data, self.wcs, plot_axes, coord_params = self._cubes[self._sequence_idx].plotter._prep_animate_args(
            self._cubes[self._sequence_idx].wcs, self._plot_axes, self._axes_units, self._data_unit)
        return super().plot_start_image(self.axes)
