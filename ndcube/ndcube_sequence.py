import copy
import numbers
import textwrap

import astropy.units as u
import numpy as np

from ndcube import utils
from ndcube.visualization import PlotterDescriptor

__all__ = ['NDCubeSequence']


class NDCubeSequenceBase:
    """
    Class representing a sequence of `~ndcube.NDCube`-like objects.

    The cubes are assumed to have the same dimensionality and axis physical types.

    Parameters
    ----------
    data_list : `list`
        List of `ndcube.NDCube`-like objects.

    meta : `dict` or None
        Meta data relevant to the sequence as a whole.

    common_axis: `int` or None
        The array axis of the cubes along which the cubes are ordered.
        For example, if the cubes are sequenced in chronological order and time is
        the 1st axis of each Cube, then common_axis should be set to 0.
        This enables the "cube_like" methods to be used, e.g.
        `ndcube.NDCubeSequence.index_as_cube` which slices the sequence as though it
        were a single cube concatenated along the common axis.
    """
    def __init__(self, data_list, meta=None, common_axis=None, **kwargs):
        self.data = data_list
        self.meta = meta
        if common_axis is not None:
            self._common_axis = int(common_axis)
        else:
            self._common_axis = common_axis

    @property
    def dimensions(self):
        """
        The length of each axis including the sequence axis.
        """
        return self._dimensions

    @property
    def _dimensions(self):
        dimensions = [len(self.data) * u.pix] + list(self.data[0].dimensions)
        if len(dimensions) > 1:
            # If there is a common axis, length of cube's along it may not
            # be the same. Therefore if the lengths are different,
            # represent them as a tuple of all the values, else as an int.
            if self._common_axis is not None:
                common_axis_lengths = [cube.data.shape[self._common_axis] for cube in self.data]
                if len(np.unique(common_axis_lengths)) != 1:
                    common_axis_dimensions = [cube.dimensions[self._common_axis]
                                              for cube in self.data]
                    dimensions[self._common_axis + 1] = u.Quantity(
                        common_axis_dimensions, unit=common_axis_dimensions[0].unit)
        return tuple(dimensions)

    @property
    def array_axis_physical_types(self):
        """
        The physical types associated with each array axis, including the sequence axis.
        """
        return [("meta.obs.sequence",)] + self.data[0].array_axis_physical_types

    @property
    def cube_like_dimensions(self):
        """
        The length of each array axis as if all cubes were concatenated along the common axis.
        """
        if not isinstance(self._common_axis, int):
            raise TypeError("Common axis must be set.")
        dimensions = list(self._dimensions)
        cube_like_dimensions = list(self._dimensions[1:])
        if dimensions[self._common_axis + 1].isscalar:
            cube_like_dimensions[self._common_axis] = u.Quantity(
                dimensions[0].value * dimensions[self._common_axis + 1].value, unit=u.pix)
        else:
            cube_like_dimensions[self._common_axis] = sum(dimensions[self._common_axis + 1])
        # Combine into single Quantity
        cube_like_dimensions = u.Quantity(cube_like_dimensions, unit=u.pix)
        return cube_like_dimensions

    @property
    def cube_like_array_axis_physical_types(self):
        """
        The physical types associated with each array axis, omitting the sequence axis.
        """
        if self._common_axis is None:
            raise ValueError("Common axis must be set.")
        return self.data[0].array_axis_physical_types

    def __getitem__(self, item):
        if isinstance(item, numbers.Integral):
            return self.data[item]
        # Create an empty sequence in which to place the sliced cubes.
        result = type(self)([], meta=self.meta, common_axis=self._common_axis)
        if isinstance(item, slice):
            result.data = self.data[item]
        else:
            if isinstance(item[0], numbers.Integral):
                result = self.data[item[0]][item[1:]]
            else:
                result.data = [cube[item[1:]] for cube in self.data[item[0]]]
            # Determine common axis after slicing.
            if self._common_axis is not None:
                drop_cube_axes = [isinstance(i, numbers.Integral) for i in item[1:]]
                if (len(drop_cube_axes) > self._common_axis and
                        drop_cube_axes[self._common_axis] is True):
                    result._common_axis = None
                else:
                    result._common_axis = \
                        self._common_axis - sum(drop_cube_axes[:self._common_axis])
        return result

    @property
    def index_as_cube(self):
        """
        Slice the NDCubesequence instance as a single cube concatenated along the common axis.

        Example
        -------
        >>> # Say we have three Cubes each cube has common_axis=0 is time and shape=(3,3,3)
        >>> data_list = [cubeA, cubeB, cubeC] # doctest: +SKIP
        >>> cs = NDCubeSequence(data_list, meta=None, common_axis=0) # doctest: +SKIP
        >>> # return zeroth time slice of cubeB in via normal NDCubeSequence indexing.
        >>> cs[1,:,0,:] # doctest: +SKIP
        >>> # Return same slice using this function
        >>> cs.index_as_cube[3:6, 0, :] # doctest: +SKIP
        """
        if self._common_axis is None:
            raise ValueError("common_axis cannot be None")
        return _IndexAsCubeSlicer(self)

    @property
    def common_axis_coords(self):
        """
        The coordinate values at each location along the common axis across all cubes.

        Only coordinates associated with the common axis in all cubes in the sequence
        are returned.  Coordinates from different cubes are concatenated along the
        common axis.  They thus represent the coordinate values at each location as
        if all cubes in the sequence were concatenated along the common axis.
        """
        common_axis = self._common_axis
        # Get coordinate objects associated with the common axis in all cubes.
        common_axis_names = set.intersection(*[set(cube.array_axis_physical_types[common_axis])
                                               for cube in self.data])
        common_coords = []
        mappings = []
        for i, cube in enumerate(self.data):
            cube_wcs = cube.combined_wcs
            common_coords.append(cube.axis_world_coords(common_axis, wcs=cube_wcs))
            mappings.append(utils.wcs.array_indices_for_world_objects(cube_wcs,
                                                                      axes=(common_axis,)))
        # For each coordinate, break up and then combine the coordinate objects across
        # the cubes into a list of coordinate objects that are length-1 and sequential
        # along the common axis.
        sequence_coords = []
        for coord_idx in range(len(common_coords[0])):
            exploded_coord = []
            for cube_idx in range(len(common_coords)):
                coord = common_coords[cube_idx][coord_idx]
                axis = np.where(np.array(mappings[cube_idx][coord_idx]) == common_axis)[0][0]
                item = [slice(None)] * len(coord.shape)
                for i in range(coord.shape[axis]):
                    item[axis] = i
                    exploded_coord.append(coord[tuple(item)])
            sequence_coords.append(exploded_coord)
        return sequence_coords

    @property
    def sequence_axis_coords(self):
        """
        Return the coordinate values along the sequence axis.

        These are compiled from the `~ndcube.GlobalCoords` objects attached to each
        `~ndcube.NDCube` where each cube represents a location along the sequence axis.
        Only coordinates that are common to all cubes are returned.
        """
        # Collect names of global coords common to all cubes.
        global_names = set.intersection(*[set(cube.global_coords.keys()) for cube in self.data])
        # For each coord, combine values from each cube's global coords property.
        return dict([(name, [cube.global_coords[name] for cube in self.data])
                     for name in global_names])

    def explode_along_axis(self, axis):
        """
        Separates slices of N-D cubes along a given cube axis into (N-1)D cubes.

        Parameters
        ----------
        axis : `int`
            The axis along which the data is to be changed.

        Returns
        -------
        `ndcube.NDCubeSequence`
            New sequence of (N-1)D cubes broken up along given axis.
        """
        # If axis is -ve then calculate the axis from the length of the dimensions of one cube.
        if axis < 0:
            axis = len(self.dimensions[1::]) + axis
        # To store the resultant cube
        result_cubes = []
        # All slices are initially initialised as slice(None, None, None)
        result_cubes_slice = [slice(None, None, None)] * len(self[0].data.shape)
        # the range of the axis that needs to be sliced
        range_of_axis = self[0].data.shape[axis]
        for ndcube in self.data:
            for index in range(range_of_axis):
                # setting the slice value to the index so that the slices are done correctly.
                result_cubes_slice[axis] = index
                # appending the sliced cubes in the result_cube list
                result_cubes.append(ndcube.__getitem__(tuple(result_cubes_slice)))
        # Determine common axis for new sequence.
        if self._common_axis is None or self._common_axis == axis:
            new_common_axis = None
        elif self._common_axis > axis:
            new_common_axis = self._common_axis - 1
        elif self._common_axis < axis:
            new_common_axis = self._common_axis
        # creating a new sequence with the result_cubes keeping the meta and common axis as axis
        return self._new_instance(result_cubes, common_axis=new_common_axis, meta=self.meta)

    def __str__(self):
        return (textwrap.dedent(f"""\
                NDCubeSequence
                --------------
                Dimensions:  {self.dimensions}
                Physical Types of Axes: {self.array_axis_physical_types}
                Common Cube Axis: {self._common_axis}"""))

    def __repr__(self):
        return f"{object.__repr__(self)}\n{str(self)}"

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    @classmethod
    def _new_instance(cls, data_list, meta=None, common_axis=None):
        """
        Instantiate a new instance of this class using given data.
        """
        return cls(data_list, meta=meta, common_axis=common_axis)


class NDCubeSequence(NDCubeSequenceBase):
    """
    Class representing a sequence of `~ndcube.NDCube`-like objects.

    The cubes are assumed to have the same dimensionality and axis physical types.

    Parameters
    ----------
    data_list : `list`
        List of `ndcube.NDCube`-like objects.

    meta : `dict` or None
        Meta data relevant to the sequence as a whole.

    common_axis: `int` or None
        The array axis of the cubes along which the cubes are ordered.
        For example, if the cubes are sequenced in chronological order and time is
        the 1st axis of each Cube, then common_axis should be set to 0.
        This enables the "cube_like" methods to be used, e.g.
        `ndcube.NDCubeSequence.index_as_cube` which slices the sequence as though it
        were a single cube concatenated along the common axis.
    """
    # We special case the default mpl plotter here so that we can only import
    # matplotlib when `.plotter` is accessed and raise an ImportError at the
    # last moment.
    plotter = PlotterDescriptor(default_type="mpl_sequence_plotter")

    def plot(self, *args, **kwargs):
        """
        A convenience function for the plotters default ``plot()`` method.

        Calling this method is the same as calling ``sequence.plotter.plot``, the
        behaviour of this method can change if the `NDCubeSequence.plotter` class is
        set to a different ``Plotter`` class.

        """
        if self.plotter is None:
            raise NotImplementedError(
                "This NDCubeSequence object does not have a .plotter defined so "
                "no default plotting functionality is available.")

        return self.plotter.plot(*args, **kwargs)

    def plot_as_cube(self, *args, **kwargs):
        raise NotImplementedError(
            "NDCubeSequence plot_as_cube is no longer supported.\n"
            "To learn why or to tell us why it should be re-instated, "
            "read and comment on issue #315:\n\nhttps://github.com/sunpy/ndcube/issues/315\n\n"
            "To see a introductory guide on how to make your own NDCubeSequence plots, "
            "see the docs:\n\n"
            "https://docs.sunpy.org/projects/ndcube/en/stable/ndcubesequence.html#plotting")


"""
Cube Sequence Helpers
"""


class _IndexAsCubeSlicer:
    """
    Helper class to make slicing in index_as_cube sliceable/indexable like a
    numpy array.

    Parameters
    ----------
    seq : `ndcube.NDCubeSequence`
        Object of NDCubeSequence.
    """

    def __init__(self, seq):
        self.seq = seq

    def __getitem__(self, item):
        common_axis = self.seq._common_axis
        common_axis_lengths = [int(cube.dimensions[common_axis].value) for cube in self.seq.data]
        n_cube_dims = len(self.seq.cube_like_dimensions)
        n_uncommon_cube_dims = n_cube_dims - 1
        # If item is iint or slice, turn into a tuple, filling in items
        # for unincluded axes with slice(None). This ensures it is
        # treated the same as tuple items.
        if isinstance(item, (numbers.Integral, slice)):
            item = [item] + [slice(None)] * n_uncommon_cube_dims
        else:
            # Item must therefore be tuple. Ensure it has an entry for each axis.
            item = list(item) + [slice(None)] * (n_cube_dims - len(item))
        # If common axis item is slice(None), result is trivial as common_axis is not changed.
        if item[common_axis] == slice(None):
            # Create item for slicing through the default API and slice.
            return self.seq[tuple([slice(None)] + item)]
        if isinstance(item[common_axis], numbers.Integral):
            # If common_axis item is an int or return an NDCube with dimensionality of N-1
            sequence_index, common_axis_index = \
                utils.sequence.cube_like_index_to_sequence_and_common_axis_indices(
                    item[common_axis], common_axis, common_axis_lengths)
            # Insert index for common axis in item for slicing the NDCube.
            cube_item = copy.deepcopy(item)
            cube_item[common_axis] = common_axis_index
            return self.seq.data[sequence_index][tuple(cube_item)]
        else:
            # item can now only be a tuple whose common axis item is a non-None slice object.
            # Convert item into iterable of SequenceItems and slice each cube appropriately.
            # item for common_axis must always be a slice for every cube,
            # even if it is only a length-1 slice.
            # Thus NDCubeSequence.index_as_cube can only slice away common axis if
            # item is int or item's first item is an int.
            # i.e. NDCubeSequence.index_as_cube cannot cause common_axis to become None
            # since in all cases where the common_axis is sliced away involve an NDCube
            # is returned, not an NDCubeSequence.
            # common_axis of returned sequence must be altered if axes in front of it
            # are sliced away.
            sequence_items = utils.sequence.cube_like_tuple_item_to_sequence_items(
                item, common_axis, common_axis_lengths, n_cube_dims)
            # Work out new common axis value if axes in front of it are sliced away.
            new_common_axis = common_axis - sum([isinstance(i, numbers.Integral)
                                                 for i in item[:common_axis]])
            # Copy sequence and alter the data and common axis.
            result = type(self.seq)([], meta=self.seq.meta, common_axis=new_common_axis)
            result.data = [self.seq.data[sequence_item.sequence_index][sequence_item.cube_item]
                           for sequence_item in sequence_items]
            return result
