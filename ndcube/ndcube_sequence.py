import numpy as np

import astropy.units as u
import sunpy.map
from sunpy.map import MapCube

from ndcube import utils
from ndcube import SequenceDimensionPair
from ndcube.visualization import animation as ani

__all__ = ['NDCubeSequence']


class NDCubeSequence:
    """
    Class representing list of cubes.

    Parameters
    ----------
    data_list : `list`
        List of cubes.

    meta : `dict` or None
        The header of the NDCubeSequence.

    common_axis: `int` or None
        The data axis which is common between the NDCubeSequence and the Cubes within.
        For example, if the Cubes are sequenced in chronological order and time is
        one of the zeroth axis of each Cube, then common_axis should be se to 0.
        This enables the option for the NDCubeSequence to be indexed as though it is
        one single Cube.
    """

    def __init__(self, data_list, meta=None, common_axis=None, **kwargs):
        self.data = data_list
        self.meta = meta
        self._common_axis = common_axis

    def __getitem__(self, item):
        if item is None or (isinstance(item, tuple) and None in item):
            raise IndexError("None indices not supported")
        # Convert item to list of SequenceSlices
        sequence_items = utils.sequence.convert_item_to_sequence_items(item, len(self.data))
        return utils.sequence._slice_sequence(self, sequence_items)

    def plot(self, *args, **kwargs):
        i = ani.ImageAnimatorNDCubeSequence(self, *args, **kwargs)
        return i

    def to_sunpy(self, *args, **kwargs):
        result = None
        if all(isinstance(instance_sequence, sunpy.map.mapbase.GenericMap)
               for instance_sequence in self.data):
            result = MapCube(self.data, *args, **kwargs)
        else:
            raise NotImplementedError("Sequence type not Implemented")
        return result

    def explode_along_axis(self, axis):
        """
        Separates slices of NDCubes in sequence along a given cube axis into (N-1)DCubes.

        Parameters
        ----------

        axis : `int`
            The axis along which the data is to be changed.
        """
        # if axis is None then set axis as common axis.
        if self._common_axis is not None:
            if self._common_axis != axis:
                raise ValueError("axis and common_axis should be equal.")
        # is axis is -ve then calculate the axis from the length of the dimensions of one cube
        if axis < 0:
            axis = len(self.dimensions.shape[1::]) + axis
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
        # creating a new sequence with the result_cubes keeping the meta and common axis as axis
        return self._new_instance(result_cubes, meta=self.meta, common_axis=axis)

    def __repr__(self):
        return (
            """Sunpy NDCubeSequence
---------------------
Length of NDCubeSequence:  {length}
Shape of 1st NDCube: {shapeNDCube}
Axis Types of 1st NDCube: {axis_type}
""".format(length=self.dimensions.shape[0], shapeNDCube=self.dimensions.shape[1::],
                axis_type=self.dimensions.axis_types[1::]))

    @property
    def dimensions(self):
        return SequenceDimensionPair(
            shape=tuple([len(self.data)]+list(self.data[0].dimensions.shape)),
            axis_types=tuple(["Sequence Axis"]+self.data[0].dimensions.axis_types))

    @property
    def cube_like_dimensions(self):
        if type(self._common_axis) is not int:
            raise TypeError("Common axis must be set.")
        dimensions = self.dimensions
        shape_list_one_cube = list(dimensions.shape[1:])
        shape_list_one_cube[self._common_axis] = (dimensions.shape[0] *
                                                  shape_list_one_cube[self._common_axis])
        return SequenceDimensionPair(shape=tuple(shape_list_one_cube),
                                     axis_types=dimensions.axis_types[1:])

    @property
    def common_axis_extra_coords(self):
        if self._common_axis in range(self.data[0].wcs.naxis):
            common_extra_coords = {}
            coord_names = list(self.data[0].extra_coords.keys())
            for coord_name in coord_names:
                if self.data[0].extra_coords[coord_name]["axis"] == self._common_axis:
                    try:
                        coord_unit = self.data[0].extra_coords[coord_name]["value"].unit
                        qs = tuple([np.asarray(
                            c.extra_coords[coord_name]["value"].to(coord_unit).value)
                                    for c in self.data])
                        common_extra_coords[coord_name] = u.Quantity(np.concatenate(qs),
                                                                     unit=coord_unit)
                    except AttributeError:
                        qs = tuple([np.asarray(c.extra_coords[coord_name]["value"])
                                    for c in self.data])
                        common_extra_coords[coord_name] = np.concatenate(qs)
        else:
            common_extra_coords = None
        return common_extra_coords

    @property
    def new_common_axis_extra_coords(self):
        if not isinstance(self._common_axis, int):
            raise ValueError("Common axis is not set.")
        common_axis_extra_coords = self._get_extra_coords_by_axis(self._common_axis)
        for coord_name in list(common_axis_extra_coords.keys()):
            common_axis_extra_coords[coord_name] = common_axis_extra_coords[coord_name].flatten()
        return common_axis_extra_coords

    @property
    def sequence_axis_extra_coords(self):
        return self._get_extra_coords_by_axis(None)

    def _get_extra_coords_by_axis(self, axis):
        """
        Return subset of extra coords corresponding to a given axis.

        Parameters
        ----------
        axis: `int` or `None`
            Number of cube data axis of interest.  For extra coords not
            corresponding to a cubes' data axis, set to None.

        Returns
        -------
        axis_extra_coords: `dict`
            Dictionary holding values of extra coords corresponding to input axis.
            Keys are coord names.

        """
        # Identify all extra coord names assigned to input cube data axis.
        axis_coord_names = []
        for cube in self.data:
            all_extra_coords = cube.extra_coords
            all_extra_coords_keys = list(all_extra_coords.keys())
            for coord_key in all_extra_coords_keys:
                if all_extra_coords[coord_key]["axis"] == axis:
                    axis_coord_names.append(coord_key)
        axis_coord_names = set(axis_coord_names)
        # Define empty dictionary which will hold the extra coord
        # values not assigned a cube data axis.
        axis_extra_coords = {}
        # Define list of None signifying unit of each coord.  It will
        # be filled in in for loop below.
        axis_coord_units = [None]*len(axis_coord_names)
        # Iterate through cubes and populate values of each extra coord
        # not assigned a cube data axis.
        cube_extra_coords = [cube.extra_coords for cube in self.data]
        for i, coord_key in enumerate(axis_coord_names):
            coord_values = np.array([None]*len(self.data), dtype=object)
            for j, cube in enumerate(self.data):
                # Construct list of coord values from each cube for given extra coord.
                try:
                    coord_values[j] = cube_extra_coords[j][coord_key]["value"]
                    # Determine whether extra coord is a quantity by checking
                    # whether any one value has a unit. As we are not
                    # assuming that all cubes have the same extra coords
                    # along the sequence axis, we will keep checking as we
                    # move through the cubes until all cubes are checked or
                    # we have found a unit.
                    if not axis_coord_units[i]:
                        if isinstance(cube_extra_coords[j][coord_key]["value"], u.Quantity):
                            axis_coord_units[i] = cube_extra_coords[j][coord_key]["value"].unit
                except KeyError:
                    pass
            # If the extra coord is normally a Quantity, replace all
            # None occurrences in coord value array with a NaN, and
            # convert coord_values from an array of Quantities to a
            # single Quantity of length equal to number of cubes in
            # sequence.
            w_none = np.where(coord_values == None)[0]
            if axis_coord_units[i]:
                # This part of if statement is coded in an apparently
                # round about way but necessitated because you can't
                # put a NaN quantity into an array and keep its unit.
                w_not_none = np.where(coord_values != None)[0]
                coord_values = u.Quantity(list(coord_values[w_not_none]),
                                          unit=axis_coord_units[i])
                coord_values = list(coord_values.value)
                for index in w_none:
                    coord_values.insert(index, np.nan)
                coord_values = u.Quantity(coord_values, unit=axis_coord_units[i])
            else:
                coord_values[w_none] = np.nan
            # Enter sequence extra coord into dictionary
            axis_extra_coords[coord_key] = coord_values
        return axis_extra_coords

    @classmethod
    def _new_instance(cls, data_list, meta=None, common_axis=None):
        """
        Instantiate a new instance of this class using given data.
        """
        return cls(data_list, meta=meta, common_axis=common_axis)

    @property
    def index_as_cube(self):
        """
        Method to slice the NDCubesequence instance as a single cube

        Example
        -------
        >>> # Say we have three Cubes each cube has common_axis=0 is time and shape=(3,3,3)
        >>> data_list = [cubeA, cubeB, cubeC] # doctest: +SKIP
        >>> cs = NDCubeSequence(data_list, meta=None, common_axis=0) # doctest: +SKIP
        >>> # return zeroth time slice of cubeB in via normal NDCubeSequence indexing.
        >>> cs[1,:,0,:] # doctest: +SKIP
        >>> # Return same slice using this function
        >>> cs.index_sequence_as_cube[3:6, 0, :] # doctest: +SKIP
        """
        if self._common_axis is None:
            raise ValueError("common_axis cannot be None")
        return _IndexAsCubeSlicer(self)


"""
Cube Sequence Helpers
"""


class _IndexAsCubeSlicer:
    """
    Helper class to make slicing in index_as_cube sliceable/indexable
    like a numpy array.

    Parameters
    ----------
    seq : `ndcube.NDCubeSequence`
        Object of NDCubeSequence.

    """

    def __init__(self, seq):
        self.seq = seq

    def __getitem__(self, item):
        return utils.sequence._index_sequence_as_cube(self.seq, item)
