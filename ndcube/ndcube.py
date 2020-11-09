
import abc
import numbers
import textwrap
import warnings
from collections import namedtuple

import astropy.nddata
import astropy.units as u
import numpy as np
import sunpy.coordinates
from astropy.utils.misc import InheritDocstrings
from astropy.wcs.wcsapi import BaseHighLevelWCS, BaseLowLevelWCS, HighLevelWCSWrapper, SlicedLowLevelWCS
from astropy.wcs.wcsapi.fitswcs import SlicedFITSWCS, custom_ctype_to_ucd_mapping

import ndcube.utils.wcs as wcs_utils
from ndcube import utils
from ndcube.mixins import NDCubePlotMixin, NDCubeSlicingMixin
from ndcube.ndcube_sequence import NDCubeSequence

__all__ = ['NDCubeABC', 'NDCubeBase', 'NDCube']


class NDCubeMetaClass(abc.ABCMeta):
    """
    A metaclass that combines `abc.ABCMeta`.
    """


class NDCubeABC(astropy.nddata.NDData, metaclass=NDCubeMetaClass):

    @abc.abstractproperty
    def dimensions(self):
        """
        The pixel dimensions of the cube.
        """

    @abc.abstractmethod
    def crop(self, *intervals, wcs=None):
        """
        Crops an NDCube given lower and upper real world bounds for each real world axis.

        Parameters
        ----------
        intervals: iterable whose elements are high level astropy objects or None
            An iterable of length-2 astropy higher level objects, e.g. SkyCoord,
            each represents the lower and upper bounds of a real world axis/axes.
            These are input to `astropy.wcs.WCS.world_to_array_index
            so their number and order must be compatible with the API of that method.
            Alternatively, None, can be provided instead of a higher level object.
            In this case, a high level object will be derived that causes the relevant
            axes to remain uncropped.

        wcs: `astropy.wcs.WCS`
            The WCS object to used to convert the world values to array indices.
            Although technically this can be any valid WCS, it will typically be
            self.wcs, self.extra_coords.wcs, or self.combined_wcs, combing both
            the WCS and extra coords.
            Default=self.wcs

        Returns
        -------
        result: `ndcube.NDCube`

        """

    @abc.abstractmethod
    def crop_by_values(self, *intervals, wcs=None):
        """
        Crops an NDCube given lower and upper real world bounds for each real world axis.

        Parameters
        ----------
        intervals: iterable of `astropy.units.Quantity`
            An iterable of length-2 `~astropy.units.Quantity` where each represents
            the lower and upper bounds of a real world axis, respectively.
            The number of quantities must equal the number of world dimensions,
            `~astropy.wcs.WCS.world_n_dim`, in the `~astropy.wcs.WCS` being used and
            must be provided in the same order as the
            `~astropy.wcs.world_axis_physical_types` property.
            Alternatively, None can be provided instead of a Quantity.
            In this case, a Quantity will be derived that causes the relevant
            axis to remain uncropped.

        wcs: `astropy.wcs.WCS`
            The WCS object to used to convert the world values to array indices.
            Although technically this can be any valid WCS, it will typically be
            self.wcs, self.extra_coords.wcs, or self.combined_wcs, combing both
            the WCS and extra coords.
            Default=self.wcs

        Returns
        -------
        result: `ndcube.NDCube`

        """


class NDCubeBase(NDCubeSlicingMixin, NDCubeABC):
    """
    Class representing N dimensional cubes. Extra arguments are passed on to
    `~astropy.nddata.NDData`.

    Parameters
    ----------
    data: `numpy.ndarray`
        The array holding the actual data in this object.

    wcs: `astropy.wcs.wcsapi.BaseLowLevelWCS`, `astropy.wcs.wcsapi.BaseHighLevelWCS`, optional
        The WCS object containing the axes' information, optional only if
        ``data`` is an `astropy.nddata.NDData` object.

    uncertainty : any type, optional
        Uncertainty in the dataset. Should have an attribute uncertainty_type
        that defines what kind of uncertainty is stored, for example "std"
        for standard deviation or "var" for variance. A metaclass defining
        such an interface is NDUncertainty - but isn’t mandatory. If the uncertainty
        has no such attribute the uncertainty is stored as UnknownUncertainty.
        Defaults to None.

    mask : any type, optional
        Mask for the dataset. Masks should follow the numpy convention
        that valid data points are marked by False and invalid ones with True.
        Defaults to None.

    meta : dict-like object, optional
        Additional meta information about the dataset. If no meta is provided
        an empty collections.OrderedDict is created. Default is None.

    unit : Unit-like or str, optional
        Unit for the dataset. Strings that can be converted to a Unit are allowed.
        Default is None.

    extra_coords : iterable of `tuple`, each with three entries
        (`str`, `int`, `astropy.units.quantity` or array-like)
        Gives the name, axis of data, and values of coordinates of a data axis not
        included in the WCS object.

    copy : bool, optional
        Indicates whether to save the arguments as copy. True copies every attribute
        before saving it while False tries to save every parameter as reference.
        Note however that it is not always possible to save the input as reference.
        Default is False.

    """

    def __init__(self, data, wcs=None, uncertainty=None, mask=None, meta=None,
                 unit=None, extra_coords=None, copy=False, **kwargs):

        super().__init__(data, wcs=wcs, uncertainty=uncertainty, mask=mask,
                         meta=meta, unit=unit, copy=copy, **kwargs)

        # Enforce that the WCS object is a low_level_wcs object, and not None.
        if self.wcs is None:
            raise TypeError("The WCS argument can not be None.")

        # Format extra coords.
        if extra_coords:
            self._extra_coords_wcs_axis = \
                utils.cube._format_input_extra_coords_to_extra_coords_wcs_axis(
                    extra_coords, wcs_utils._pixel_keep(wcs), wcs.pixel_n_dim, data.shape)
        else:
            self._extra_coords_wcs_axis = None

    @property
    def dimensions(self):
        """
        Returns a named tuple with two attributes: 'shape' gives the shape of
        the data dimensions; 'axis_types' gives the WCS axis type of each
        dimension, e.g. WAVE or HPLT-TAN for wavelength of helioprojected
        latitude.
        """
        return u.Quantity(self.data.shape, unit=u.pix)

    @property
    def array_axis_physical_types(self):
        """
        Returns the WCS physical types associated with each array axis.

        Returns an iterable of tuples where each tuple corresponds to an array axis and
        holds strings denoting the WCS physical types associated with that array axis.
        Since multiple physical types can be associated with one array axis, tuples can
        be of different lengths. Likewise, as a single physical type can correspond to
        multiple array axes, the same physical type string can appear in multiple tuples.
        """
        world_axis_physical_types = np.array(self.wcs.world_axis_physical_types)
        axis_correlation_matrix = self.wcs.axis_correlation_matrix
        return [tuple(world_axis_physical_types[axis_correlation_matrix[:, i]])
                for i in range(axis_correlation_matrix.shape[1])][::-1]

    def axis_world_coords(self, *axes, edges=False):
        """
        Returns WCS coordinate values of all pixels for all axes.

        Parameters
        ----------
        axes: `int` or `str`, or multiple `int` or `str`
            Axis number in numpy ordering or unique substring of
            `~ndcube.NDCube.world_axis_physical_types`
            of axes for which real world coordinates are desired.
            axes=None implies all axes will be returned.

        edges: `bool`
            The edges argument helps in returning `pixel_edges`
            instead of `pixel_values`. Default value is False,
            which returns `pixel_values`. True return `pixel_edges`

        Returns
        -------
        axes_coords: `list`
            High level object giving the real world coords for the axes requested by user.
            For example, SkyCoords.

        Example
        -------
        >>> NDCube.all_world_coords(('lat', 'lon')) # doctest: +SKIP
        >>> NDCube.all_world_coords(2) # doctest: +SKIP

        """
        raise NotImplementedError()

    def axis_world_coords_values(self, *axes, edges=False):
        """
        Returns WCS coordinate values of all pixels for desired axes.

        Parameters
        ----------
        axes: `int` or `str`, or multiple `int` or `str`
            Axis number in numpy ordering or unique substring of
            `~ndcube.NDCube.wcs.world_axis_physical_types`
            of axes for which real world coordinates are desired.
            axes=None implies all axes will be returned.

        edges: `bool`
            If True, the coords at the edges of the pixels are returned
            rather than the coords at the center of the pixels.
            Note that there are n+1 edges for n pixels which is reflected
            in the returned coords.
            Default=False, i.e. pixel centers are returned.

        Returns
        -------
        coord_values: `collections.namedtuple`
            Real world coords labeled with their real world physical types
            for the axes requested by the user.
            Returned in same order as axis_names.

        Example
        -------
        >>> NDCube.all_world_coords_values(('lat', 'lon')) # doctest: +SKIP
        >>> NDCube.all_world_coords_values(2) # doctest: +SKIP

        """
        wcs = self.wcs
        if not isinstance(wcs, BaseLowLevelWCS):
            wcs = wcs.low_level_wcs
        # Create meshgrid of all pixel coordinates.
        # If user, wants edges, set pixel values to pixel edges.
        # Else make pixel centers.
        wcs_shape = self.data.shape[::-1]
        if edges:
            wcs_shape = tuple(np.array(wcs_shape) + 1)
            pixel_inputs = np.meshgrid(*[np.arange(i) - 0.5 for i in wcs_shape],
                                       indexing='ij', sparse=True)
        else:
            pixel_inputs = np.meshgrid(*[np.arange(i) for i in wcs_shape],
                                       indexing='ij', sparse=True)

        # Get world coords for all axes and all pixels.
        axes_coords = wcs.pixel_to_world_values(*pixel_inputs)
        if wcs.world_n_dim == 1:
            axes_coords = [axes_coords]
        # Ensure it's a list not a tuple
        axes_coords = list(axes_coords)

        # Reduce duplication across independent dimensions for each coord
        # and transpose to make dimensions mimic numpy array order rather than WCS order.
        for i, axis_coord in enumerate(axes_coords):
            slices = np.array([slice(None)] * wcs.pixel_n_dim)
            slices[np.invert(wcs.axis_correlation_matrix[i])] = 0
            axes_coords[i] = axis_coord[tuple(slices)].T * u.Unit(wcs.world_axis_units[i])

        world_axis_physical_types = wcs.world_axis_physical_types
        # If user has supplied axes, extract only the
        # world coords that correspond to those axes.
        if axes:
            # Convert input axes to WCS world axis indices.
            world_indices = set()
            for axis in axes:
                if isinstance(axis, numbers.Integral):
                    # If axis is int, it is a numpy order array axis.
                    # Convert to pixel axis in WCS order.
                    axis = wcs_utils.convert_between_array_and_pixel_axes(
                            np.array([axis]), wcs.pixel_n_dim)[0]
                    # Get WCS world axis indices that correspond to the WCS pixel axis
                    # and add to list of indices of WCS world axes whose coords will be returned.
                    world_indices.update(wcs_utils.pixel_axis_to_world_axes(
                        axis, wcs.axis_correlation_matrix))
                elif isinstance(axis, str):
                    # If axis is str, it is a physical type or substring of a physical type.
                    world_indices.update({wcs_utils.physical_type_to_world_axis(
                        axis, world_axis_physical_types)})
                else:
                    raise TypeError(f"Unrecognized axis type: {axis, type(axis)}. "
                                    "Must be of type (numbers.Integral, str)")
            # Use inferred world axes to extract the desired coord value
            # and corresponding physical types.
            world_indices = np.array(list(world_indices), dtype=int)
            axes_coords = np.array(axes_coords)[world_indices]
            world_axis_physical_types = tuple(np.array(world_axis_physical_types)[world_indices])

        # Return in array order.
        # First replace characters in physical types forbidden for namedtuple identifiers.
        identifiers = []
        for physical_type in world_axis_physical_types[::-1]:
            identifier = physical_type.replace(":", "_")
            identifier = identifier.replace(".", "_")
            identifier = identifier.replace("-", "__")
            identifiers.append(identifier)
        CoordValues = namedtuple("CoordValues", identifiers)
        return CoordValues(*axes_coords[::-1])

    @property
    def extra_coords(self):
        """
        Dictionary of extra coords where each key is the name of an extra
        coordinate supplied by user during instantiation of the NDCube.

        The value of each key is itself a dictionary with the following
        keys:   | 'axis': `int`   |     The number of the data axis to
        which the extra coordinate corresponds.   | 'value':
        `astropy.units.Quantity` or array-like   |     The value of the
        extra coordinate at each pixel/array element along the   |
        corresponding axis (given by the 'axis' key, above).  Note this
        means   |     that the length of 'value' must be equal to the
        length of the data axis   |     to which is corresponds.
        """

        if not self._extra_coords_wcs_axis:
            result = None
        else:
            result = {}
            for key in list(self._extra_coords_wcs_axis.keys()):
                result[key] = {
                    "axis": utils.cube.wcs_axis_to_data_ape14(
                        self._extra_coords_wcs_axis[key]["wcs axis"],
                        wcs_utils._pixel_keep(self.wcs),
                        self.wcs.low_level_wcs.pixel_n_dim),
                    "value": self._extra_coords_wcs_axis[key]["value"]}
        return result

    def crop(self, *intervals, wcs=None):
        # The docstring is defined in NDCubeBase
        return self._crop(*intervals, wcs=wcs, crop_by_values=False)

    def crop_by_values(self, *intervals, wcs=None):
        # The docstring is defined in NDCubeBase
        return self._crop(*intervals, wcs=wcs, crop_by_values=True)

    def _crop(self, *intervals, wcs=None, crop_by_values=False):
        # If no intervals provided, return NDCube without slicing.
        none_intervals = np.array([interval is None for interval in intervals])
        if none_intervals.all():
            return self
        input_intervals = tuple(intervals)
        intervals = list(intervals)
        # Set default wcs.
        if wcs is None:
            wcs = self.wcs
        # Define functions to be used in converting between array indices and world coords
        # based in input kwarg.
        if crop_by_values:
            try:
                world_to_array_index = wcs.world_to_array_index_values
                array_index_to_world = wcs.array_index_to_world_values
            except AttributeError:
                world_to_array_index = wcs.low_level_wcs.world_to_array_index_values
                array_index_to_world = wcs.low_level_wcs.array_index_to_world_values
        else:
            world_to_array_index = wcs.world_to_array_index
            array_index_to_world = wcs.array_index_to_world
        world_axis_units = wcs.world_axis_units
        world_axis_physical_types = wcs.world_axis_physical_types
        # If user did not provide all intervals,
        # calculate missing intervals based on whole cube range along those axes.
        if none_intervals.any():
            # Calculate intervals for first and last index for all axes.
            array_intervals = [[0, np.round(d.value - 1).astype(int)] for d in self.dimensions]
            intervals = list(array_index_to_world(*array_intervals))
            # Overwrite intervals with user-supplied ones, if provided.
            for i, interval_is_none in enumerate(none_intervals):
                if not interval_is_none:
                    intervals[i] = input_intervals[i]
        # Convert intervals to array indices.
        intervals_indices = world_to_array_index(*intervals)
        # Ensure return type is tuple of lists, even if only one axis returned.
        if not isinstance(intervals_indices, tuple):
            intervals_indices = (intervals_indices,)
        # Construct item which which to slice NDCube.
        item = []
        for i, indices in enumerate(intervals_indices):
            # If upper limit index less than zero,
            # then interval does not overlap with cube range. Raise error.
            if indices[-1] < 0:
                physical_type = world_axis_physical_types[::-1][i]
                cube_range = self.axis_world_coords_values(physical_type)
                raise IndexError("Input real world interval beyond range of NDCube. "
                                 f"Physical type: {physical_type}; "
                                 f"Input interval: {intervals[::-1][i]}; "
                                 f"NDCube range: ({cube_range[0]}, {cube_range[-1]})")
            # Construct slice for this axis and append to item.
            # Increment upper idex by 1 to ensure the upper world coord
            # is included in sliced cube.
            item.append(slice(max(0, indices[0]), indices[-1] + 1))
        return self[tuple(item)]

    def __str__(self):
        return textwrap.dedent(f"""\
                NDCube
                ------
                Dimensions: {self.dimensions}
                Physical Types of Axes: {self.array_axis_physical_types}""")

    def __repr__(self):
        return f"{object.__repr__(self)}\n{str(self)}"

    def explode_along_axis(self, axis):
        """
        Separates slices of NDCubes along a given cube axis into a
        NDCubeSequence of (N-1)DCubes.

        Parameters
        ----------
        axis : `int`
            The axis along which the data is to be changed.

        Returns
        -------
        result : `ndcube_sequence.NDCubeSequence`
        """
        # If axis is -ve then calculate the axis from the length of the dimensions of one cube
        if axis < 0:
            axis = len(self.dimensions) + axis
        # To store the resultant cube
        result_cubes = []
        # All slices are initially initialised as slice(None, None, None)
        cube_slices = [slice(None, None, None)] * self.data.ndim
        # Slicing the cube inside result_cube
        for i in range(self.data.shape[axis]):
            # Setting the slice value to the index so that the slices are done correctly.
            cube_slices[axis] = i
            # Set to None the metadata of sliced cubes.
            item = tuple(cube_slices)
            sliced_cube = self[item]
            sliced_cube.meta = None
            # Appending the sliced cubes in the result_cube list
            result_cubes.append(sliced_cube)
        # Creating a new NDCubeSequence with the result_cubes and common axis as axis
        return NDCubeSequence(result_cubes, common_axis=axis, meta=self.meta)


class NDCube(NDCubeBase, NDCubePlotMixin, astropy.nddata.NDArithmeticMixin):
    pass
