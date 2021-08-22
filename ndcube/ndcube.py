import abc
import textwrap
import warnings
import itertools
from copy import deepcopy
from collections import namedtuple
from collections.abc import Mapping

import astropy.nddata
import astropy.units as u
import numpy as np

try:
    # Import sunpy coordinates if available to register the frames and WCS functions with astropy
    import sunpy.coordinates  # pylint: disable=unused-import  # NOQA
except ImportError:
    pass
from astropy.wcs import WCS
from astropy.wcs.utils import _split_matrix
from astropy.wcs.wcsapi import BaseHighLevelWCS, HighLevelWCSWrapper

from ndcube import utils
from ndcube.extra_coords import ExtraCoords
from ndcube.global_coords import GlobalCoords
from ndcube.mixins import NDCubeSlicingMixin
from ndcube.ndcube_sequence import NDCubeSequence
from ndcube.utils.wcs_high_level_conversion import high_level_objects_to_values, values_to_high_level_objects
from ndcube.visualization import PlotterDescriptor
from ndcube.wcs.wrappers import CompoundLowLevelWCS

__all__ = ['NDCubeABC', 'NDCubeBase', 'NDCube']


class NDCubeABC(astropy.nddata.NDData, metaclass=abc.ABCMeta):

    @abc.abstractproperty
    def dimensions(self):
        """
        The array dimensions of the cube.
        """

    @abc.abstractmethod
    def crop(self, lower_corner, upper_corner, wcs=None):
        """
        Crop given world coordinate objects describing the lower and upper corners of a region.

        The region of interest is defined in pixel space, by converting the world
        coordinates of the corners to pixel coordinates and then cropping the
        smallest pixel region which contains the corners specified.
        This means that the edges of the world coordinate region specified by
        the coordinates are not guaranteed to be included in the cropped output.
        This is normally noticeable when cropping a celestial coordinate in a
        frame which differs from the native frame of the coordinates in the WCS.

        Parameters
        ----------
        lower_corner: iterable whose elements are None or high level astropy objects
            An iterable of length-1 astropy higher level objects, e.g. SkyCoord,
            representing the real world coordinates of the lower corner of
            the region of interest.
            These are input to `astropy.wcs.WCS.world_to_array_index`
            so their number and order must be compatible with the API of that method.
            Alternatively, None, can be provided instead of a higher level object.
            In this case, the corresponding array axes will be cropped starting from
            0th array index.

        upper_corner: iterable whose elements are None or high level astropy objects
            An iterable of length-1 astropy higher level objects, e.g. SkyCoord,
            representing the real world coordinates of the upper corner of
            the region of interest.
            These are input to `astropy.wcs.WCS.world_to_array_index`
            so their number and order must be compatible with the API of that method.
            Alternatively, None, can be provided instead of a higher level object.
            In this case, the corresponding array axes will be cropped to include
            the final array index.

        wcs: `astropy.wcs.wcsapi.BaseHighLevelWCS`
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
    def crop_by_values(self, lower_corner, upper_corner, units=None, wcs=None):
        """
        Crops an NDCube given lower and upper real world bounds for each real world axis.

        The region of interest is defined in pixel space, by converting the world
        coordinates of the corners to pixel coordinates and then cropping the
        smallest pixel region which contains the corners specified.
        This means that the edges of the world coordinate region specified by
        the coordinates are not guaranteed to be included in the cropped output.
        This is normally noticeable when cropping a celestial coordinate in a
        frame which differs from the native frame of the coordinates in the WCS.

        Parameters
        ----------
        lower_corner: iterable whose elements are None, `astropy.units.Quantity` or `float`
            An iterable of length-1 Quantities or floats, representing
            the real world coordinate values of the lower corner of
            the region of interest.
            These are input to `astropy.wcs.WCS.world_to_array_index_values`
            so their number and order must be compatible with the API of that method,
            i.e. they must be in world axis order.
            Alternatively, None, can be provided instead of a Quantity or float.
            In this case, the corresponding array axes will be cropped starting from
            0th array index.

        upper_corner: iterable whose elements are None, `astropy.units.Quantity` or `float`
            An iterable of length-1 Quantities or floats, representing
            the real world coordinate values of the upper corner of
            the region of interest.
            These are input to `astropy.wcs.WCS.world_to_array_index_values`
            so their number and order must be compatible with the API of that method,
            i.e. they must be in world axis order.
            Alternatively, None, can be provided instead of a Quantity or float.
            In this case, the corresponding array axes will be cropped to include
            the final array index.

        units: iterable of `astropy.units.Unit`
            The unit of the corresponding entries in lower_corner and upper_corner.
            Must therefore be the same length as lower_corner and upper_corner.
            Only used if the corresponding type is not a `astropy.units.Quantity`.

        wcs: `astropy.wcs.wcsapi.BaseLowLevelWCS`
            The WCS object to used to convert the world values to array indices.
            Although technically this can be any valid WCS, it will typically be
            self.wcs, self.extra_coords.wcs, or self.combined_wcs, combing both
            the WCS and extra coords.
            Default=self.wcs

        Returns
        -------
        result: `ndcube.NDCube`

        """


class NDCubeLinkedDescriptor:
    """
    A descriptor which gives the property a reference to the cube to which it is attached.
    """
    def __init__(self, default_type):
        self._default_type = default_type
        self._property_name = None

    def __set_name__(self, owner, name):
        """
        This function is called when the class the descriptor is attached to is initialized.

        The *class* and not the instance.
        """
        # property name is the name of the attribute on the parent class
        # pointing at an instance of this descriptor.
        self._property_name = name
        # attribute name is the name of the attribute on the parent class where
        # the data is stored.
        self._attribute_name = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return

        if getattr(obj, self._attribute_name, None) is None and self._default_type is not None:
            self.__set__(obj, self._default_type)

        return getattr(obj, self._attribute_name)

    def __set__(self, obj, value):
        if isinstance(value, self._default_type):
            value._ndcube = obj
        elif issubclass(value, self._default_type):
            value = value(obj)
        else:
            raise ValueError(
                f"Unable to set value for {self._property_name} it should "
                f"be an instance or subclass of {self._default_type}")

        setattr(obj, self._attribute_name, value)


class NDCubeBase(NDCubeSlicingMixin, NDCubeABC):
    """
    Class representing N-D data described by a single array and set of WCS transformations.

    Parameters
    ----------
    data: array-like or `astropy.nddata.NDData`
        The array holding the actual data in this object.

    wcs: `astropy.wcs.wcsapi.BaseLowLevelWCS`, `astropy.wcs.wcsapi.BaseHighLevelWCS`, optional
        The WCS object containing the axes' information, optional only if
        ``data`` is an `astropy.nddata.NDData` object.

    uncertainty : any type, optional
        Uncertainty in the dataset. Should have an attribute uncertainty_type
        that defines what kind of uncertainty is stored, for example "std"
        for standard deviation or "var" for variance. A metaclass defining such
        an interface is `~astropy.nddata.NDUncertainty` - but isn’t mandatory.
        If the uncertainty has no such attribute the uncertainty is stored as
        `~astropy.nddata.UnknownUncertainty`.
        Defaults to None.

    mask : any type, optional
        Mask for the dataset. Masks should follow the numpy convention
        that valid data points are marked by `False` and invalid ones with `True`.
        Defaults to `None`.

    meta : dict-like object, optional
        Additional meta information about the dataset. If no meta is provided
        an empty dictionary is created.

    unit : Unit-like or `str`, optional
        Unit for the dataset. Strings that can be converted to a `~astropy.unit.Unit` are allowed.
        Default is `None` which results in dimensionless units.

    copy : bool, optional
        Indicates whether to save the arguments as copy. `True` copies every attribute
        before saving it while `False` tries to save every parameter as reference.
        Note however that it is not always possible to save the input as reference.
        Default is `False`.

    """
    # Instances of Extra and Global coords are managed through descriptors
    _extra_coords = NDCubeLinkedDescriptor(ExtraCoords)
    _global_coords = NDCubeLinkedDescriptor(GlobalCoords)

    def __init__(self, data, wcs=None, uncertainty=None, mask=None, meta=None,
                 unit=None, copy=False, **kwargs):

        super().__init__(data, uncertainty=uncertainty, mask=mask,
                         meta=meta, unit=unit, copy=copy, **kwargs)
        if not self.wcs:
            self.wcs = wcs  # This line is required as a patch for an astropy bug.
        # Above line is in if statement to prevent WCS being overwritten with None
        # if we are instantiating from an NDCube.

        # Enforce that the WCS object is not None
        if self.wcs is None:
            raise TypeError("The WCS argument can not be None.")

        # Get existing extra_coords if initializing from an NDCube
        if hasattr(data, "extra_coords"):
            extra_coords = data.extra_coords
            if copy:
                extra_coords = deepcopy(extra_coords)
            self._extra_coords = extra_coords

        # Get existing global_coords if initializing from an NDCube
        if hasattr(data, "global_coords"):
            global_coords = data._global_coords
            if copy:
                global_coords = deepcopy(global_coords)
            self._global_coords = global_coords

    @property
    def extra_coords(self):
        """
        An `.ExtraCoords` object holding extra coordinates aligned to array axes.
        """
        return self._extra_coords

    @property
    def global_coords(self):
        """
        A `.GlobalCoords` object holding coordinate metadata not aligned to an array axis.
        """
        return self._global_coords

    @property
    def combined_wcs(self):
        """
        A `~astropy.wcs.wcsapi.BaseHighLevelWCS` object which combines ``.wcs`` with ``.extra_coords``.
        """
        if not self.extra_coords.wcs:
            return self.wcs

        mapping = list(range(self.wcs.pixel_n_dim)) + list(self.extra_coords.mapping)
        return HighLevelWCSWrapper(
            CompoundLowLevelWCS(self.wcs.low_level_wcs, self._extra_coords.wcs, mapping=mapping)
        )

    @property
    def dimensions(self):
        return u.Quantity(self.data.shape, unit=u.pix)

    @property
    def array_axis_physical_types(self):
        """
        Returns the physical types associated with each array axis.

        Returns an iterable of tuples where each tuple corresponds to an array axis and
        holds strings denoting the physical types associated with that array axis.
        Since multiple physical types can be associated with one array axis, tuples can
        be of different lengths. Likewise, as a single physical type can correspond to
        multiple array axes, the same physical type string can appear in multiple tuples.

        The physical types are drawn from the WCS ExtraCoords objects.
        """
        wcs = self.combined_wcs
        world_axis_physical_types = np.array(wcs.world_axis_physical_types)
        axis_correlation_matrix = wcs.axis_correlation_matrix
        return [tuple(world_axis_physical_types[axis_correlation_matrix[:, i]])
                for i in range(axis_correlation_matrix.shape[1])][::-1]

    def _generate_world_coords(self, pixel_corners, wcs):
        # TODO: We can improve this by not always generating all coordinates
        # To make our lives easier here we generate all the coordinates for all
        # pixels and then choose the ones we want to return to the user based
        # on the axes argument. We could be smarter by integrating this logic
        # into the main loop, this would potentially reduce the number of calls
        # to pixel_to_world_values

        # Create meshgrid of all pixel coordinates.
        # If user, wants pixel_corners, set pixel values to pixel pixel_corners.
        # Else make pixel centers.
        pixel_shape = self.data.shape[::-1]
        if pixel_corners:
            pixel_shape = tuple(np.array(pixel_shape) + 1)
            ranges = [np.arange(i) - 0.5 for i in pixel_shape]
        else:
            ranges = [np.arange(i) for i in pixel_shape]

        # Limit the pixel dimensions to the ones present in the ExtraCoords
        if isinstance(wcs, ExtraCoords):
            ranges = [ranges[i] for i in wcs.mapping]
            wcs = wcs.wcs

        world_coords = [None] * wcs.world_n_dim
        for (pixel_axes_indices, world_axes_indices) in _split_matrix(wcs.axis_correlation_matrix):
            # First construct a range of pixel indices for this set of coupled dimensions
            sub_range = [ranges[idx] for idx in pixel_axes_indices]
            # Then get a set of non correlated dimensions
            non_corr_axes = set(list(range(wcs.pixel_n_dim))) - set(pixel_axes_indices)
            # And inject 0s for those coordinates
            for idx in non_corr_axes:
                sub_range.insert(idx, 0)
            # Generate a grid of broadcastable pixel indices for all pixel dimensions
            grid = np.meshgrid(*sub_range, indexing='ij')
            # Convert to world coordinates
            world = wcs.pixel_to_world_values(*grid)
            # TODO: this isinstance check is to mitigate https://github.com/spacetelescope/gwcs/pull/332
            if wcs.world_n_dim == 1 and not isinstance(world, tuple):
                world = [world]
            # Extract the world coordinates of interest and remove any non-correlated axes
            # Transpose the world coordinates so they match array ordering not pixel
            for idx in world_axes_indices:
                array_slice = np.zeros((wcs.pixel_n_dim,), dtype=object)
                array_slice[wcs.axis_correlation_matrix[idx]] = slice(None)
                tmp_world = world[idx][tuple(array_slice)].T
                world_coords[idx] = tmp_world

        for i, (coord, unit) in enumerate(zip(world_coords, wcs.world_axis_units)):
            world_coords[i] = coord << u.Unit(unit)

        return world_coords

    @utils.misc.sanitise_wcs
    def axis_world_coords(self, *axes, pixel_corners=False, wcs=None):
        """
        Returns WCS coordinate values of all pixels for all axes.

        Parameters
        ----------
        axes: `int` or `str`, or multiple `int` or `str`, optional
            Axis number in numpy ordering or unique substring of
            `~ndcube.NDCube.world_axis_physical_types`
            of axes for which real world coordinates are desired.
            axes=None implies all axes will be returned.

        pixel_corners: `bool`, optional
            If `True` then instead of returning the coordinates at the centers of the pixels,
            the coordinates at the pixel corners will be returned. This
            increases the size of the output by 1 in all dimensions as all corners are returned.

        wcs: `astropy.wcs.wcsapi.BaseHighLevelWCS`, optional
            The WCS object to used to calculate the world coordinates.
            Although technically this can be any valid WCS, it will typically be
            ``self.wcs``, ``self.extra_coords``, or ``self.combined_wcs`` which combines both
            the WCS and extra coords.
            Defaults to the ``.wcs`` property.

        Returns
        -------
        axes_coords: `list`
            An iterable of "high level" objects giving the real world
            coords for the axes requested by user.
            For example, a tuple of `~astropy.coordinates.SkyCoord` objects.
            The types returned are determined by the WCS object.
            The dimensionality of these objects should match that of
            their corresponding array dimensions, unless ``pixel_corners=True``
            in which case the length along each axis will be 1 greater than the number of pixels.

        Example
        -------
        >>> NDCube.all_world_coords(('lat', 'lon')) # doctest: +SKIP
        >>> NDCube.all_world_coords(2) # doctest: +SKIP

        """
        if isinstance(wcs, BaseHighLevelWCS):
            wcs = wcs.low_level_wcs

        axes_coords = self._generate_world_coords(pixel_corners, wcs)

        if isinstance(wcs, ExtraCoords):
            wcs = wcs.wcs

        axes_coords = values_to_high_level_objects(*axes_coords, low_level_wcs=wcs)

        if not axes:
            return tuple(axes_coords)

        object_names = np.array([wao_comp[0] for wao_comp in wcs.world_axis_object_components])
        unique_obj_names = utils.misc.unique_sorted(object_names)
        world_axes_for_obj = [np.where(object_names == name)[0] for name in unique_obj_names]

        # Create a mapping from world index in the WCS to object index in axes_coords
        world_index_to_object_index = {}
        for object_index, world_axes in enumerate(world_axes_for_obj):
            for world_index in world_axes:
                world_index_to_object_index[world_index] = object_index

        world_indices = utils.wcs.calculate_world_indices_from_axes(wcs, axes)
        object_indices = utils.misc.unique_sorted(
            [world_index_to_object_index[world_index] for world_index in world_indices]
        )

        return tuple(axes_coords[i] for i in object_indices)

    @utils.misc.sanitise_wcs
    def axis_world_coords_values(self, *axes, pixel_corners=False, wcs=None):
        """
        Returns WCS coordinate values of all pixels for desired axes.

        Parameters
        ----------
        axes: `int` or `str`, or multiple `int` or `str`, optional
            Axis number in numpy ordering or unique substring of
            `~ndcube.NDCube.wcs.world_axis_physical_types`
            of axes for which real world coordinates are desired.
            axes=None implies all axes will be returned.

        pixel_corners: `bool`, optional
            If `True` then instead of returning the coordinates of the pixel
            centers the coordinates of the pixel corners will be returned.  This
            increases the size of the output along each dimension by 1 as all corners are returned.

        wcs: `astropy.wcs.wcsapi.BaseHighLevelWCS`, optional
            The WCS object to used to calculate the world coordinates.
            Although technically this can be any valid WCS, it will typically be
            ``self.wcs``, ``self.extra_coords``, or ``self.combined_wcs``, combing both
            the WCS and extra coords.
            Defaults to the ``.wcs`` property.

        Returns
        -------
        axes_coords: `list`
            An iterable of "high level" objects giving the real world
            coords for the axes requested by user.
            For example, a tuple of `~astropy.coordinates.SkyCoord` objects.
            The types returned are determined by the WCS object.
            The dimensionality of these objects should match that of
            their corresponding array dimensions, unless ``pixel_corners=True``
            in which case the length along each axis will be 1 greater than the number of pixels.

        Example
        -------
        >>> NDCube.all_world_coords_values(('lat', 'lon')) # doctest: +SKIP
        >>> NDCube.all_world_coords_values(2) # doctest: +SKIP

        """
        if isinstance(wcs, BaseHighLevelWCS):
            wcs = wcs.low_level_wcs

        axes_coords = self._generate_world_coords(pixel_corners, wcs)

        if isinstance(wcs, ExtraCoords):
            wcs = wcs.wcs

        world_axis_physical_types = wcs.world_axis_physical_types

        # If user has supplied axes, extract only the
        # world coords that correspond to those axes.
        if axes:
            world_indices = utils.wcs.calculate_world_indices_from_axes(wcs, axes)
            axes_coords = [axes_coords[i] for i in world_indices]
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

    @utils.misc.sanitise_wcs
    def crop(self, lower_corner, upper_corner, wcs=None):
        # The docstring is defined in NDCubeABC
        lower_corner, upper_corner = utils.misc.sanitize_corners(lower_corner, upper_corner)

        # Quit out early if we are no-op
        lower_nones = np.array([lower is None for lower in lower_corner])
        upper_nones = np.array([upper is None for upper in upper_corner])
        if (lower_nones & upper_nones).all():
            return self

        lower_corner, upper_corner = self._fill_in_crop_nones(lower_corner, upper_corner, wcs, False)

        if isinstance(wcs, BaseHighLevelWCS):
            wcs = wcs.low_level_wcs

        lower_corner_values = high_level_objects_to_values(*lower_corner, low_level_wcs=wcs)
        upper_corner_values = high_level_objects_to_values(*upper_corner, low_level_wcs=wcs)
        lower_corner_values = [v << u.Unit(unit) for v, unit in zip(lower_corner_values, wcs.world_axis_units)]
        upper_corner_values = [v << u.Unit(unit) for v, unit in zip(upper_corner_values, wcs.world_axis_units)]

        points = self._bounding_box_to_points(lower_corner_values, upper_corner_values, wcs)
        return self._crop_from_points(*points, wcs=wcs)

    @utils.misc.sanitise_wcs
    def crop_by_values(self, lower_corner, upper_corner, units=None, wcs=None):
        # The docstring is defined in NDCubeABC
        # Sanitize inputs.
        lower_corner, upper_corner = utils.misc.sanitize_corners(lower_corner, upper_corner)

        # Quit out early if we are no-op
        lower_nones = np.array([lower is None for lower in lower_corner])
        upper_nones = np.array([upper is None for upper in upper_corner])
        if (lower_nones & upper_nones).all():
            return self

        n_coords = len(lower_corner)
        if units is None:
            units = [None] * n_coords
        elif len(units) != n_coords:
            raise ValueError("units must be None or have same length as corner inputs.")

        # Convert float inputs to quantities using units.
        types_with_units = (u.Quantity, type(None))
        for i, (lower, upper, unit) in enumerate(zip(lower_corner, upper_corner, units)):
            lower_is_float = not isinstance(lower, types_with_units)
            upper_is_float = not isinstance(upper, types_with_units)
            if unit is None and (lower_is_float or upper_is_float):
                raise TypeError("If corner value is not a Quantity or None, "
                                "unit must be a valid astropy Unit or unit string."
                                f"index: {i}; lower type: {type(lower)}; "
                                f"upper type: {type(upper)}; unit: {unit}")
            if lower_is_float:
                lower_corner[i] = u.Quantity(lower, unit=unit)
            if upper_is_float:
                upper_corner[i] = u.Quantity(upper, unit=unit)
            # Convert each corner value to the same unit.
            if lower_corner[i] is not None and upper_corner[i] is not None:
                upper_corner[i] = upper_corner[i].to(lower_corner[i].unit)

        lower_corner, upper_corner = self._fill_in_crop_nones(lower_corner, upper_corner, wcs, True)

        # Convert coordinates to units used by WCS as WCS.world_to_array_index
        # does not handle quantities.
        lower_corner = utils.misc.convert_quantities_to_units(lower_corner,
                                                              self.wcs.world_axis_units)
        upper_corner = utils.misc.convert_quantities_to_units(upper_corner,
                                                              self.wcs.world_axis_units)

        points = self._bounding_box_to_points(lower_corner, upper_corner, wcs)
        return self._crop_from_points(*points, wcs=wcs)

    def _fill_in_crop_nones(self, lower_corner, upper_corner, wcs, crop_by_values):
        """
        Replace any instance of None in the inputs with the bounds for that axis.
        """
        lower_nones = np.array([lower is None for lower in lower_corner])
        upper_nones = np.array([upper is None for upper in upper_corner])

        if crop_by_values:
            if isinstance(wcs, BaseHighLevelWCS):
                array_index_to_world = wcs.low_level_wcs.array_index_to_world_values
            else:
                array_index_to_world = wcs.array_index_to_world_values
        else:
            array_index_to_world = wcs.array_index_to_world

        # If user did not provide all intervals,
        # calculate missing intervals based on whole cube range along those axes.
        if lower_nones.any() or upper_nones.any():
            # Calculate real world coords for first and last index for all axes.
            array_intervals = [[0, np.round(d.value - 1).astype(int)] for d in self.dimensions]
            intervals = array_index_to_world(*array_intervals)
            # Overwrite None corner values with world coords of first or last index.
            iterable = zip(lower_nones, upper_nones, intervals)
            for i, (lower_is_none, upper_is_none, interval) in enumerate(iterable):
                if lower_is_none:
                    lower_corner[i] = interval[0]
                if upper_is_none:
                    upper_corner[i] = interval[-1]

        return lower_corner, upper_corner

    def _bounding_box_to_points(self, lower_corner_values, upper_corner_values, wcs):
        """
        Convert two corners of a bounding box to the points of all corners.
        """
        return tuple(itertools.product(*zip(lower_corner_values, upper_corner_values)))

    def _crop_from_points(self, *world_points_values, wcs):
        """
        Crop to the minimum cube in pixel-space that contains all the specified world points.

        Parameters
        ----------
        world_points_values
            The world coordinates in wcsapi "values" form (i.e. arrays /
            floats), for however many world points should be contained in the
            output cube. Each argument should be a tuple with number of
            coordinates equal to the number of world axes.

        wcs : `~astropy.wcs.wcsapi.BaseHighLevelWCS`, `~astropy.wcs.wcsapi.BaseLowLevelWCS`
            The WCS to use to convert the world coordinates to array indices.

        Returns
        -------
        new_cube
            A new ``NDCube`` instance.
        """
        if isinstance(wcs, BaseHighLevelWCS):
            wcs = wcs.low_level_wcs

        # Convert all points to array indices.
        point_indices = []
        for point in world_points_values:
            indices = wcs.world_to_array_index_values(*point)

            if not isinstance(indices, tuple):
                indices = (indices,)

            point_indices.append(indices)

        point_indices = np.array(point_indices)
        lower = np.min(point_indices, axis=0)
        upper = np.max(point_indices, axis=0) + 1

        # Wrap the limits to the size of the array
        lower = [int(np.clip(index, 0, self.data.shape[i])) for i, index in enumerate(lower)]
        upper = [int(np.clip(index, 0, self.data.shape[i])) for i, index in enumerate(upper)]

        item = tuple(slice(l, u) for l, u in zip(lower, upper))

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
        Separates slices of NDCubes along a given axis into an NDCubeSequence of (N-1)DCubes.

        Parameters
        ----------
        axis : `int`
            The array axis along which the data is to be changed.

        Returns
        -------
        result : `ndcube.NDCubeSequence`
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
        return NDCubeSequence(result_cubes, meta=self.meta)

    def _validate_algorithm_and_order(self, algorithm, order):
        order_compatibility = {
            'interpolation': ['nearest-neighbor', 'bilinear', 'biquadratic', 'bicubic'],
            'adaptive': ['nearest-neighbor', 'bilinear'],
            'exact': []
        }

        if algorithm in order_compatibility:
            if order_compatibility[algorithm] and order not in order_compatibility[algorithm]:
                raise ValueError(f"For '{algorithm}' algorithm, the 'order' argument must be "
                                 f"one of {', '.join(order_compatibility[algorithm])}.")

        else:
            raise ValueError(f"The 'algorithm' argument must be one of "
                             f"{', '.join(order_compatibility.keys())}.")

    def reproject_to(self, target_wcs, algorithm='interpolation', shape_out=None, order='bilinear',
                     output_array=None, parallel=False, return_footprint=False):
        """
        Reprojects this NDCube to the coordinates described by another WCS object.

        Parameters
        ----------
        algorithm: `str`
            The algorithm to use for reprojecting. This can be any of: 'interpolation', 'adaptive',
            and 'exact'.

        target_wcs : `astropy.wcs.wcsapi.BaseHighLevelWCS`, `astropy.wcs.wcsapi.BaseLowLevelWCS`,
            or `astropy.io.fits.Header`
            The WCS object to which the ``NDCube`` is to be reprojected.

        shape_out: `tuple`, optional
            The shape of the output data array. The ordering of the dimensions must follow NumPy
            ordering and not the WCS pixel shape.
            If not specified, `~astropy.wcs.wcsapi.BaseLowLevelWCS.array_shape` attribute
            (if available) from the low level API of the ``target_wcs`` is used.

        order: `int` or `str`
            The order of the interpolation (used only when the 'interpolation' or 'adaptive'
            algorithm is selected).
            For 'interpolation' algorithm, this can be any of: 'nearest-neighbor', 'bilinear',
            'biquadratic', and 'bicubic'.
            For 'adaptive' algorithm, this can be either 'nearest-neighbor' or 'bilinear'.

        output_array: `numpy.ndarray`, optional
            An array in which to store the reprojected data. This can be any numpy array
            including a memory map, which may be helpful when dealing with extremely large files.

        parallel: `bool` or `int`
            Flag for parallel implementation (used only when the 'exact' algorithm is selected).
            If ``True``, a parallel implementation is chosen and the number of processes is
            selected automatically as the number of logical CPUs detected on the machine.
            If ``False``, a serial implementation is chosen.
            If the flag is a positive integer n greater than one, a parallel implementation
            using n processes is chosen.

        return_footprint: `bool`
            Whether to return the footprint in addition to the output NDCube.

        Returns
        -------
        resampled_cube : `ndcube.NDCube`
            A new resultant NDCube object, the supplied ``target_wcs`` will be the ``.wcs`` attribute of the output ``NDCube``.

        footprint: `numpy.ndarray`
            Footprint of the input array in the output array. Values of 0 indicate no coverage or
            valid values in the input image, while values of 1 indicate valid values.

        Notes
        -----
        This method doesn't support handling of the ``mask``, ``extra_coords``, and ``uncertainty`` attributes yet.
        However, ``meta`` and ``global_coords`` are copied to the output ``NDCube``.
        """
        try:
            from reproject import reproject_adaptive, reproject_exact, reproject_interp
            from reproject.wcs_utils import has_celestial
        except ModuleNotFoundError:
            raise ImportError("The NDCube.reproject_to method requires the optional package `reproject`.")

        if isinstance(target_wcs, Mapping):
            target_wcs = WCS(header=target_wcs)

        low_level_target_wcs = utils.wcs.get_low_level_wcs(target_wcs, 'target_wcs')

        # 'adaptive' and 'exact' algorithms work only on 2D celestial WCS.
        if algorithm == 'adaptive' or algorithm == 'exact':
            if low_level_target_wcs.pixel_n_dim != 2 or low_level_target_wcs.world_n_dim != 2:
                raise ValueError('For adaptive and exact algorithms, target_wcs must be 2D.')

            if not has_celestial(target_wcs):
                raise ValueError('For adaptive and exact algorithms, '
                                 'target_wcs must contain celestial axes only.')

        if not utils.wcs.compare_wcs_physical_types(self.wcs, target_wcs):
            raise ValueError('Given target_wcs is not compatible with this NDCube, the physical types do not match.')

        # If shape_out is not specified explicity, try to extract it from the low level WCS
        if not shape_out:
            if hasattr(low_level_target_wcs, 'array_shape') and low_level_target_wcs.array_shape is not None:
                shape_out = low_level_target_wcs.array_shape
            else:
                raise ValueError("shape_out must be specified if target_wcs does not have the array_shape attribute.")

        self._validate_algorithm_and_order(algorithm, order)

        if algorithm == 'interpolation':
            data = reproject_interp(self, output_projection=target_wcs, shape_out=shape_out,
                                    order=order, output_array=output_array,
                                    return_footprint=return_footprint)

        elif algorithm == 'adaptive':
            data = reproject_adaptive(self, output_projection=target_wcs, shape_out=shape_out,
                                      order=order, return_footprint=return_footprint)

        elif algorithm == 'exact':
            data = reproject_exact(self, output_projection=target_wcs, shape_out=shape_out,
                                   parallel=parallel, return_footprint=return_footprint)

        if return_footprint:
            data, footprint = data

        resampled_cube = type(self)(data, wcs=target_wcs, meta=deepcopy(self.meta))
        resampled_cube._global_coords = deepcopy(self.global_coords)

        if return_footprint:
            return resampled_cube, footprint

        return resampled_cube


class NDCube(NDCubeBase):
    """
    Class representing N-D data described by a single array and set of WCS transformations.

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
    # We special case the default mpl plotter here so that we can only import
    # matplotlib when `.plotter` is accessed and raise an ImportError at the
    # last moment.
    plotter = PlotterDescriptor(default_type="mpl_plotter")
    """
    A `~.MatplotlibPlotter` instance providing visualization methods.

    The type of this attribute can be changed to provide custom visualization functionality.
    """

    def _as_mpl_axes(self):
        if hasattr(self.plotter, "_as_mpl_axes"):
            return self.plotter._as_mpl_axes()
        else:
            warnings.warn(f"The current plotter {self.plotter} does not have a '_as_mpl_axes' method. "
                          "The default MatplotlibPlotter._as_mpl_axes method will be used instead.",
                          UserWarning)

            plotter = MatplotlibPlotter(self)
            return plotter._as_mpl_axes()

    def plot(self, *args, **kwargs):
        """
        A convenience function for the plotters default ``plot()`` method.

        Calling this method is the same as calling ``cube.plotter.plot``, the
        behaviour of this method can change if the `NDCube.plotter` class is
        set to a different ``Plotter`` class.

        """
        if self.plotter is None:
            raise NotImplementedError(
                "This NDCube object does not have a .plotter defined so "
                "no default plotting functionality is available.")

        return self.plotter.plot(*args, **kwargs)
