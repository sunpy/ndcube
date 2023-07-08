import abc
import textwrap
import warnings
from copy import deepcopy
from typing import Any, Tuple, Union, Iterable, Optional
from collections import namedtuple
from collections.abc import Mapping

import astropy.nddata
import astropy.units as u
import numpy as np
from astropy.units import UnitsError

try:
    # Import sunpy coordinates if available to register the frames and WCS functions with astropy
    import sunpy.coordinates  # NOQA
except ImportError:
    pass

from astropy.wcs import WCS
from astropy.wcs.utils import _split_matrix
from astropy.wcs.wcsapi import BaseHighLevelWCS, HighLevelWCSWrapper

from ndcube import utils
from ndcube.extra_coords.extra_coords import ExtraCoords, ExtraCoordsABC
from ndcube.global_coords import GlobalCoords, GlobalCoordsABC
from ndcube.mixins import NDCubeSlicingMixin
from ndcube.ndcube_sequence import NDCubeSequence
from ndcube.utils.wcs_high_level_conversion import values_to_high_level_objects
from ndcube.visualization import PlotterDescriptor
from ndcube.wcs.wrappers import CompoundLowLevelWCS, ResampledLowLevelWCS

__all__ = ['NDCubeABC', 'NDCubeLinkedDescriptor']

# Create mapping to masked array types based on data array type for use in analysis methods.
ARRAY_MASK_MAP = {}
ARRAY_MASK_MAP[np.ndarray] = np.ma.masked_array
try:
    import dask.array
    ARRAY_MASK_MAP[dask.array.core.Array] = dask.array.ma.masked_array
except ImportError:
    pass


class NDCubeABC(astropy.nddata.NDDataBase):

    @property
    @abc.abstractmethod
    def extra_coords(self) -> ExtraCoordsABC:
        """
        Coordinates not described by ``NDCubeABC.wcs`` which vary along one or more axes.
        """

    @property
    @abc.abstractmethod
    def global_coords(self) -> GlobalCoordsABC:
        """
        Coordinate metadata which applies to the whole cube.
        """

    @property
    @abc.abstractmethod
    def combined_wcs(self) -> BaseHighLevelWCS:
        """
        The WCS transform for the NDCube, including the coordinates specified in ``.extra_coords``.

        This transform should implement the high level wcsapi, and have
        ``pixel_n_dim`` equal to the number of array dimensions in the
        `~ndcube.NDCube`. The number of world dimensions should be equal to the
        number of world dimensions in ``self.wcs`` and in ``self.extra_coords`` combined.
        """

    @property
    @abc.abstractmethod
    def array_axis_physical_types(self) -> Iterable[Tuple[str, ...]]:
        """
        Returns the WCS physical types that vary along each array axis.

        Returns an iterable of tuples where each tuple corresponds to an array axis and
        holds strings denoting the WCS physical types associated with that array axis.
        Since multiple physical types can be associated with one array axis, tuples can
        be of different lengths. Likewise, as a single physical type can correspond to
        multiple array axes, the same physical type string can appear in multiple tuples.

        The physical types returned by this property are drawn from the
        `~ndcube.NDCube.combined_wcs` property so they include the coordinates contained in
        `~ndcube.NDCube.extra_coords`.
        """

    @abc.abstractmethod
    def axis_world_coords(self,
                          *axes: Union[int, str],
                          pixel_corners: bool = False,
                          wcs: Optional[Union[BaseHighLevelWCS, ExtraCoordsABC]] = None
                          ) -> Iterable[Any]:
        """
        Returns objects representing the world coordinates of pixel centers for a desired axes.

        Parameters
        ----------
        axes: `int` or `str`, or multiple `int` or `str`, optional
            Axis number in numpy ordering or unique substring of
            `ndcube.NDCube.wcs.world_axis_physical_types <astropy.wcs.wcsapi.BaseWCSWrapper>`
            of axes for which real world coordinates are desired.
            Not specifying axes inputs causes results for all axes to be returned.
        pixel_corners: `bool`, optional
            If `True` then instead of returning the coordinates at the centers of the pixels,
            the coordinates at the pixel corners will be returned. This
            increases the size of the output by 1 in all dimensions as all corners are returned.
        wcs: `astropy.wcs.wcsapi.BaseHighLevelWCS`, optional
            The WCS object to used to calculate the world coordinates.
            Although technically this can be any valid WCS, it will typically be
            ``self.wcs``, ``self.extra_coords``, or ``self.combined_wcs`` combining both
            the WCS and extra coords.
            Default=self.wcs
        Returns
        -------
        axes_coords: iterable
            An iterable of "high level" objects giving the real world
            coords for the axes requested by user.
            For example, a tuple of `~astropy.coordinates.SkyCoord` objects.
            The types returned are determined by the WCS object.
            The dimensionality of these objects should match that of
            their corresponding array dimensions, unless ``pixel_corners=True``
            in which case the length along each axis will be 1 greater than
            the number of pixels.
        Examples
        --------
        >>> NDCube.axis_world_coords('lat', 'lon') # doctest: +SKIP
        >>> NDCube.axis_world_coords(2) # doctest: +SKIP

        """

    @abc.abstractmethod
    def axis_world_coords_values(self,
                                 *axes: Union[int, str],
                                 pixel_corners: bool = False,
                                 wcs: Optional[Union[BaseHighLevelWCS, ExtraCoordsABC]] = None
                                 ) -> Iterable[u.Quantity]:
        """
        Returns the world coordinate values of all pixels for desired axes.
        In contrast to :meth:`ndcube.NDCube.axis_world_coords`, this method returns
        `~astropy.units.Quantity` objects. Which only provide units rather than full
        coordinate metadata provided by high-level coordinate objects.

        Parameters
        ----------
        axes: `int` or `str`, or multiple `int` or `str`, optional
            Axis number in numpy ordering or unique substring of
            `ndcube.NDCube.wcs.world_axis_physical_types <astropy.wcs.wcsapi.BaseWCSWrapper>`
            of axes for which real world coordinates are desired.
            axes=None implies all axes will be returned.

        pixel_corners: `bool`, optional
            If `True` then coordinates at pixel corners will be returned rather than at pixel centers.
            This increases the size of the output along each dimension by 1
            as all corners are returned.

        wcs: `~astropy.wcs.wcsapi.BaseHighLevelWCS` or `~ndcube.ExtraCoordsABC`, optional
            The WCS object to be used to calculate the world coordinates.
            Although technically this can be any valid WCS, it will typically be
            ``self.wcs``, ``self.extra_coords``, or ``self.combined_wcs``, combing both
            the WCS and extra coords.
            Defaults to the ``.wcs`` property.

        Returns
        -------
        axes_coords: `tuple` of `~astropy.units.Quantity`
            An iterable of raw coordinate values for all pixels for the requested axes.
            The returned units are determined by the WCS object.
            The dimensionality of these objects should match that of
            their corresponding array dimensions, unless ``pixel_corners=True``
            in which case the length along each axis will be 1 greater than the number of pixels.

        Examples
        --------
        >>> NDCube.axis_world_coords_values('lat', 'lon') # doctest: +SKIP
        >>> NDCube.axis_world_coords_values(2) # doctest: +SKIP

        """

    @abc.abstractmethod
    def crop(self,
             *points: Iterable[Any],
             wcs: Optional[Union[BaseHighLevelWCS, ExtraCoordsABC]] = None
             ) -> "NDCubeABC":
        """
        Crop using real world coordinates.
        This method crops the NDCube to the smallest bounding box in pixel space that
        contains all the provided world coordinate points.

        This function takes the points defined as high-level astropy coordinate objects
        such as `~astropy.coordinates.SkyCoord`, `~astropy.coordinates.SpectralCoord`, etc.

        Parameters
        ----------
        points: iterable of iterables
            Tuples of high level coordinate objects
            e.g. `~astropy.coordinates.SkyCoord`.
            Each iterable of coordinate objects represents a single location
            in the data array in real world coordinates.

            The coordinates of the points as they are passed to
            `~astropy.wcs.wcsapi.BaseHighLevelWCS.world_to_array_index`.
            Therefore their number and order must be compatible with the API
            of that method, i.e. they must be passed in world order.

        wcs: `~astropy.wcs.wcsapi.BaseHighLevelWCS` or `~ndcube.ExtraCoordsABC`
            The WCS to use to calculate the pixel coordinates based on the input.
            Will default to the ``.wcs`` property if not given. While any valid WCS
            could be used it is expected that either the ``.wcs`` or
            ``.extra_coords`` properties will be used.

        Returns
        -------
        `~ndcube.ndcube.NDCubeABC`

        Examples
        --------
        An example of cropping a region of interest on the Sun from a 3-D image-time cube:
        >>> point1 = [SkyCoord(-50*u.deg, -40*u.deg, frame=frames.HeliographicStonyhurst), None]  # doctest: +SKIP
        >>> point2 = [SkyCoord(0*u.deg, -6*u.deg, frame=frames.HeliographicStonyhurst), None]  # doctest: +SKIP
        >>> NDCube.crop(point1, point2) # doctest: +SKIP

        """

    @abc.abstractmethod
    def crop_by_values(self,
                       *points: Iterable[Union[u.Quantity, float]],
                       units: Optional[Iterable[Union[str, u.Unit]]] = None,
                       wcs: Optional[Union[BaseHighLevelWCS, ExtraCoordsABC]] = None
                       ) -> "NDCubeABC":
        """
        Crop using real world coordinates.
        This method crops the NDCube to the smallest bounding box in pixel space that
        contains all the provided world coordinate points.

        This function takes points as iterables of low-level coordinate objects,
        i.e. `~astropy.units.Quantity` objects. This differs from :meth:`~ndcube.NDCube.crop`
        which takes high-level coordinate objects requiring all the relevant coordinate
        information such as coordinate frame etc. Hence this method's API is more basic
        but less explicit.

        Parameters
        ----------
        points: iterable
            Tuples of coordinate values, the length of the tuples must be
            equal to the number of world dimensions. These points are
            passed to ``wcs.world_to_array_index_values`` so their units
            and order must be compatible with that method.

        units: `str` or `~astropy.units.Unit`
            If the inputs are set without units, the user must set the units
            inside this argument as `str` or `~astropy.units.Unit` objects.
            The length of the iterable must equal the number of world dimensions
            and must have the same order as the coordinate points.

        wcs: `~astropy.wcs.wcsapi.BaseHighLevelWCS` or `~ndcube.ExtraCoordsABC`
            The WCS to use to calculate the pixel coordinates based on the input.
            Will default to the ``.wcs`` property if not given. While any valid WCS
            could be used it is expected that either the ``.wcs`` or
            ``.extra_coords`` properties will be used.

        Returns
        -------
        `~ndcube.ndcube.NDCubeABC`

        Examples
        --------
        An example of cropping a region of interest on the Sun from a 3-D image-time cube:
        >>> NDCube.crop_by_values((-600, -600, 0), (0, 0, 0), units=(u.arcsec, u.arcsec, u.s)) # doctest: +SKIP
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


class NDCubeBase(NDCubeABC, astropy.nddata.NDData, NDCubeSlicingMixin):
    """
    Class representing N-D data described by a single array and set of WCS transformations.

    Parameters
    ----------
    data : array-like or `astropy.nddata.NDData`
        The array holding the actual data in this object.

    wcs : `astropy.wcs.wcsapi.BaseLowLevelWCS`, `astropy.wcs.wcsapi.BaseHighLevelWCS`, optional
        The WCS object containing the axes' information, optional only if
        ``data`` is an `astropy.nddata.NDData` object.

    uncertainty : Any, optional
        Uncertainty in the dataset. Should have an attribute uncertainty_type
        that defines what kind of uncertainty is stored, for example "std"
        for standard deviation or "var" for variance. A metaclass defining such
        an interface is `~astropy.nddata.NDUncertainty` - but isn't mandatory.
        If the uncertainty has no such attribute the uncertainty is stored as
        `~astropy.nddata.UnknownUncertainty`.
        Defaults to None.

    mask : Any, optional
        Mask for the dataset. Masks should follow the numpy convention
        that valid data points are marked by `False` and invalid ones with `True`.
        Defaults to `None`.

    meta : dict-like object, optional
        Additional meta information about the dataset. If no meta is provided
        an empty dictionary is created.

    unit : Unit-like or `str`, optional
        Unit for the dataset. Strings that can be converted to a `~astropy.units.Unit` are allowed.
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
        # Docstring in NDCubeABC.
        return self._extra_coords

    @property
    def global_coords(self):
        # Docstring in NDCubeABC.
        return self._global_coords

    @property
    def combined_wcs(self):
        # Docstring in NDCubeABC.
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
        # Docstring in NDCubeABC.
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
            if wcs is None:
                return []

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

    @utils.cube.sanitize_wcs
    def axis_world_coords(self, *axes, pixel_corners=False, wcs=None):

        # Docstring in NDCubeABC.
        if isinstance(wcs, BaseHighLevelWCS):
            wcs = wcs.low_level_wcs

        axes_coords = self._generate_world_coords(pixel_corners, wcs)

        if isinstance(wcs, ExtraCoords):
            wcs = wcs.wcs
            if not wcs:
                return tuple()

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

    @utils.cube.sanitize_wcs
    def axis_world_coords_values(self, *axes, pixel_corners=False, wcs=None):
        # Docstring in NDCubeABC.
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

    def crop(self, *points, wcs=None):
        # The docstring is defined in NDCubeABC
        # Calculate the array slice item corresponding to bounding box and return sliced cube.
        item = self._get_crop_item(*points, wcs=wcs)
        return self[item]

    @utils.cube.sanitize_wcs
    def _get_crop_item(self, *points, wcs=None):
        # Sanitize inputs.
        no_op, points, wcs = utils.cube.sanitize_crop_inputs(points, wcs)
        # Quit out early if we are no-op
        if no_op:
            return tuple([slice(None)] * wcs.pixel_n_dim)
        else:
            comp = [c[0] for c in wcs.world_axis_object_components]
            # Trim to unique component names - `np.unique(..., return_index=True)
            # keeps sorting alphabetically, set() seems just nondeterministic.
            for k, c in enumerate(comp):
                if comp.count(c) > 1:
                    comp.pop(k)
            classes = [wcs.world_axis_object_classes[c][0] for c in comp]
            for i, point in enumerate(points):
                if len(point) != len(comp):
                    raise ValueError(f"{len(point)} components in point {i} do not match "
                                     f"WCS with {len(comp)} components.")
                for j, value in enumerate(point):
                    if not (value is None or isinstance(value, classes[j])):
                        raise TypeError(f"{type(value)} of component {j} in point {i} is "
                                        f"incompatible with WCS component {comp[j]} "
                                        f"{classes[j]}.")
            return utils.cube.get_crop_item_from_points(points, wcs, False)

    def crop_by_values(self, *points, units=None, wcs=None):
        # The docstring is defined in NDCubeABC
        # Calculate the array slice item corresponding to bounding box and return sliced cube.
        item = self._get_crop_by_values_item(*points, units=units, wcs=wcs)
        return self[item]

    @utils.cube.sanitize_wcs
    def _get_crop_by_values_item(self, *points, units=None, wcs=None):
        # Sanitize inputs.
        no_op, points, wcs = utils.cube.sanitize_crop_inputs(points, wcs)
        # Quit out early if we are no-op
        if no_op:
            return tuple([slice(None)] * wcs.pixel_n_dim)
        # Convert float inputs to quantities using units.
        n_coords = len(points[0])
        if units is None:
            units = [None] * n_coords
        elif len(units) != n_coords:
            raise ValueError(f"Units must be None or have same length {n_coords} as corner inputs.")
        types_with_units = (u.Quantity, type(None))
        for i, point in enumerate(points):
            if len(point) != wcs.world_n_dim:
                raise ValueError(f"{len(point)} dimensions in point {i} do not match "
                                 f"WCS with {wcs.world_n_dim} world dimensions.")
            for j, (value, unit) in enumerate(zip(point, units)):
                value_is_float = not isinstance(value, types_with_units)
                if value_is_float:
                    if unit is None:
                        raise TypeError(
                            "If an element of a point is not a Quantity or None, "
                            "the corresponding unit must be a valid astropy Unit or unit string."
                            f"index: {i}; coord type: {type(value)}; unit: {unit}")
                    points[i][j] = u.Quantity(value, unit=unit)
                if value is not None:
                    try:
                        points[i][j] = points[i][j].to(wcs.world_axis_units[j])
                    except UnitsError as err:
                        raise UnitsError(f"Unit '{points[i][j].unit}' of coordinate object {j} in point {i} is "
                                         f"incompatible with WCS unit '{wcs.world_axis_units[j]}'") from err

        return utils.cube.get_crop_item_from_points(points, wcs, True)

    def __str__(self):
        return textwrap.dedent(f"""\
                NDCube
                ------
                Dimensions: {self.dimensions}
                Physical Types of Axes: {self.array_axis_physical_types}
                Unit: {self.unit}
                Data Type: {self.data.dtype}""")

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

    def reproject_to(self, target_wcs, algorithm='interpolation', shape_out=None, return_footprint=False, **reproject_args):
        """
        Reprojects the instance to the coordinates described by another WCS object.

        Parameters
        ----------
        target_wcs : `astropy.wcs.wcsapi.BaseHighLevelWCS`, `astropy.wcs.wcsapi.BaseLowLevelWCS`,
            or `astropy.io.fits.Header`
            The WCS object to which the `ndcube.NDCube` is to be reprojected.

        algorithm: {'interpolation' | 'adaptive' | 'exact'}
            The algorithm to use for reprojecting.
            When set to "interpolation" `~reproject.reproject_interp` is used,
            when set to "adaptive" `~reproject.reproject_adaptive` is used and
            when set to "exact" `~reproject.reproject_exact` is used.

        shape_out: `tuple`, optional
            The shape of the output data array. The ordering of the dimensions must follow NumPy
            ordering and not the WCS pixel shape.
            If not specified, `~astropy.wcs.wcsapi.BaseLowLevelWCS.array_shape` attribute
            (if available) from the low level API of the ``target_wcs`` is used.

        return_footprint : `bool`
            If `True` the footprint is returned in addition to the new `~ndcube.NDCube`.
            Defaults to `False`.

        **reproject_args
            All other arguments are passed through to the reproject function
            being called. The function being called depends on the
            ``algorithm=`` keyword argument, see that for more details.

        Returns
        -------
        reprojected_cube : `ndcube.NDCube`
            A new resultant NDCube object, the supplied ``target_wcs`` will be the ``.wcs`` attribute of the output `~ndcube.NDCube`.

        footprint: `numpy.ndarray`
            Footprint of the input array in the output array.
            Values of 0 indicate no coverage or valid values in the input
            image, while values of 1 indicate valid values.

        See Also
        --------

        reproject.reproject_interp
        reproject.reproject_adaptive
        reproject.reproject_exact

        Notes
        -----
        This method doesn't support handling of the ``mask``, ``extra_coords``, and ``uncertainty`` attributes yet.
        However, ``meta`` and ``global_coords`` are copied to the output `ndcube.NDCube`.
        """
        try:
            from reproject import reproject_adaptive, reproject_exact, reproject_interp
            from reproject.wcs_utils import has_celestial
        except ModuleNotFoundError:
            raise ImportError("The NDCube.reproject_to method requires the optional package `reproject`.")

        algorithms = {
            "interpolation": reproject_interp,
            "adaptive": reproject_adaptive,
            "exact": reproject_exact,
        }

        if algorithm not in algorithms.keys():
            raise ValueError(f"{algorithm=} is not valid, it must be one of {', '.join(algorithms.keys())}.")

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

        # TODO: Upstream this check into reproject
        # If shape_out is not specified explicitly,
        # try to extract it from the low level WCS
        if not shape_out:
            if hasattr(low_level_target_wcs, 'array_shape') and low_level_target_wcs.array_shape is not None:
                shape_out = low_level_target_wcs.array_shape
            else:
                raise ValueError("shape_out must be specified if target_wcs does not have the array_shape attribute.")

        data = algorithms[algorithm](self,
                                     output_projection=target_wcs,
                                     shape_out=shape_out,
                                     return_footprint=return_footprint,
                                     **reproject_args)

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

    uncertainty : Any, optional
        Uncertainty in the dataset. Should have an attribute uncertainty_type
        that defines what kind of uncertainty is stored, for example "std"
        for standard deviation or "var" for variance. A metaclass defining
        such an interface is NDUncertainty - but isn't mandatory. If the uncertainty
        has no such attribute the uncertainty is stored as UnknownUncertainty.
        Defaults to None.

    mask : Any, optional
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
    # Enabling the NDCube reflected operators is a bit subtle. The NDCube
    # reflected operator will be used only if the Quantity non-reflected operator
    # returns NotImplemented. The Quantity operator strips the unit from the
    # Quantity and tries to combine the value with the NDCube using NumPy's
    # __array_ufunc__(). If NumPy believes that it can proceed, this will result
    # in an error. We explicitly set __array_ufunc__ = None so that the NumPy
    # call, and consequently the Quantity operator, will return NotImplemented.
    __array_ufunc__ = None

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
        behaviour of this method can change if the `ndcube.NDCube.plotter` class is
        set to a different ``Plotter`` class.

        """
        if self.plotter is None:
            raise NotImplementedError(
                "This NDCube object does not have a .plotter defined so "
                "no default plotting functionality is available.")

        return self.plotter.plot(*args, **kwargs)

    def _new_instance_from_op(self, new_data, new_unit, new_uncertainty):
        # This implicitly assumes that the arithmetic operation does not alter
        # the WCS, mask, or metadata.
        new_cube = type(self)(new_data,
                              unit=new_unit,
                              wcs=self.wcs,
                              mask=deepcopy(self.mask),
                              meta=deepcopy(self.meta),
                              uncertainty=new_uncertainty)
        if self.extra_coords is not None:
            new_cube._extra_coords = deepcopy(self.extra_coords)
        if self.global_coords is not None:
            new_cube._global_coords = deepcopy(self.global_coords)
        return new_cube

    def __neg__(self):
        return self._new_instance_from_op(-self.data, deepcopy(self.unit),
                                          deepcopy(self.uncertainty))

    def __add__(self, value):
        if hasattr(value, 'unit'):
            if isinstance(value, u.Quantity):
                # NOTE: if the cube does not have units, we cannot
                # perform arithmetic between a unitful quantity.
                # This forces a conversion to a dimensionless quantity
                # so that an error is thrown if value is not dimensionless
                cube_unit = u.Unit('') if self.unit is None else self.unit
                new_data = self.data + value.to_value(cube_unit)
            else:
                # NOTE: This explicitly excludes other NDCube objects and NDData objects
                # which could carry a different WCS than the NDCube
                return NotImplemented
        elif self.unit not in (None, u.Unit("")):
            raise TypeError("Cannot add a unitless object to an NDCube with a unit.")
        else:
            new_data = self.data + value
        return self._new_instance_from_op(new_data, deepcopy(self.unit), deepcopy(self.uncertainty))

    def __radd__(self, value):
        return self.__add__(value)

    def __sub__(self, value):
        return self.__add__(-value)

    def __rsub__(self, value):
        return self.__neg__().__add__(value)

    def __mul__(self, value):
        if hasattr(value, 'unit'):
            if isinstance(value, u.Quantity):
                # NOTE: if the cube does not have units, set the unit
                # to dimensionless such that we can perform arithmetic
                # between the two.
                cube_unit = u.Unit('') if self.unit is None else self.unit
                value_unit = value.unit
                value = value.to_value()
                new_unit = cube_unit * value_unit
            else:
                return NotImplemented
        else:
            new_unit = self.unit
        new_data = self.data * value
        new_uncertainty = (type(self.uncertainty)(self.uncertainty.array * value)
                           if self.uncertainty is not None else None)
        new_cube = self._new_instance_from_op(new_data, new_unit, new_uncertainty)
        return new_cube

    def __rmul__(self, value):
        return self.__mul__(value)

    def __truediv__(self, value):
        return self.__mul__(1/value)

    def to(self, new_unit, **kwargs):
        """Convert instance to another unit.

        Converts the data, uncertainty and unit and returns a new instance
        with other attributes unchanged.

        Parameters
        ----------
        new_unit: `astropy.units.Unit`
            The unit to convert to.
        kwargs:
            Passed to the unit conversion method, self.unit.to.

        Returns
        -------
        : `~ndcube.NDCube`
            A new instance with the new unit and data and uncertainties scales accordingly.
        """
        new_unit = u.Unit(new_unit)
        return self * (self.unit.to(new_unit, **kwargs) * new_unit / self.unit)

    def rebin(self, bin_shape, operation=np.mean, operation_ignores_mask=False, handle_mask=np.all,
              propagate_uncertainties=False, new_unit=None, **kwargs):
        """
        Downsample array by combining contiguous pixels into bins.

        Values in bins are determined by applying a function to the pixel values within it.
        The number of pixels in each bin in each dimension is given by the bin_shape input.
        This must be an integer fraction of the cube's array size in each dimension.
        If the NDCube instance has uncertainties attached, they are propagated
        depending on binning method chosen.

        Parameters
        ----------
        bin_shape : array-like
            The number of pixels in a bin in each dimension.
            Must be the same length as number of dimensions in data.
            Each element must be in int. If they are not they will be rounded
            to the nearest int.
        operation : function
            Function applied to the data to derive values of the bins.
            Default is `numpy.mean`
        operation_ignores_mask: `bool`
            Determines how masked values are handled.
            If False (default), masked values are excluded when calculating rebinned value.
            If True, masked values are used in calculating rebinned value.
        handle_mask: `None` or function
            Function to apply to each bin in the mask to calculate the new mask values.
            If `None` resultant mask is `None`.
            Default is `numpy.all`
        propagate_uncertainties: `bool` or function.
            If False, uncertainties are dropped.
            If True, default algorithm is used (`~ndcube.utils.cube.propagate_rebin_uncertainties`)
            Can also be set to a function which performs custom uncertainty propagation.
            Additional kwargs provided to this method are passed onto this function.
            See Notes section on how to write a custom ``propagate_uncertainties`` function.
        new_unit: `astropy.units.Unit`, optional
            If the rebinning operation alters the data unit, the new unit can be
            provided here.
        kwargs
            All kwargs are passed to the error propagation function.

        Returns
        -------
        new_cube: `~ndcube.NDCube`
            The resolution-degraded cube.

        References
        ----------
        https://mail.scipy.org/pipermail/numpy-discussion/2010-July/051760.html

        Notes
        -----
        **Rebining Algorithm**
        Rebinning is achieved by reshaping the N-D array to a 2N-D array and
        applying the function over the odd-numbered axes. To demonstrate,
        consider the following example. Let's say you have an array::

             x = np.array([[0, 0, 0, 1, 1, 1],
                           [0, 0, 1, 1, 0, 0],
                           [1, 1, 0, 0, 1, 1],
                           [0, 0, 0, 0, 1, 1],
                           [1, 0, 1, 0, 1, 1],
                           [0, 0, 1, 0, 0, 0]])

        and you want to sum over 2x2 non-overlapping sub-arrays. This summing can
        be done by reshaping the array::

             y = x.reshape(3,2,3,2)

        and then summing over the 1st and third directions::

             y2 = y.sum(axis=3).sum(axis=1)

        which gives the expected array::

             array([[0, 3, 2],
                    [2, 0, 4],
                    [1, 2, 2]])

        **Defining Custom Error Propagation**
        To perform custom uncertainty propagation, a function must be provided via the
        propgate_uncertainty kwarg. This function must accept, although doesn't have to
        use, the following args:

        uncertainty: `astropy.nddata.NDUncertainty` but not `astropy.nddata.UnknownUncertainty`
            The uncertainties associated with the data.
        data: array-like
            The data associated with the above uncertainties.
            Must have same shape as uncertainty.
        mask: array-like of `bool` or `None`
            Indicates whether any uncertainty elements should be ignored in propagation.
            True elements cause corresponding uncertainty elements to be ignored.
            False elements cause corresponding uncertainty elements to be propagated.
            Must have same shape as above.
            If None, no uncertainties are ignored.

        All kwarg inputs to the rebin method are also passed on transparently to the
        propagation function. Hence additional inputs to the propagation function can be
        included as kwargs to :meth:`ndcube.NDCube.rebin`.

        The shape of the uncertainty, data and mask inputs are such that the first
        dimension represents the pixels in a given bin whose data and uncertainties
        are aggregated by the rebin process. The shape of the remaining dimensions
        must be the same as the final rebinned data. A silly but informative
        example of a custom propagation function might be::

             def my_propagate(uncertainty, data, mask, **kwargs):
                 # As a silly example, propagate uncertainties by summing those in same bin.
                 # Note not all args are used, but function must accept them.
                 n_pixels_per_bin = data.shape[0]  # 1st dimension of inputs gives pixels in bin.
                 final_shape = data.shape[1:]  # Trailing dims give shape of put rebinned data.
                 # Propagate uncerts by adding them.
                 new_uncert = numpy.zeros(final_shape)
                 for i in range(n_pixels_per_bin):
                     new_uncert += uncertainty.array[i]
                 # Alternatively: new_uncerts = uncertainty.array.sum(axis=0)
                 return type(uncertainty)(new_uncert)  # Convert to original uncert type and return.
        """
        # Sanitize input.
        new_unit = new_unit or self.unit
        # Make sure the input bin dimensions are integers.
        bin_shape = np.rint(bin_shape).astype(int)
        offsets = (bin_shape - 1) / 2
        if all(bin_shape == 1):
            return self
        # Ensure bin_size has right number of entries and each entry is an
        # integer fraction of the array shape in each dimension.
        data_shape = self.dimensions.value.astype(int)
        naxes = len(data_shape)
        if len(bin_shape) != naxes:
            raise ValueError("bin_shape must have an entry for each array axis.")
        if (np.mod(data_shape, bin_shape) != 0).any():
            raise ValueError(
                "bin shape must be an integer fraction of the data shape in each dimension. "
                f"data shape: {data_shape};  bin shape: {bin_shape}")

        # Reshape array so odd dimensions represent pixels to be binned
        # then apply function over those axes.
        m = None if (self.mask is None or self.mask is False or operation_ignores_mask) else self.mask
        data = self.data
        if m is not None:
            for array_type, masked_type in ARRAY_MASK_MAP.items():
                if isinstance(self.data, array_type):
                    break
            else:
                masked_type = np.ma.masked_array
                warn.warning("data and mask arrays of different or unrecognized types. "
                             "Casting them into a numpy masked array.")
            data = masked_type(self.data, m)

        reshape = np.empty(data_shape.size + bin_shape.size, dtype=int)
        new_shape = (data_shape / bin_shape).astype(int)
        reshape[0::2] = new_shape
        reshape[1::2] = bin_shape
        reshape = tuple(reshape)
        reshaped_data = data.reshape(reshape)
        operation_axes = tuple(range(len(reshape) - 1, 0, -2))
        new_data = operation(reshaped_data, axis=operation_axes)
        if isinstance(new_data, ARRAY_MASK_MAP[np.ndarray]):
            new_data = new_data.data
        if handle_mask is None:
            new_mask = None
        elif isinstance(self.mask, (type(None), bool)):  # Preserve original mask type.
            new_mask = self.mask
        else:
            reshaped_mask = self.mask.reshape(reshape)
            new_mask = handle_mask(reshaped_mask, axis=operation_axes)

        # Propagate uncertainties if propagate_uncertainties kwarg set.
        new_uncertainty = None
        if propagate_uncertainties:
            if self.uncertainty is None:
                warnings.warn("Uncertainties cannot be propagated as there are no uncertainties, "
                              "i.e. self.uncertainty is None.")
            elif isinstance(self.uncertainty, astropy.nddata.UnknownUncertainty):
                warnings.warn("self.uncertainty is of type UnknownUncertainty which does not "
                              "support uncertainty propagation.")
            elif (not operation_ignores_mask
                  and (self.mask is True or (self.mask is not None
                                             and not isinstance(self.mask, bool)
                                             and self.mask.all()))):
                warnings.warn("Uncertainties cannot be propagated as all values are masked and "
                              "operation_ignores_mask is False.")
            else:
                if propagate_uncertainties is True:
                    propagate_uncertainties = utils.cube.propagate_rebin_uncertainties
                # If propagate_uncertainties, use astropy's infrastructure.
                # For this the data and uncertainty must be reshaped
                # so the first dimension represents the flattened size of a single bin
                # while the rest represent the shape of the new data. Then the elements
                # in each bin can be iterated (all bins being treated in parallel) and
                # their uncertainties propagated.
                bin_size = bin_shape.prod()
                flat_shape = [bin_size] + list(new_shape)
                dummy_axes = tuple(range(1, len(reshape), 2))
                flat_data = np.moveaxis(reshaped_data, dummy_axes, tuple(range(naxes)))
                flat_data = flat_data.reshape(flat_shape)
                reshaped_uncertainty = self.uncertainty.array.reshape(tuple(reshape))
                flat_uncertainty = np.moveaxis(reshaped_uncertainty, dummy_axes, tuple(range(naxes)))
                flat_uncertainty = flat_uncertainty.reshape(flat_shape)
                flat_uncertainty = type(self.uncertainty)(flat_uncertainty)
                if m is not None:
                    reshaped_mask = self.mask.reshape(tuple(reshape))
                    flat_mask = np.moveaxis(reshaped_mask, dummy_axes, tuple(range(naxes)))
                    flat_mask = flat_mask.reshape(flat_shape)
                else:
                    flat_mask = None
                # Propagate uncertainties.
                new_uncertainty = propagate_uncertainties(
                    flat_uncertainty, flat_data, flat_mask,
                    operation=operation, operation_ignores_mask=operation_ignores_mask,
                    handle_mask=handle_mask, new_unit=new_unit, **kwargs)

        # Resample WCS
        new_wcs = ResampledLowLevelWCS(self.wcs.low_level_wcs, bin_shape[::-1])

        # Reform NDCube.
        new_cube = type(self)(new_data, new_wcs, uncertainty=new_uncertainty, mask=new_mask,
                              meta=self.meta, unit=new_unit)
        new_cube._global_coords = self._global_coords
        # Reconstitute extra coords
        if not self.extra_coords.is_empty:
            new_array_grids = [None if bin_shape[i] == 1 else
                               np.arange(offsets[i], data_shape[i] + offsets[i], bin_shape[i])
                               for i in range(naxes)]
            new_cube._extra_coords = self.extra_coords.resample(bin_shape, ndcube=new_cube)

        return new_cube
