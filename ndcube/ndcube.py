
import abc
import textwrap
import warnings
import numbers

import astropy.nddata
import astropy.units as u
import numpy as np
import sunpy.coordinates
from astropy.utils.misc import InheritDocstrings
from astropy.wcs.wcsapi import BaseHighLevelWCS, BaseLowLevelWCS, HighLevelWCSWrapper, SlicedLowLevelWCS
from astropy.wcs.wcsapi.fitswcs import SlicedFITSWCS, custom_ctype_to_ucd_mapping

from ndcube import utils
from ndcube.mixins import NDCubePlotMixin, NDCubeSlicingMixin
from ndcube.ndcube_sequence import NDCubeSequence
from ndcube.utils.cube import _get_dimension_for_pixel, _pixel_centers_or_edges, unique_data_axis
import ndcube.utils.wcs as wcs_utils

__all__ = ['NDCubeABC', 'NDCubeBase', 'NDCube', 'NDCubeOrdered']


class NDCubeMetaClass(abc.ABCMeta):
    """
    A metaclass that combines `abc.ABCMeta`.
    """


class NDCubeABC(astropy.nddata.NDData, metaclass=NDCubeMetaClass):

    @abc.abstractmethod
    def pixel_to_world(self, *quantity_axis_list):
        """
        Convert a pixel coordinate to a data (world) coordinate by using
        `~astropy.wcs.WCS.all_pix2world`.

        Parameters
        ----------
        quantity_axis_list : iterable
            An iterable of `~astropy.units.Quantity` with unit as pixel `pix`.
            Note that these quantities must be entered as separate arguments, not as one list.

        origin : `int`.
            Origin of the top-left corner. i.e. count from 0 or 1.
            Normally, origin should be 0 when passing numpy indices, or 1 if
            passing values from FITS header or map attributes.
            See `~astropy.wcs.WCS.wcs_pix2world` for more information.
            Default is 0.

        Returns
        -------
        coord : `list`
            A list of arrays containing the output coordinates
            reverse of the wcs axis order.
        """

    @abc.abstractmethod
    def world_to_pixel(self, *quantity_axis_list):
        """
        Convert a world coordinate to a data (pixel) coordinate by using
        `~astropy.wcs.WCS.all_world2pix`.

        Parameters
        ----------
        quantity_axis_list : iterable
            A iterable of `~astropy.units.Quantity`.
            Note that these quantities must be entered as separate arguments, not as one list.

        origin : `int`
            Origin of the top-left corner. i.e. count from 0 or 1.
            Normally, origin should be 0 when passing numpy indices, or 1 if
            passing values from FITS header or map attributes.
            See `~astropy.wcs.WCS.wcs_world2pix` for more information.
            Default is 0.

        Returns
        -------
        coord : `list`
            A list of arrays containing the output coordinates
            reverse of the wcs axis order.
        """

    @abc.abstractproperty
    def dimensions(self):
        pass

    @abc.abstractproperty
    def world_axis_physical_types(self):
        pass

    @abc.abstractmethod
    def crop_by_coords(self, lower_corner, interval_widths=None, upper_corner=None, units=None):
        """
        Crops an NDCube given minimum values and interval widths along axes.

        Parameters
        ----------
        lower_corner: iterable of `astropy.units.Quantity` or `float`
            The minimum desired values along each relevant axis after cropping
            described in physical units consistent with the NDCube's wcs object.
            The length of the iterable must equal the number of data dimensions
            and must have the same order as the data.

        interval_widths: iterable of `astropy.units.Quantity` or `float`
            The width of the region of interest in each dimension in physical
            units consistent with the NDCube's wcs object. The length of the
            iterable must equal the number of data dimensions and must have
            the same order as the data. This argument will be removed in versions
            2.0, please use upper_corner argument.

        upper_corner: iterable of `astropy.units.Quantity` or `float`
            The maximum desired values along each relevant axis after cropping
            described in physical units consistent with the NDCube's wcs object.
            The length of the iterable must equal the number of data dimensions
            and must have the same order as the data.

        units: iterable of `astropy.units.quantity.Quantity`, optionnal
            If the inputs are set without units, the user must set the units
            inside this argument as `str`.
            The length of the iterable must equal the number of data dimensions
            and must have the same order as the data.

        Returns
        -------
        result: NDCube
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
    def world_axis_physical_types(self):
        """
        Returns an iterable of strings describing the physical type for each
        world axis.

        The strings conform to the International Virtual Observatory
        Alliance standard, UCD1+ controlled Vocabulary.  For a
        description of the standard and definitions of the different
        strings and string components, see
        http://www.ivoa.net/documents/latest/UCDlist.html.
        """

        # Use the context manager to access the physical types,
        # which are not present in the APE14.
        # APE14 physical types are covered by default.
        with custom_ctype_to_ucd_mapping(wcs_utils.wcs_ivoa_mapping):
            ctype = self.wcs.low_level_wcs.world_axis_physical_types

        return tuple(ctype[::-1])

    def pixel_to_world(self, *quantity_axis_list):
        # The docstring is defined in NDDataBase

        quantity_axis_list = quantity_axis_list[::-1]
        pixel_to_world = self.wcs.pixel_to_world(*quantity_axis_list)
        if isinstance(pixel_to_world, (tuple, list)):
            return pixel_to_world[::-1]

        return pixel_to_world

    def world_to_pixel(self, *quantity_axis_list):
        # The docstring is defined in NDDataBase

        quantity_axis_list = quantity_axis_list[::-1]
        world_to_pixel = self.wcs.world_to_pixel(*quantity_axis_list)

        # Adding the units of the output
        result = [u.Quantity(world_to_pixel[index], unit=u.pix) for index in range(self.wcs.low_level_wcs.pixel_n_dim)]
        return result[::-1]

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

    def axis_world_coord_values(self, *axes, edges=False):
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
        axis_names: `tuple` of `str`
            The physical types of the coords returned.

        axes_coords: `tuple` of `astropy.units.Quantity`
            Real world coords for axes requested by user.
            Returned in same order as axis_names.

        Example
        -------
        >>> NDCube.all_world_coords_values(('lat', 'lon')) # doctest: +SKIP
        >>> NDCube.all_world_coords_values(2) # doctest: +SKIP

        """
        # Create meshgrid of all pixel coordinates.
        # If user, wants edges, set pixel values to pixel edges.
        # Else make pixel centers.
        wcs_shape = self.data.shape[::-1]
        if edges:
            wcs_shape = tuple(np.array(wcs_shape) + 1)
            pixel_inputs = np.meshgrid(*[np.arange(i) - 0.5 for i in wcs_shape], indexing='ij')
        else:
            pixel_inputs = np.meshgrid(*[np.arange(i) for i in wcs_shape], indexing='ij')

        # Get world coords for all axes and all pixels.
        axes_coords = self.wcs.pixel_to_world(*pixel_inputs)

        # Reduce duplication across independent dimensions for each coord
        # and transpose to make dimensions mimic numpy array order rather than WCS order.
        for i, axis_coord in enumerate(axes_coords):
            slices = np.array([slice(None)] * self.wcs.world_n_dim)
            slices[np.invert(self.wcs.axis_correlation_matrix[i])] = 0
            axes_coords[i] = axis_coord[tuple(slices)].T

        world_axis_physical_types = self.wcs.world_axis_physical_types
        # If user has supplied axes, extract only the
        # world coords that correspond to those axes.
        if axes:
            # Convert input axes to WCS world axis indices.
            world_indices = set()
            for axis in axes:
                if isinstance(axis, numbers.Integral):
                    # If axis is int, it is a numpy order array axis.
                    # Convert to pixel axis in WCS order.
                    axis = wcs_utils.reflect_axis_index(np.array([axis]), self.wcs.pixel_n_dim)[0]
                    # Get WCS world axis indices that correspond to the WCS pixel axis
                    # and add to list of indices of WCS world axes whose coords will be returned.
                    world_indices.update(wcs_utils.pixel_axis_to_world_axes(
                        axis, self.wcs.axis_correlation_matrix))
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
        return world_axis_physical_types[::-1], tuple(axes_coords[::-1])


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
                        self._extra_coords_wcs_axis[key]["wcs axis"], wcs_utils._pixel_keep(self.wcs),
                        self.wcs.low_level_wcs.pixel_n_dim),
                    "value": self._extra_coords_wcs_axis[key]["value"]}
        return result

    def crop_by_coords(self, lower_corner, interval_widths=None, upper_corner=None, units=None):
        # The docstring is defined in NDDataBase
        n_dim = self.data.ndim
        # Raising a value error if the arguments have not the same dimensions.
        # Calculation of upper_corner with the inputing interval_widths
        # This part of the code will be removed in version 2.0
        if interval_widths:
            warnings.warn(
                "interval_widths will be removed from the API in version 2.0"
                ", please use upper_corner argument.")
            if upper_corner:
                raise ValueError("Only one of interval_widths or upper_corner "
                                 "can be set. Recommend using upper_corner as "
                                 "interval_widths is deprecated.")
            if (len(lower_corner) != len(interval_widths)) or (len(lower_corner) != n_dim):
                raise ValueError("lower_corner and interval_widths must have "
                                 "same number of elements as number of data "
                                 "dimensions.")
            upper_corner = [lower_corner[i] + interval_widths[i] for i in range(n_dim)]
        # Raising a value error if the arguments have not the same dimensions.
        if (len(lower_corner) != len(upper_corner)) or (len(lower_corner) != n_dim):
            raise ValueError("lower_corner and upper_corner must have the same "
                             "number of elements as number of data dimensions.")

        lower_corner = list(lower_corner)
        upper_corner = list(upper_corner)

        if units:
            # Raising a value error if units have not the data dimensions.
            if len(units) != n_dim:
                raise ValueError('units must have same number of elements as '
                                 'number of data dimensions.')
            # If inputs are not Quantity objects, they are modified into specified units
            lower_corner = [u.Quantity(lower_corner[i], unit=units[i])
                            for i in range(self.data.ndim)]
            upper_corner = [u.Quantity(upper_corner[i], unit=units[i])
                            for i in range(self.data.ndim)]
        else:
            if any([not isinstance(x, u.Quantity) for x in lower_corner + upper_corner]):
                raise TypeError("lower_corner and interval_widths/upper_corner must be "
                                "of type astropy.units.Quantity or the units kwarg "
                                "must be set.")
        # Get all corners of region of interest.
        all_world_corners_grid = np.meshgrid(
            *[u.Quantity([lower_corner[i], upper_corner[i]], unit=lower_corner[i].unit).value
              for i in range(self.data.ndim)])
        all_world_corners = [all_world_corners_grid[i].flatten() * lower_corner[i].unit
                             for i in range(n_dim)]

        # Convert them back to units of world_axis_units
        world_axis_units = self.wcs.low_level_wcs.world_axis_units[::-1]
        all_world_corners = [entries.to(world_axis_units[i]) for i, entries in enumerate(all_world_corners)]

        # Here we are using `wcs.world_to_pixel_values` instead of NDCube's world_to_pixel as the latter
        # requires high_level astropy objects to operate
        # Since it is a low_level API, so input parameters need to be adjusted in wcs ordering.

        # Convert to pixel coordinates
        all_world_corners = all_world_corners[::-1]
        all_pix_corners = self.wcs.low_level_wcs.world_to_pixel_values(*all_world_corners)
        all_pix_corners = all_pix_corners[::-1]
        all_pix_corners = tuple(u.Quantity(a, u.pix).value for a in all_pix_corners)

        # Derive slicing item with which to slice NDCube.
        # Be sure to round down min pixel and round up + 1 the max pixel.
        item = tuple([slice(int(np.clip(axis_pixels.min(), 0, None)),
                            int(np.ceil(axis_pixels.max())) + 1)
                      for axis_pixels in all_pix_corners])
        return self[item]

    def crop_by_extra_coord(self, coord_name, min_coord_value, max_coord_value):
        """
        Crops an NDCube given a minimum value and interval width along an extra
        coord.

        Parameters
        ----------
        coord_name: `str`
            Name of extra coordinate by which to crop.

        min_coord_value: Single value `astropy.units.Quantity`
            The minimum desired value of the extra coord after cropping.
            Unit must be consistent with the extra coord on which cropping is based.

        min_coord_value: Single value `astropy.units.Quantity`
            The maximum desired value of the extra coord after cropping.
            Unit must be consistent with the extra coord on which cropping is based.

        Returns
        -------
        result: `ndcube.NDCube`
        """
        if not isinstance(coord_name, str):
            raise TypeError("The API for this function has changed. "
                            "Please give coord_name, min_coord_value, max_coord_value")
        extra_coord_dict = self.extra_coords[coord_name]
        if isinstance(extra_coord_dict["value"], u.Quantity):
            extra_coord_values = extra_coord_dict["value"]
        else:
            extra_coord_values = np.asarray(extra_coord_dict["value"])
        w = np.logical_and(extra_coord_values >= min_coord_value,
                           extra_coord_values < max_coord_value)
        w = np.arange(len(extra_coord_values))[w]
        item = [slice(None)] * len(self.dimensions)
        item[extra_coord_dict["axis"]] = slice(w[0], w[-1] + 1)
        return self[tuple(item)]

    def __str__(self):
        return textwrap.dedent(f"""\
                NDCube
                ---------------------
                {{wcs}}
                ---------------------
                Length of NDCube: {self.dimensions}
                Axis Types of NDCube: {self.world_axis_physical_types}""").format(wcs=str(self.wcs))

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


class NDCubeOrdered(NDCube):
    """
    Class representing N dimensional cubes with oriented WCS. Extra arguments
    are passed on to NDData's init. The order is TIME, SPECTRAL, SOLAR-x,
    SOLAR-Y and any other dimension. For example, in an x, y, t cube the order
    would be (t,x,y) and in a lambda, t, y cube the order will be (t, lambda,
    y). Extra arguments are passed on to NDData's init.

    Parameters
    ----------
    data: `numpy.ndarray`
        The array holding the actual data in this object.

    wcs: `ndcube.wcs.wcs.WCS`
        The WCS object containing the axes' information. The axes'
        priorities are time, spectral, celestial. This means that if
        present, each of these axis will take precedence over the others.

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

    copy : bool, optional
        Indicates whether to save the arguments as copy. True copies every attribute
        before saving it while False tries to save every parameter as reference.
        Note however that it is not always possible to save the input as reference.
        Default is False.
    """

    def __init__(self, data, wcs, uncertainty=None, mask=None, meta=None,
                 unit=None, extra_coords=None, copy=False, **kwargs):
        axtypes = list(wcs.wcs.ctype)[::-1]
        array_order = utils.cube.select_order(axtypes)
        result_data = data.transpose(array_order)
        result_wcs = utils.wcs.reindex_wcs(wcs, np.array(array_order))
        if uncertainty is not None:
            result_uncertainty = uncertainty.transpose(array_order)
        else:
            result_uncertainty = None
        if mask is not None:
            result_mask = mask.transpose(array_order)
        else:
            result_mask = None
        # Reorder extra coords if needed.
        if extra_coords:
            reordered_extra_coords = []
            for coord in extra_coords:
                coord_list = list(coord)
                coord_list[1] = array_order[coord_list[1]]
                reordered_extra_coords.append(tuple(coord_list))

        super().__init__(result_data, result_wcs, uncertainty=result_uncertainty,
                         mask=result_mask, meta=meta, unit=unit,
                         extra_coords=reordered_extra_coords,
                         copy=copy, **kwargs)
