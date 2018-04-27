# -*- coding: utf-8 -*-

import abc
import warnings

import numpy as np
import astropy.nddata
import astropy.units as u
from astropy.utils.misc import InheritDocstrings

from ndcube import utils
from ndcube.utils.wcs import wcs_ivoa_mapping
from ndcube.mixins import NDCubeSlicingMixin, NDCubePlotMixin


__all__ = ['NDCubeBase', 'NDCube', 'NDCubeOrdered']


class NDCubeMetaClass(abc.ABCMeta, InheritDocstrings):
    """
    A metaclass that combines `abc.ABCMeta` and `~astropy.utils.misc.InheritDocstrings`.
    """


class NDCubeBase(astropy.nddata.NDData, metaclass=NDCubeMetaClass):

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

    # InheritDocstrings doesn't work on property methods.
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


class NDCube(NDCubeSlicingMixin, NDCubePlotMixin, astropy.nddata.NDArithmeticMixin, NDCubeBase):
    """
    Class representing N dimensional cubes.
    Extra arguments are passed on to `~astropy.nddata.NDData`.

    Parameters
    ----------
    data: `numpy.ndarray`
        The array holding the actual data in this object.

    wcs: `ndcube.wcs.wcs.WCS`
        The WCS object containing the axes' information

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

    missing_axis : `list` of `bool`
        Designates which axes in wcs object do not have a corresponding axis is the data.
        True means axis is "missing", False means axis corresponds to a data axis.
        Ordering corresponds to the axis ordering in the WCS object, i.e. reverse of data.
        For example, say the data's y-axis corresponds to latitude and x-axis corresponds
        to wavelength.  In order the convert the y-axis to latitude the WCS must contain
        a "missing" longitude axis as longitude and latitude are not separable.

    """

    def __init__(self, data, wcs, uncertainty=None, mask=None, meta=None,
                 unit=None, extra_coords=None, copy=False, missing_axis=None, **kwargs):
        if missing_axis is None:
            self.missing_axis = [False]*wcs.naxis
        else:
            self.missing_axis = missing_axis
        if data.ndim is not wcs.naxis:
            count = 0
            for bool_ in self.missing_axis:
                if not bool_:
                    count += 1
            if count is not data.ndim:
                raise ValueError("The number of data dimensions and number of "
                                 "wcs non-missing axes do not match.")
        # Format extra coords.
        if extra_coords:
            self._extra_coords_wcs_axis = \
              utils.cube._format_input_extra_coords_to_extra_coords_wcs_axis(
                  extra_coords, self.missing_axis, data.shape)
        else:
            self._extra_coords_wcs_axis = None
        # Initialize NDCube.
        super().__init__(data, wcs=wcs, uncertainty=uncertainty, mask=mask,
                         meta=meta, unit=unit, copy=copy, **kwargs)

    @property
    def dimensions(self):
        """
        Returns a named tuple with two attributes: 'shape' gives the shape
        of the data dimensions; 'axis_types' gives the WCS axis type of each dimension,
        e.g. WAVE or HPLT-TAN for wavelength of helioprojected latitude.

        """
        return u.Quantity(self.data.shape, unit=u.pix)

    @property
    def world_axis_physical_types(self):
        """
        Returns an iterable of strings describing the physical type for each world axis.

        The strings conform to the International Virtual Observatory Alliance
        standard, UCD1+ controlled Vocabulary.  For a description of the standard and
        definitions of the different strings and string components,
        see http://www.ivoa.net/documents/latest/UCDlist.html.

        """
        ctype = list(self.wcs.wcs.ctype)
        axes_ctype = []
        for i, axis in enumerate(self.missing_axis):
            if not axis:
                key = ctype[i]
                if "-TAN" in key:
                    key = key[:-4]
                axes_ctype.append(wcs_ivoa_mapping.get(key, default=None))
        return tuple(axes_ctype[::-1])

    def pixel_to_world(self, *quantity_axis_list):
        # The docstring is defined in NDDataBase

        origin = 0
        list_arg = []
        indexed_not_as_one = []
        result = []
        quantity_index = 0
        for i in range(len(self.missing_axis)):
            wcs_index = self.wcs.naxis-1-i
            # the cases where the wcs dimension was made 1 and the missing_axis is True
            if self.missing_axis[wcs_index]:
                list_arg.append(self.wcs.wcs.crpix[wcs_index]-1+origin)
            else:
                # else it is not the case where the dimension of wcs is 1.
                list_arg.append(quantity_axis_list[quantity_index].to(u.pix).value)
                quantity_index += 1
                # appending all the indexes to be returned in the answer
                indexed_not_as_one.append(wcs_index)
        list_arguments = list_arg[::-1]
        pixel_to_world = self.wcs.all_pix2world(*list_arguments, origin)
        # collecting all the needed answer in this list.
        for index in indexed_not_as_one[::-1]:
            result.append(u.Quantity(pixel_to_world[index], unit=self.wcs.wcs.cunit[index]))
        return result[::-1]

    def world_to_pixel(self, *quantity_axis_list):
        # The docstring is defined in NDDataBase

        origin = 0
        list_arg = []
        indexed_not_as_one = []
        result = []
        quantity_index = 0
        for i in range(len(self.missing_axis)):
            wcs_index = self.wcs.naxis-1-i
            # the cases where the wcs dimension was made 1 and the missing_axis is True
            if self.missing_axis[wcs_index]:
                list_arg.append(self.wcs.wcs.crval[wcs_index]+1-origin)
            else:
                # else it is not the case where the dimension of wcs is 1.
                list_arg.append(
                    quantity_axis_list[quantity_index].to(self.wcs.wcs.cunit[wcs_index]).value)
                quantity_index += 1
                # appending all the indexes to be returned in the answer
                indexed_not_as_one.append(wcs_index)
        list_arguments = list_arg[::-1]
        world_to_pixel = self.wcs.all_world2pix(*list_arguments, origin)
        # collecting all the needed answer in this list.
        for index in indexed_not_as_one[::-1]:
            result.append(u.Quantity(world_to_pixel[index], unit=u.pix))
        return result[::-1]

    def axis_world_coords(self, *axes):
        """
        Returns WCS coordinate values of all pixels for all axes.

        Parameters
        ----------
        axes: `int` or `str`, or multiple `int` or `str`
            Axis number in numpy ordering or unique substring of
            `~ndcube.NDCube.world_axis_physical_types`
            of axes for which real world coordinates are desired.
            axes=None implies all axes will be returned.

        Returns
        -------
        axes_coords: `list` of `astropy.units.Quantity`
            Real world coords for axes in order requested by user.

        Example
        -------
        >>> NDCube.all_world_coords(('lat', 'lon')) # doctest: +SKIP
        >>> NDCube.all_world_coords(2) # doctest: +SKIP

        """
        # Define the dimensions of the cube and the total number of axes.
        cube_dimensions = np.array(self.dimensions.value, dtype=int)
        n_dimensions = cube_dimensions.size
        world_axis_types = self.world_axis_physical_types
        # Determine axis numbers of user supplied axes.
        if axes == ():
            int_axes = np.arange(n_dimensions)
        else:
            if isinstance(axes, int):
                int_axes = np.array([axes])
            elif isinstance(axes, str):
                int_axes = np.array([
                    utils.cube.get_axis_number_from_axis_name(axes, world_axis_types)])
            else:
                int_axes = np.empty(len(axes), dtype=int)
                for i, axis in enumerate(axes):
                    if isinstance(axis, int):
                        int_axes[i] = axis
                    elif isinstance(axis, str):
                        int_axes[i] = utils.cube.get_axis_number_from_axis_name(
                            axis, world_axis_types)
        # Ensure user has not entered the same axis twice.
        repeats = set([x for x in int_axes if np.where(int_axes == x)[0].size > 1])
        if repeats:
            raise ValueError("The following axes were specified more than once: {}".format(
                ' '.join(map(str, repeats))))
        n_axes = len(int_axes)
        axes_coords = [None] * n_axes
        axes_translated = np.zeros_like(int_axes, dtype=bool)
        # Determine which axes are dependent on others.
        # Ensure the axes are in numerical order.
        dependent_axes = [list(utils.wcs.get_dependent_data_axes(self.wcs, axis, self.missing_axis))
                          for axis in int_axes]
        n_dependent_axes = [len(da) for da in dependent_axes]
        # Iterate through each axis and perform WCS translation.
        for i, axis in enumerate(int_axes):
            # If axis has already been translated, do not do so again.
            if not axes_translated[i]:
                if n_dependent_axes[i] == 1:
                    # Construct pixel quantities in each dimension letting
                    # other dimensions all have 0 pixel value.
                    quantity_list = [
                        u.Quantity(np.zeros(cube_dimensions[dependent_axes[i]]),
                                   unit=u.pix)] * n_dimensions
                    # Replace array in quantity list corresponding to current axis with
                    # np.arange array.
                    quantity_list[axis] = u.Quantity(np.arange(cube_dimensions[axis]), unit=u.pix)
                else:
                    # If the axis is dependent on another, perform
                    # translations on all dependent axes.
                    # Construct pixel quantities in each dimension letting
                    # other dimensions all have 0 pixel value.
                    quantity_list = [u.Quantity(np.zeros(tuple(
                        [cube_dimensions[k] for k in dependent_axes[i]])),
                        unit=u.pix)] * n_dimensions
                    # Construct orthogonal pixel index arrays for dependent axes.
                    dependent_pixel_quantities = np.meshgrid(
                        *[np.arange(cube_dimensions[k]) * u.pix
                          for k in dependent_axes[i]], indexing="ij")
                    for k, axis in enumerate(dependent_axes[i]):
                        quantity_list[axis] = dependent_pixel_quantities[k]
                # Perform wcs translation
                dependent_axes_coords = self.pixel_to_world(*quantity_list)
                # Place world coords into output list
                for dependent_axis in dependent_axes[i]:
                    if dependent_axis in int_axes:
                        # Due to error check above we know dependent
                        # axis can appear in int_axes at most once.
                        j = np.where(int_axes == dependent_axis)[0][0]
                        axes_coords[j] = dependent_axes_coords[dependent_axis]
                        # Remove axis from list that have now been translated.
                        axes_translated[j] = True
        if len(axes_coords) == 1:
            return axes_coords[0]
        else:
            return tuple(axes_coords)

    @property
    def extra_coords(self):
        """
        Dictionary of extra coords where each key is the name of an extra
        coordinate supplied by user during instantiation of the NDCube.

        The value of each key is itself a dictionary with the following keys:
          | 'axis': `int`
          |     The number of the data axis to which the extra coordinate corresponds.
          | 'value': `astropy.units.Quantity` or array-like
          |     The value of the extra coordinate at each pixel/array element along the
          |     corresponding axis (given by the 'axis' key, above).  Note this means
          |     that the length of 'value' must be equal to the length of the data axis
          |     to which is corresponds.
        """

        if not self._extra_coords_wcs_axis:
            result = None
        else:
            result = {}
            for key in list(self._extra_coords_wcs_axis.keys()):
                result[key] = {
                    "axis": utils.cube.wcs_axis_to_data_axis(
                        self._extra_coords_wcs_axis[key]["wcs axis"],
                        self.missing_axis),
                    "value": self._extra_coords_wcs_axis[key]["value"]}
        return result

    def crop_by_coords(self, lower_corner, interval_widths=None, upper_corner=None, units=None):
        # The docstring is defined in NDDataBase

        n_dim = self.data.ndim
        # Raising a value error if the arguments have not the same dimensions.
        # Calculation of upper_corner with the inputing interval_widths
        # This part of the code will be removed in version 2.0
        if interval_widths:
            warnings.warn("interval_widths will be removed from the API in "
                          "version 2.0, please use upper_corner argument.")
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
            raise ValueError("lower_corner and upper_corner must have same"
                             "number of elements as number of data dimensions.")
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
        all_world_corners = [all_world_corners_grid[i].flatten()*lower_corner[i].unit
                             for i in range(n_dim)]
        # Convert to pixel coordinates
        all_pix_corners = self.world_to_pixel(*all_world_corners)
        # Derive slicing item with which to slice NDCube.
        # Be sure to round down min pixel and round up + 1 the max pixel.
        item = tuple([slice(int(np.clip(axis_pixels.value.min(), 0, None)),
                            int(np.ceil(axis_pixels.value.max()))+1)
                      for axis_pixels in all_pix_corners])
        return self[item]

    def crop_by_extra_coord(self, min_coord_value, interval_width, coord_name):
        """
        Crops an NDCube given a minimum value and interval width along an extra coord.

        Parameters
        ----------
        min_coord_value: Single value `astropy.units.Quantity`
            The minimum desired value of the extra coord after cropping.
            Unit must be consistent with the extra coord on which cropping is based.

        interval_width: Single value `astropy.units.Quantity`
            The width of the interval along the extra coord axis in physical units
            consistent with the extra coord.  Unit must be consistent with the extra
            coord on which cropping is based.

        extra_coord: `str`
            Name of extra coordinate.

        Returns
        -------
        result: `ndcube.NDCube`

        """
        extra_coord_dict = self.extra_coords[coord_name]
        if isinstance(extra_coord_dict["value"], u.Quantity):
            extra_coord_values = extra_coord_dict["value"]
        else:
            extra_coord_values = np.asarray(extra_coord_dict["value"])
        w = np.logical_and(extra_coord_values >= min_coord_value,
                           extra_coord_values < min_coord_value + interval_width)
        w = np.arange(len(extra_coord_values))[w]
        item = [slice(None)]*len(self.dimensions)
        item[extra_coord_dict["axis"]] = slice(w[0], w[1]+1)
        return self[tuple(item)]

    def __repr__(self):
        return (
            """NDCube
---------------------
{wcs}
---------------------
Length of NDCube: {lengthNDCube}
Axis Types of NDCube: {axis_type}
""".format(wcs=self.wcs.__repr__(), lengthNDCube=self.dimensions,
           axis_type=self.world_axis_physical_types))


class NDCubeOrdered(NDCube):
    """
    Class representing N dimensional cubes with oriented WCS.
    Extra arguments are passed on to NDData's init.
    The order is TIME, SPECTRAL, SOLAR-x, SOLAR-Y and any other dimension.
    For example, in an x, y, t cube the order would be (t,x,y) and in a
    lambda, t, y cube the order will be (t, lambda, y).
    Extra arguments are passed on to NDData's init.

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
                 unit=None, extra_coords=None, copy=False, missing_axis=None, **kwargs):
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
                         copy=copy, missing_axis=missing_axis, **kwargs)
