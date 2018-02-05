# -*- coding: utf-8 -*-
# Author: Ankit Baruah and Daniel Ryan <ryand5@tcd.ie>

import copy

import astropy.units as u
import astropy.nddata
import numpy as np
import matplotlib.pyplot as plt
import sunpy.map
from sunpy.map.mapcube import MapCube

from ndcube import cube_utils
from ndcube import wcs_util
from ndcube import DimensionPair, SequenceDimensionPair
from ndcube.visualization import animation as ani
from ndcube.plot_helpers import _plot_1D_cube, _plot_2D_cube, _plot_3D_cube


__all__ = ['NDCube', 'NDCubeOrdered']


class NDCube(astropy.nddata.NDArithmeticMixin, astropy.nddata.NDData):
    """
    Class representing N dimensional cubes.
    Extra arguments are passed on to NDData's init.

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
            self._extra_coords_wcs_axis = _format_input_extra_coords_to_extra_coords_wcs_axis(
                extra_coords, self.missing_axis, data.shape)
        else:
            self._extra_coords_wcs_axis = None
        # Initialize NDCube.
        super(NDCube, self).__init__(data, wcs=wcs, uncertainty=uncertainty, mask=mask,
                                     meta=meta, unit=unit, copy=copy, **kwargs)

    def pixel_to_world(self, quantity_axis_list, origin=0):
        """
        Convert a pixel coordinate to a data (world) coordinate by using
        `~astropy.wcs.WCS.all_pix2world`.

        Parameters
        ----------
        quantity_axis_list : `list`
            A list of `~astropy.units.Quantity` with unit as pixel `pix`.

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

    def world_to_pixel(self, quantity_axis_list, origin=0):
        """
        Convert a world coordinate to a data (pixel) coordinate by using
        `~astropy.wcs.WCS.all_world2pix`.

        Parameters
        ----------
        quantity_axis_list : `list`
            A list of `~astropy.units.Quantity`.

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

    def to_sunpy(self):
        wcs_axes = list(self.wcs.wcs.ctype)
        missing_axis = self.missing_axis
        index_not_one = []
        if 'TIME' in wcs_axes and len(self.dimensions.shape) is 1:
            result = self.pixel_to_world([u.Quantity(self.data, unit=u.pix)])
        elif 'HPLT-TAN' in wcs_axes and 'HPLN-TAN' in wcs_axes \
                and len(self.dimensions.shape) is 2:
            if not missing_axis[wcs_axes.index("HPLT-TAN")] \
                    and not missing_axis[wcs_axes.index("HPLN-TAN")]:
                result = sunpy.map.Map(self.data, self.meta)
        else:
            raise NotImplementedError("Object type not Implemented")
        return result

    @property
    def dimensions(self):
        """
        Returns a named tuple with two attributes: 'shape' gives the shape
        of the data dimensions; 'axis_types' gives the WCS axis type of each dimension,
        e.g. WAVE or HPLT-TAN for wavelength of helioprojected latitude.
        """
        ctype = list(self.wcs.wcs.ctype)
        axes_ctype = []
        for i, axis in enumerate(self.missing_axis):
            if not axis:
                axes_ctype.append(ctype[i])
        shape = u.Quantity(self.data.shape, unit=u.pix)
        return DimensionPair(shape=shape, axis_types=axes_ctype[::-1])

    def plot(self, axes=None, image_axes=[-1, -2], unit_x_axis=None, unit_y_axis=None,
             axis_ranges=None, unit=None, origin=0, **kwargs):
        """
        Plots an interactive visualization of this cube with a slider
        controlling the wavelength axis for data having dimensions greater than 2.
        Plots an x-y graph onto the current axes for 2D or 1D data. Keyword arguments are passed
        on to matplotlib.
        Parameters other than data and wcs are passed to ImageAnimatorWCS, which in turn
        passes them to imshow for data greater than 2D.

        Parameters
        ----------
        image_axes: `list`
            The two axes that make the image.
            Like [-1,-2] this implies cube instance -1 dimension
            will be x-axis and -2 dimension will be y-axis.

        axes: `astropy.visualization.wcsaxes.core.WCSAxes` or None:
            The axes to plot onto. If None the current axes will be used.

        unit_x_axis: `astropy.units.Unit`
            The unit of x axis for 2D plots.

        unit_y_axis: `astropy.units.Unit`
            The unit of y axis for 2D plots.

        unit: `astropy.unit.Unit`
            The data is changed to the unit given or the cube.unit if not given, for 1D plots.

        axis_ranges: list of physical coordinates for array or None
            If None array indices will be used for all axes.
            If a list it should contain one element for each axis of the numpy array.
            For the image axes a [min, max] pair should be specified which will be
            passed to :func:`matplotlib.pyplot.imshow` as extent.
            For the slider axes a [min, max] pair can be specified or an array the
            same length as the axis which will provide all values for that slider.
            If None is specified for an axis then the array indices will be used
            for that axis.
        """
        axis_data = ['x' for i in range(2)]
        axis_data[image_axes[1]] = 'y'
        if self.data.ndim >= 3:
            plot = _plot_3D_cube(self, image_axes=image_axes, unit_x_axis=unit_x_axis,
                                 unit_y_axis=unit_y_axis, axis_ranges=axis_ranges, **kwargs)
        elif self.data.ndim is 2:
            plot = _plot_2D_cube(self, axes=axes, image_axes=axis_data[::-1], **kwargs)
        elif self.data.ndim is 1:
            plot = _plot_1D_cube(self, unit=unit, origin=origin)
        return plot

    def crop_by_coords(self, lower_left_corner, dimension_widths):
        """
        Crops an NDCube given a lower left corner and widths of region of interest.

        Parameters
        ----------
        lower_left_corner: `list` of `astropy.units.Quantity`
            The lower left corner of the region of interest described in physical units
            consistent with the NDCube's wcs object.  The length of the iterable must
            equal the number of data dimensions and must have the same order as the data.

        dimension_widths: iterable of `astropy.units.Quantity`
            The width of the region of interest in each dimension in physical units
            consistent with the NDCube's wcs object.  The length of the iterable must
            equal the number of data dimensions and must have the same order as the data.

        Returns
        -------
        result: NDCube

        """
        n_dim = len(self.dimensions.shape)
        if len(lower_left_corner) != len(dimension_widths) != n_dim:
            raise ValueError("lower_left_corner and dimension_widths must have "
                             "same number of elements as number of data dimensions.")
        # Convert coords of lower left corner to pixel units.
        lower_pixels = self.world_to_pixel(lower_left_corner)
        upper_pixels = self.world_to_pixel([lower_left_corner[i]+dimension_widths[i]
                                            for i in range(n_dim)])
        # Round pixel values to nearest integer.
        lower_pixels = [int(np.rint(l.value)) for l in lower_pixels]
        upper_pixels = [int(np.rint(u.value)) for u in upper_pixels]
        slic = tuple([slice(lower_pixels[i], upper_pixels[i]) for i in range(n_dim)])
        return self[slic]

    @property
    def extra_coords(self):
        if not self._extra_coords_wcs_axis:
            result = None
        else:
            result = {}
            for key in list(self._extra_coords_wcs_axis.keys()):
                result[key] = {
                    "axis": _wcs_axis_to_data_axis(self._extra_coords_wcs_axis[key]["wcs axis"],
                                                   self.missing_axis),
                    "value": self._extra_coords_wcs_axis[key]["value"]}
        return result

    def __getitem__(self, item):
        if item is None or (isinstance(item, tuple) and None in item):
            raise IndexError("None indices not supported")
        data = self.data[item]
        # here missing axis is reversed as the item comes already in the reverse order
        # of the input
        wcs, missing_axis = wcs_util._wcs_slicer(
            self.wcs, copy.deepcopy(self.missing_axis[::-1]), item)
        if self.mask is not None:
            mask = self.mask[item]
        else:
            mask = None
        if self.uncertainty is not None:
            if isinstance(self.uncertainty.array, np.ndarray):
                if self.uncertainty.array.shape == self.data.shape:
                    uncertainty = self.uncertainty[item]
                else:
                    uncertainty = self.uncertainty
            else:
                uncertainty = self.uncertainty
        else:
            uncertainty = None
        if self.extra_coords is None:
            new_extra_coords_dict = None
        else:
            old_extra_coords = self.extra_coords
            extra_coords_keys = list(old_extra_coords.keys())
            new_extra_coords = copy.deepcopy(self._extra_coords_wcs_axis)
            for ck in extra_coords_keys:
                axis_ck = old_extra_coords[ck]["axis"]
                if isinstance(item, (slice, int)):
                    if axis_ck == 0:
                        new_extra_coords[ck]["value"] = new_extra_coords[ck]["value"][item]
                if isinstance(item, tuple):
                    try:
                        slice_item_extra_coords = item[axis_ck]
                        new_extra_coords[ck]["value"] = \
                            new_extra_coords[ck]["value"][slice_item_extra_coords]
                    except IndexError:
                        pass
                    except TypeError:
                        pass
                new_extra_coords_dict = _extra_coords_to_input_format(new_extra_coords,
                                                                      missing_axis)
        return NDCube(data, wcs=wcs, mask=mask, uncertainty=uncertainty, meta=self.meta,
                      unit=self.unit, copy=False, missing_axis=missing_axis,
                      extra_coords=new_extra_coords_dict)

    def __repr__(self):
        return (
            """Sunpy NDCube
---------------------
{wcs}
---------------------
Length of NDCube: {lengthNDCube}
Axis Types of NDCube: {axis_type}
""".format(wcs=self.wcs.__repr__(), lengthNDCube=self.dimensions[0], axis_type=self.dimensions[1]))


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
                 unit=None, copy=False, missing_axis=None, **kwargs):
        axtypes = list(wcs.wcs.ctype)
        array_order = cube_utils.select_order(axtypes)
        result_data = data.transpose(array_order)
        wcs_order = np.array(array_order)[::-1]
        result_wcs = wcs_util.reindex_wcs(wcs, wcs_order)
        super(NDCubeOrdered, self).__init__(result_data, result_wcs, uncertainty=uncertainty,
                                            mask=mask, meta=meta, unit=unit, copy=copy,
                                            missing_axis=missing_axis, **kwargs)


def _extra_coords_to_input_format(extra_coords, missing_axis):
    """
    Converts NDCube.extra_coords attribute to format required as input for new NDCube.

    Paramaters
    ----------
    extra_coords: dict
        An NDCube.extra_coords instance.

    Returns
    -------
    input_format: `list`
        Infomation on extra coords in format required by `NDCube.__init__`.

    """
    coord_names = list(extra_coords.keys())
    result = []
    for name in coord_names:
        coord_keys = list(extra_coords[name].keys())
        if "wcs axis" in coord_keys and "axis" not in coord_keys:
            axis = _wcs_axis_to_data_axis(extra_coords[name]["wcs axis"], missing_axis)
        elif "axis" in coord_keys and "wcs axis" not in coord_keys:
            axis = extra_coords[name]["axis"]
        else:
            raise KeyError("extra coords dict can have keys 'wcs axis' or 'axis'.  Not both.")
        result.append((name, axis, extra_coords[name]["value"]))
    return result


def _wcs_axis_to_data_axis(wcs_axis, missing_axis):
    if wcs_axis is None:
        result = None
    else:
        if missing_axis[wcs_axis]:
            result = None
        else:
            data_ordered_wcs_axis = len(missing_axis)-wcs_axis-1
            result = data_ordered_wcs_axis-sum(missing_axis[::-1][:data_ordered_wcs_axis])
    return result


def _data_axis_to_wcs_axis(data_axis, missing_axis):
    if data_axis is None:
        result = None
    else:
        result = len(missing_axis)-np.where(np.cumsum(
            [b is False for b in missing_axis][::-1]) == data_axis+1)[0][0]-1
    return result


def _format_input_extra_coords_to_extra_coords_wcs_axis(extra_coords, missing_axis, data_shape):
    extra_coords_wcs_axis = {}
    coord_format_error = "Coord must have three properties supplied, " + \
                         "name (str), axis (int), values (Quantity or array-like)." + \
                         " Input coord: {0}"
    coord_0_format_error = "1st element of extra coordinate tuple must be a " \
                           "string giving the coordinate's name."
    coord_1_format_error = "2nd element of extra coordinate tuple must be None " \
                           "or an int giving the data axis " \
                           "to which the coordinate corresponds."
    coord_len_error = "extra coord ({0}) must have same length as data axis " + \
                      "to which it is assigned: coord length, {1} != data axis length, {2}"
    for coord in extra_coords:
        # Check extra coord has the right number and types of info.
        if len(coord) != 3:
            raise ValueError(coord_format_error.format(coord))
        if not isinstance(coord[0], str):
            raise ValueError(coord_0_format_error.format(coord))
        if coord[1] is not None and not isinstance(coord[1], int) and \
                not isinstance(coord[1], np.int64):
            raise ValueError(coord_1_format_error)
        # Unless extra coord corresponds to a missing axis, check length
        # of coord is same is data axis to which is corresponds.
        if coord[1] is not None:
            if not missing_axis[::-1][coord[1]]:

                if len(coord[2]) != data_shape[coord[1]]:
                    raise ValueError(coord_len_error.format(coord[0], len(coord[2]),
                                                            data_shape[coord[1]]))
        # Determine wcs axis corresponding to data axis of coord
        extra_coords_wcs_axis[coord[0]] = {
            "wcs axis": _data_axis_to_wcs_axis(coord[1], missing_axis),
            "value": coord[2]}
    return extra_coords_wcs_axis
