from sunpy.visualization.imageanimator import ImageAnimatorWCS
from collections import namedtuple
import matplotlib.pyplot as plt
import sunpy.visualization.wcsaxes_compat as wcsaxes_compat
import astropy.units as u
import sunpycube.cube.cube_utils
import sunpycube.wcs_util
import astropy.nddata
import numpy as np
import copy

DimensionPair = namedtuple('DimensionPair', 'lengths axis_types')

__all__ = ['NDCube']


class NDCube(astropy.nddata.NDData):
    """
    Class representing N dimensional cubes.
    Extra arguments are passed on to NDData's init.

    Attributes
    ----------
    data: `numpy.ndarray`
        The array holding the actual data in this object.

    wcs: `sunpycube.wcs.wcs.WCS`
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

    copy : bool, optional
        Indicates whether to save the arguments as copy. True copies every attribute
        before saving it while False tries to save every parameter as reference.
        Note however that it is not always possible to save the input as reference. Default is False.
    """

    def __init__(self, data, uncertainty=None, mask=None, wcs=None, meta=None,
                 unit=None, copy=False, missing_axis=None, **kwargs):
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
                raise ValueError(
                    "The number of data dimensions and number of wcs non-missing axes do not match.")
        super(NDCube, self).__init__(data, uncertainty=uncertainty, mask=mask,
                                     wcs=wcs, meta=meta, unit=unit, copy=copy, **kwargs)

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
        for i, _ in enumerate(self.missing_axis):
            # the cases where the wcs dimension was made 1 and the missing_axis is True
            if self.missing_axis[self.wcs.naxis-1-i]:
                list_arg.append(self.wcs.wcs.crpix[self.wcs.naxis-1-i]-1+origin)
            else:
                # else it is not the case where the dimension of wcs is 1.
                list_arg.append(quantity_axis_list[quantity_index])
                quantity_index += 1
            # appending all the indexes to be returned in the answer
                indexed_not_as_one.append(self.wcs.naxis-1-i)
        list_arguemnts = list_arg[::-1]
        pixel_to_world = self.wcs.all_pix2world(*list_arguemnts, origin)
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
        for i, _ in enumerate(self.missing_axis):
            # the cases where the wcs dimension was made 1 and the missing_axis is True
            if self.missing_axis[self.wcs.naxis-1-i]:
                list_arg.append(self.wcs.wcs.crval[self.wcs.naxis-1-i]+1-origin)
            else:
                # else it is not the case where the dimension of wcs is 1.
                list_arg.append(quantity_axis_list[quantity_index])
                quantity_index += 1
            # appending all the indexes to be returned in the answer
                indexed_not_as_one.append(self.wcs.naxis-1-i)
        list_arguemnts = list_arg[::-1]
        world_to_pixel = self.wcs.all_world2pix(*list_arguemnts, origin)
        # collecting all the needed answer in this list.
        for index in indexed_not_as_one[::-1]:
            result.append(u.Quantity(world_to_pixel[index], unit=u.pix))
        return result[::-1]

    def to_sunpy(self):
        pass

    @property
    def dimensions(self):
        """
        Returns a named tuple with two attributes: 'lengths' gives the lengths
        of the data dimensions; 'axis_types' gives the WCS axis type of each dimension,
        e.g. WAVE or HPLT-TAN for wavelength of helioprojected latitude.
        """
        ctype = list(self.wcs.wcs.ctype)
        axes_ctype = []
        for i, axis in enumerate(self.missing_axis):
            if not axis:
                axes_ctype.append(ctype[i])
        shape = u.Quantity(self.data.shape, unit=u.pix)
        return DimensionPair(lengths=shape, axis_types=axes_ctype[::-1])

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
            plot = _plot_3D_cube(self, image_axes=axis_data, unit_x_axis=unit_x_axis, unit_y_axis=unit_y_axis,
                                 axis_ranges=axis_ranges, *kwargs)
        elif self.data.ndim is 2:
            plot = _plot_2D_cube(self, axes=axes, image_axes=axis_data, **kwargs)
        elif self.data.ndim is 1:
            plot = _plot_1D_cube(self, unit=unit, origin=origin)
        return plot

    def __getitem__(self, item):
        if item is None or (isinstance(item, tuple) and None in item):
            raise IndexError("None indices not supported")
        data = self.data[item]
        # here missing axis is reversed as the item comes already in the reverse order
        # of the input
        wcs, missing_axis = sunpycube.wcs_util._wcs_slicer(
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
        result = NDCube(data, wcs=wcs, mask=mask, uncertainty=uncertainty,
                        meta=self.meta, unit=self.unit, copy=False, missing_axis=missing_axis)
        return result


class NDCubeOrdered(NDCube):
    """
    Class representing N dimensional cubes with oriented WCS.
    Extra arguments are passed on to NDData's init.
    The order is TIME, SPECTRAL, SOLAR-x, SOLAR-Y and any other dimension.
    For example, in an x, y, t cube the order would be (t,x,y) and in a
    lambda, t, y cube the order will be (t, lambda, y).
    Extra arguments are passed on to NDData's init.

    Attributes
    ----------
    data: `numpy.ndarray`
        The array holding the actual data in this object.

    wcs: `sunpycube.wcs.wcs.WCS`
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
        Note however that it is not always possible to save the input as reference. Default is False.
    """

    def __init__(self, data, uncertainty=None, mask=None, wcs=None, meta=None,
                 unit=None, copy=False, missing_axis=None, **kwargs):
        axtypes = list(wcs.wcs.ctype)
        array_order = sunpycube.cube.cube_utils.select_order(axtypes)
        result_data = data.transpose(array_order)
        wcs_order = np.array(array_order)[::-1]
        result_wcs = sunpycube.wcs_util.reindex_wcs(wcs, wcs_order)
        super(NDCubeOrdered, self).__init__(result_data, uncertainty=uncertainty, mask=mask,
                                            wcs=result_wcs, meta=meta, unit=unit, copy=copy, missing_axis=missing_axis, **kwargs)


def _plot_3D_cube(cube, image_axes=None, unit_x_axis=None, unit_y_axis=None,
                  axis_ranges=None, **kwargs):
    """
    Plots an interactive visualization of this cube using sliders to move through axes
    plot using in the image.
    Parameters other than data and wcs are passed to ImageAnimatorWCS, which in turn
    passes them to imshow.

    Parameters
    ----------
    image_axes: `list`
        The two axes that make the image.
        Like [-1,-2] this implies cube instance -1 dimension
        will be x-axis and -2 dimension will be y-axis.

    unit_x_axis: `astropy.units.Unit`
        The unit of x axis.

    unit_y_axis: `astropy.units.Unit`
        The unit of y axis.

    axis_ranges: `list` of physical coordinates for array or None
        If None array indices will be used for all axes.
        If a list it should contain one element for each axis of the numpy array.
        For the image axes a [min, max] pair should be specified which will be
        passed to :func:`matplotlib.pyplot.imshow` as extent.
        For the slider axes a [min, max] pair can be specified or an array the
        same length as the axis which will provide all values for that slider.
        If None is specified for an axis then the array indices will be used
        for that axis.
    """
    i = ImageAnimatorWCS(cube.data, wcs=cube.wcs, unit_x_axis=unit_x_axis, unit_y_axis=unit_y_axis,
                         axis_ranges=axis_ranges, **kwargs)
    return i


def _plot_2D_cube(cube, axes=None, image_axes=['x', 'y'], **kwargs):
    """
    Plots a 2D image onto the current
    axes. Keyword arguments are passed on to matplotlib.

    Parameters
    ----------
    axes: `astropy.visualization.wcsaxes.core.WCSAxes` or `None`:
        The axes to plot onto. If None the current axes will be used.

    image_axes: `list`.
        The first axis in WCS object will become the first axis of image_axes and
        second axis in WCS object will become the seconf axis of image_axes.
    """
    if axes is None:
        if cube.wcs.naxis is not 2:
            missing_axis = cube.missing_axis
            slice_list = []
            axis_index = []
            index = 0
            for i, bool_ in enumerate(missing_axis):
                if not bool_:
                    slice_list.append(image_axes[index])
                    index += 1
                else:
                    slice_list.append(1)
            if index is not 2:
                raise ValueError("Dimensions of WCS and data don't match")
        axes = wcsaxes_compat.gca_wcs(cube.wcs, slices=slice_list)
    plot = axes.imshow(cube.data, **kwargs)
    return plot


def _plot_1D_cube(cube, unit=None, origin=0):
    """
    Plots a graph.
    Keyword arguments are passed on to matplotlib.

    Parameters
    ----------
    unit: `astropy.unit.Unit`
        The data is changed to the unit given or the cube.unit if not given.
    """
    index_not_one = []
    for i, _bool in enumerate(cube.missing_axis):
        if _bool:
            index_not_one.append(i)
    if unit is None:
        unit = cube.wcs.wcs.cunit[index_not_one[0]]
    plot = plt.plot(cube.pixel_to_world(
        [u.Quantity(np.arange(cube.data.shape[0]), unit=u.pix)], origin=origin)[0].to(unit), cube.data)
    return plot
