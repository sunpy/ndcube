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

PixelPair = namedtuple('PixelPair', 'dimensions axes')

__all__ = ['NDCube', 'Cube2D', 'Cube1D']


class NDCube(astropy.nddata.NDData):
    """docstring for NDCube"""

    def __init__(self, data, uncertainty=None, mask=None, wcs=None, meta=None, unit=None, copy=False, missing_axis=None, **kwargs):
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
        origin : `int`
            Origin of the top-left corner. i.e. count from 0 or 1.
            Normally, origin should be 0 when passing numpy indices, or 1 if
            passing values from FITS header or map attributes.
            See `~astropy.wcs.WCS.wcs_pix2world` for more information.

        quantity_axis_list : `list`
            A list of `~astropy.units.Quantity`.

        Returns
        -------

        coord : `list`
            A list of arrays containing the output coordinates.

        """
        list_arg = []
        indexed_not_as_one = []
        result = []
        quantity_index = 0
        for i, _ in enumerate(self.missing_axis):
            # the cases where the wcs dimension was made 1 and the missing_axis is True
            if self.missing_axis[self.wcs.naxis-1-i]:
                list_arg.append(self.wcs.wcs.crpix[i]-1+origin)
            else:
                # else it is not the case where the dimension of wcs is 1.
                list_arg.append(quantity_axis_list[quantity_index])
                quantity_index += 1
            # appending all the indexes to be returned in the answer
                indexed_not_as_one.append(i)
        pixel_to_world = self.wcs.all_pix2world(*list_arg, origin)
        # collecting all the needed answer in this list.
        for index in indexed_not_as_one:
            result.append(u.Quantity(pixel_to_world[index], unit=self.wcs.wcs.cunit[index]))
        return result

    def world_to_pixel(self, quantity_axis_list, origin=0):
        list_arg = []
        indexed_not_as_one = []
        result = []
        quantity_index = 0
        for i, _ in enumerate(self.missing_axis):
            # the cases where the wcs dimension was made 1 and the missing_axis is True
            if self.missing_axis[self.wcs.naxis-1-i]:
                list_arg.append(self.wcs.wcs.crval[i]+1-origin)
            else:
                # else it is not the case where the dimension of wcs is 1.
                list_arg.append(quantity_axis_list[quantity_index])
                quantity_index += 1
            # appending all the indexes to be returned in the answer
                indexed_not_as_one.append(i)
        world_to_pixel = self.wcs.all_world2pix(*list_arg, origin)
        # collecting all the needed answer in this list.
        for index in indexed_not_as_one:
            result.append(u.Quantity(world_to_pixel[index], unit=self.wcs.wcs.cunit[index]))
        return result

    def to_sunpy(self):
        pass

    @property
    def dimensions(self):
        """
        The dimensions of the data (x axis first, y axis second, z axis third ...so on) and the type of axes.
        """
        ctype = list(self.wcs.wcs.ctype)[::-1]
        axes_ctype = []
        for i, axis in enumerate(self.missing_axis):
            if not axis:
                axes_ctype.append(ctype[i])
        shape = self.data.shape
        return PixelPair(dimensions=shape, axes=axes_ctype)

    def plot(self, axes=None, axis_data=['x', 'y'], unit=None, origin=0, *args, **kwargs):
        if self.data.ndim >= 3:
            plot = _plot_3D_cube(self, *args, *kwargs)
        elif self.data.ndim is 2:
            plot = _plot_2D_cube(self, axes=axes, axis_data=axis_data, **kwargs)
        elif self.data.ndim is 1:
            plot = _plot_1D_cube(self, unit=unit, origin=origin)
        return plot

    def __getitem__(self, item):
        if item is None or (isinstance(item, tuple) and None in item):
            raise IndexError("None indices not supported")
        data = self.data[item]
        wcs, missing_axis = sunpycube.wcs_util._wcs_slicer(
            self.wcs, copy.deepcopy(self.missing_axis), item)
        if self.mask is not None:
            mask = self.mask[item]
        else:
            mask = None
        result = NDCube(data, wcs=wcs, mask=mask, uncertainty=self.uncertainty,
                        meta=self.meta, unit=self.unit, copy=False, missing_axis=missing_axis)
        return result


class NDCubeOrdered(NDCube):
    """docstring for NDCubeOrdered"""

    def __init__(self, data, uncertainty=None, mask=None, wcs=None, meta=None, unit=None, copy=False, missing_axis=None, **kwargs):
        axtypes = list(wcs.wcs.ctype)
        array_order = sunpycube.cube.cube_utils.select_order(axtypes)
        result_data = data.transpose(array_order)
        wcs_order = np.array(array_order)[::-1]
        result_wcs = sunpycube.wcs_util.reindex_wcs(wcs, wcs_order)
        super(NDCubeOrdered, self).__init__(result_data, uncertainty=uncertainty, mask=mask,
                                            wcs=result_wcs, meta=meta, unit=unit, copy=copy, missing_axis=missing_axis, **kwargs)


def _plot_3D_cube(cube, *args, **kwargs):
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
    i = ImageAnimatorWCS(cube.data, wcs=cube.wcs, *args, **kwargs)
    return i


def _plot_2D_cube(cube, axes=None, axis_data=['x', 'y'], **kwargs):
    """
    Plots an x-y graph at a certain specified wavelength onto the current
    axes. Keyword arguments are passed on to matplotlib.

    Parameters
    ----------
    axes: `astropy.visualization.wcsaxes.core.WCSAxes` or `None`:
        The axes to plot onto. If None the current axes will be used.

    axis_data: `list`.
        The first axis in WCS object will become the first axis of axis_data and
        second axis in WCS object will become the seconf axis of axis_data.
    """
    if axes is None:
        if cube.wcs.naxis is not 2:
            missing_axis = cube.missing_axis
            slice_list = []
            axis_index = []
            index = 0
            for i, bool_ in enumerate(missing_axis):
                if not bool_:
                    slice_list.append(axis_data[index])
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
