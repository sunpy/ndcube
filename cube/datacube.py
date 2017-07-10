# -*- coding: utf-8 -*-
# Author: Mateo Inchaurrandieta <mateo.inchaurrandieta@gmail.com>
# pylint: disable=E1101, E0611
"""
Main class for representing cubes - 3D sets of continuous data by time and/or
wavelength
"""
# NOTE: This module uses version 1.02 of "Time coordinates in FITS" by
# Rots et al, available at http://hea-www.cfa.harvard.edu/~arots/TimeWCS/
# This draft standard may change.

# standard libraries
import datetime

# external libraries
import numpy as np
import matplotlib.pyplot as plt
import astropy.nddata
import copy
from astropy import units as u
from astropy.units import sday  # sidereal day

# Sunpy modules
from sunpy.map import GenericMap
try:
    from sunpy.util.metadata import MetaDict
except ImportError:
    from sunpy.map import MapMeta as MetaDict
import sunpy.visualization.wcsaxes_compat as wcsaxes_compat
from sunpy.visualization.imageanimator import ImageAnimatorWCS
from sunpy.lightcurve import LightCurve
from sunpycube.spectra.spectrum import Spectrum
from sunpycube.spectra.spectrogram import Spectrogram
from sunpycube.spectra.spectral_cube import SpectralCube
from sunpycube.cube import cube_utils as cu
from sunpycube.visualization import animation as ani
from sunpycube import wcs_util as wu

__all__ = ['Cube', 'CubeSequence']
# TODO: use uncertainties in all calculations and conversions


class Cube(astropy.nddata.NDDataArray):
    """
    Class representing cubes.
    Extra arguments are passed on to NDDataArray's init.

    Attributes
    ----------
    data: numpy ndarray
        The spectral cube holding the actual data in this object. The axes'
        priorities are time, spectral, celestial. This means that if
        present, each of these axis will take precedence over the others.
        For example, in an x, y, t cube the order would be (t,x,y) and in a
        lambda, t, y cube the order will be (t, lambda, y).

    axes_wcs: sunpy.wcs.wcs.WCS object
        The WCS object containing the axes' information
    errors: numpy ndarray
        one-sigma errors for the data. If the error array is present, there
        should also be a mask keyword argument
    """

    def __init__(self, data, wcs, errors=None, **kwargs):
        mask = kwargs.pop('mask', np.zeros(data.shape, dtype=bool))
        if errors is not None:
            data, wcs, err_array, mask = cu.orient(data, wcs, errors.array,
                                                   mask)
            errors.array = err_array
            kwargs.update({'uncertainty': errors})
        else:
            data, wcs, mask = cu.orient(data, wcs, mask)
        astropy.nddata.NDDataArray.__init__(self, data=data, mask=mask,
                                            **kwargs)
        self.axes_wcs = wcs
        # We don't send this to NDDataArray because it's not
        # supported as of astropy 1.0. Eventually we will.
        # Also it's called axes_wcs because wcs belongs to astropy.nddata and
        # that messes up slicing.

    def plot_wavelength_slice(self, offset, axes=None, style='imshow', **kwargs):
        """
        Plots an x-y graph at a certain specified wavelength onto the current
        axes. Keyword arguments are passed on to matplotlib.

        Parameters
        ----------
        offset: `int` or `float`
            The offset from the primary wavelength to plot. If it's an int it
            will plot the nth wavelength from the primary; if it's a float then
            it will plot the closest wavelength. If the offset is out of range,
            it will plot the primary wavelength (offset 0)

        axes: `astropy.visualization.wcsaxes.core.WCSAxes` or `None`:
            The axes to plot onto. If None the current axes will be used.

        style: 'imshow' or 'pcolormesh'
            The style of plot to be used. Default is 'imshow'
        """
        if axes is None:
            axes = wcsaxes_compat.gca_wcs(self.axes_wcs, slices=("x", "y", offset))

        data = self._choose_wavelength_slice(offset)
        if data is None:
            data = self._choose_wavelength_slice(0)

        if style == 'imshow':
            plot = axes.imshow(data, **kwargs)
        elif style == 'pcolormesh':
            plot = axes.pcolormesh(data, **kwargs)

        return plot

    def plot_x_slice(self, offset, axes=None, style='imshow', **kwargs):
        """
        Plots an x-y graph at a certain specified wavelength onto the current
        axes. Keyword arguments are passed on to matplotlib.

        Parameters
        ----------
        offset: `int` or `float`
            The offset from the initial x value to plot. If it's an int it
            will plot slice n from the start; if it's a float then
            it will plot the closest x-distance. If the offset is out of range,
            it will plot the primary wavelength (offset 0)

        axes: `astropy.visualization.wcsaxes.core.WCSAxes` or None:
            The axes to plot onto. If None the current axes will be used.

        style: 'imshow' or 'pcolormesh'
            The style of plot to be used. Default is 'imshow'
        """
        if axes is None:
            axes = wcsaxes_compat.gca_wcs(self.axes_wcs, slices=("x", offset, "y"))

        data = self._choose_x_slice(offset)
        if data is None:
            data = self._choose_x_slice(0)

        if style == 'imshow':
            plot = axes.imshow(data, **kwargs)
        elif style == 'pcolormesh':
            plot = axes.pcolormesh(data, **kwargs)

        return plot

    def animate(self, *args, **kwargs):
        """
        Plots an interactive visualization of this cube with a slider
        controlling the wavelength axis.
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
        i = ImageAnimatorWCS(self.data, wcs=self.axes_wcs, *args, **kwargs)
        return i

    def _choose_wavelength_slice(self, offset):
        """
        Retrieves an x-y slice at a wavelength specified by the cube's
        primary wavelength plus the given offset.

        Parameters
        ----------
        offset: `int` or astropy quantity
            Offset from the cube's primary wavelength. If the value is an int,
            then it returns that slice. Otherwise, it will return the nearest
            wavelength to the one specified.
        """
        if 'WAVE' not in self.axes_wcs.wcs.ctype:
            raise cu.CubeError(2, "Spectral dimension not present")
        if self.data.ndim == 4:
            raise cu.CubeError(4, "Can only work with 3D cubes")

        axis = 1 if self.axes_wcs.wcs.ctype[-1] in ['TIME', 'UTC'] else 0
        arr = None
        length = self.data.shape[axis]
        if isinstance(offset, int) and offset >= 0 and offset < length:
            arr = self.data.take(offset, axis=axis)

        if isinstance(offset, u.Quantity):
            delta = self.axes_wcs.wcs.cdelt[-1 - axis] * u.m
            wloffset = offset.to(u.m) / delta
            wloffset = int(wloffset)
            if wloffset >= 0 and wloffset < self.data.shape[axis]:
                arr = self.data.take(wloffset, axis=axis)

        return arr

    def _choose_x_slice(self, offset):
        """
        Retrieves a lambda-y slice at an x coordinate specified by the cube's
        primary wavelength plus the given offset.

        Parameters
        ----------
        offset: `int` or astropy quantity
            Offset from the cube's initial x. If the value is an int,
            then it returns that slice. Otherwise, it will return the nearest
            wavelength to the one specified.
        """
        arr = None
        axis = 1 if self.axes_wcs.wcs.ctype[-2] != 'WAVE' else 2
        length = self.data.shape[axis]
        if isinstance(offset, int) and offset >= 0 and offset < length:
            arr = self.data.take(offset, axis=axis)

        if isinstance(offset, u.Quantity):
            unit = self.axes_wcs.wcs.cunit[-1 - axis]
            delta = self.axes_wcs.wcs.cdelt[-1 - axis] * unit
            wloffset = offset.to(unit) / delta
            wloffset = int(wloffset)
            if wloffset >= 0 and wloffset < self.data.shape[axis]:
                arr = self.data.take(wloffset, axis=axis)

        return arr

    @classmethod
    def _new_instance(cls, data, wcs, errors=None, **kwargs):
        """
        Instantiate a new instance of this class using given data.
        """
        return cls(data, wcs, errors=errors, **kwargs)

    def slice_to_map(self, chunk, snd_dim=None, *args, **kwargs):
        """
        Converts a given frequency chunk to a SunPy Map. Extra parameters are
        passed on to Map.

        Parameters
        ----------
        chunk: int or astropy quantity or tuple
            The piece of the cube to convert to a map. If it's a single number,
            then it will return that single-slice map, otherwise it will
            aggregate the given range. Depending on the cube, this may
            correspond to a time or an energy dimension
        snd_dim: int or astropy quantity or tuple, optional
            Only used for hypercubes, the wavelength to choose from; works in
            the same way as chunk.
        """
        if self.axes_wcs.wcs.ctype[-2] == 'WAVE' and self.data.ndim == 3:
            error = "Cannot construct a map with only one spatial dimension"
            raise cu.CubeError(3, error)
        if isinstance(chunk, tuple):
            item = slice(cu.pixelize(chunk[0], self.axes_wcs, 0),
                         cu.pixelize(chunk[1], self.axes_wcs, 0), None)
            maparray = self.data[item].sum(0)
        else:
            maparray = self.data[cu.pixelize(chunk, self.axes_wcs, 0)]

        if self.data.ndim == 4:
            if snd_dim is None:
                error = "snd_dim must be given when slicing hypercubes"
                raise cu.CubeError(4, error)

            if isinstance(snd_dim, tuple):
                item = slice(cu.pixelize(snd_dim[0], self.axes_wcs, 1),
                             cu.pixelize(snd_dim[1], self.axes_wcs, 1), None)
                maparray = maparray[item].sum(0)
            else:
                maparray = maparray[cu.pixelize(snd_dim, self.axes_wcs, 1)]

        mapheader = MetaDict(self.meta)
        gmap = GenericMap(data=maparray, header=mapheader, *args, **kwargs)
        return gmap

    def slice_to_lightcurve(self, wavelength, y_coord=None, x_coord=None):
        """
        For a time-lambda-y cube, returns a lightcurve with curves at the
        specified wavelength and given y-coordinate. If no y is given, all of
        them will be used (meaning the lightcurve object could contain more
        than one timecurve.)
        Parameters
        ----------
        wavelength: int or astropy quantity
            The wavelength to take the y-coordinates from
        y_coord: int or astropy quantity, optional
            The y-coordinate to take the lightcurve from.
        x_coord: int or astropy quantity, optional
            In the case of hypercubes, specify an extra celestial coordinate.
        """
        if self.axes_wcs.wcs.ctype[-1] not in ['TIME', 'UTC']:
            raise cu.CubeError(1,
                               'Cannot create a lightcurve with no time axis')
        if self.axes_wcs.wcs.ctype[-2] != 'WAVE':
            raise cu.CubeError(2, 'A spectral axis is needed in a lightcurve')
        if self.data.ndim == 3:
            data = self._choose_wavelength_slice(wavelength)
            if y_coord is not None:
                data = data[:, cu.pixelize(y_coord, self.axes_wcs, 1)]
        else:
            if y_coord is None and x_coord is None:
                raise cu.CubeError(4, "At least one coordinate must be given")
            if y_coord is None:
                y_coord = slice(None, None, None)
            else:
                y_coord = cu.pixelize(y_coord, self.axes_wcs, 2)
            if x_coord is None:
                x_coord = slice(None, None, None)
            else:
                x_coord = cu.pixelize(x_coord, self.axes_wcs, 3)
            item = (slice(None, None, None), wavelength, y_coord, x_coord)
            data = self.data[item]

        return LightCurve(data=data, meta=self.meta)

    def slice_to_spectrum(self, *coords, **kwargs):
        """
        For a cube containing a spectral dimension, returns a sunpy spectrum.
        The given coordinates represent which values to take. If they are None,
        then the corresponding axis is summed.

        Parameters
        ----------
        fst_coord: int or None
            The first coordinate to pick. Keep in mind that depending on the
            cube, this may be in the first or second axis. If None, the whole
            axis will be taken and its values summed.
        snd_coord: int or None
            The second coordinate to pick. This will always correspond to the
            third axis. If None, the whole axis will be taken and its values
            summed.
        """
        if 'WAVE' not in self.axes_wcs.wcs.ctype:
            raise cu.CubeError(2, 'Spectral axis needed to create a spectrum')
        axis = 0 if self.axes_wcs.wcs.ctype[-1] == 'WAVE' else 1
        pixels = [cu.pixelize(coord, self.axes_wcs, axis) for coord in coords]
        item = range(len(pixels))
        if axis == 0:
            item[1:] = pixels
            item[0] = slice(None, None, None)
            item = [slice(None, None, None) if i is None else i for i in item]
        else:
            item[0] = pixels[0]
            item[1] = slice(None, None, None)
            item[2:] = pixels[1:]
            item = [slice(None, None, None) if i is None else i for i in item]

        data = self.data[item]
        errors = (None if self.uncertainty is None else self.uncertainty[item])
        mask = None if self.mask is None else self.mask[item]
        kwargs.update({'uncertainty': errors, 'mask': mask})
        for i in range(len(pixels)):
            if pixels[i] is None:
                if i == 0:
                    sumaxis = 1 if axis == 0 else 0
                else:
                    sumaxis = 1 if i == 2 else i
                data = data.sum(axis=sumaxis)

        wavelength_axis = self.wavelength_axis()
        freq_axis, cunit = wavelength_axis.value, wavelength_axis.unit
        err = self.uncertainty[item] if self.uncertainty is not None else None
        kwargs.update({'uncertainty': err})
        return Spectrum(np.array(data), np.array(freq_axis), cunit, **kwargs)

    def slice_to_spectrogram(self, y_coord, x_coord=None, **kwargs):
        """
        For a time-lambda-y cube, given a y-coordinate, returns a sunpy
        spectrogram. Keyword arguments are passed on to Spectrogram's __init__.

        Parameters
        ----------
        y_coord: int
            The y-coordinate to pick when converting to a spectrogram.
        x_coord: int
            The x-coordinate to pick. This is only used for hypercubes.
        """
        if self.axes_wcs.wcs.ctype[-1] not in ['TIME', 'UTC']:
            raise cu.CubeError(1,
                               'Cannot create a spectrogram with no time axis')
        if self.axes_wcs.wcs.ctype[-2] != 'WAVE':
            raise cu.CubeError(2, 'A spectral axis is needed in a spectrogram')
        if self.data.ndim == 3:
            data = self.data[:, :, cu.pixelize(y_coord, self.axes_wcs, 2)]
        else:
            if x_coord is None:
                raise cu.CubeError(4, 'An x-coordinate is needed for 4D cubes')
            data = self.data[:, :, cu.pixelize(y_coord, self.axes_wcs, 2),
                             cu.pixelize(x_coord, self.axes_wcs, 3)]
        time_axis = self.time_axis().value
        freq_axis = self.wavelength_axis().value

        if 'DATE_OBS'in self.meta:
            tformat = '%Y-%m-%dT%H:%M:%S.%f'
            start = datetime.datetime.strptime(self.meta['DATE_OBS'], tformat)
        else:
            start = datetime.datetime(1, 1, 1)

        if 'DATE_END' in self.meta:
            tformat = '%Y-%m-%dT%H:%M:%S.%f'
            end = datetime.datetime.strptime(self.meta['DATE_END'], tformat)
        else:
            dif = time_axis[-1] - time_axis[0]
            unit = self.axes_wcs.wcs.cunit[-1]
            dif = dif * u.Unit(unit)
            days = dif.to(sday)
            lapse = datetime.timedelta(days.value)
            end = start + lapse
        return Spectrogram(data=data, time_axis=time_axis, freq_axis=freq_axis,
                           start=start, end=end, **kwargs)

    def slice_to_cube(self, axis, chunk, **kwargs):
        """
        For a hypercube, return a 3-D cube that has been cut along the given
        axis and with data corresponding to the given chunk.

        Parameters
        ----------
        axis: int
            The axis to cut from the hypercube
        chunk: int, astropy Quantity or tuple:
            The data to take from the axis
        """
        if self.data.ndim == 3:
            raise cu.CubeError(4, 'Can only slice a hypercube into a cube')

        item = [slice(None, None, None) for _ in range(4)]
        if isinstance(chunk, tuple):
            if cu.iter_isinstance(chunk, (u.Quantity, u.Quantity)):
                pixel0 = cu.convert_point(chunk[0].value, chunk[0].unit,
                                          self.axes_wcs, axis)
                pixel1 = cu.convert_point(chunk[1].value, chunk[1].unit,
                                          self.axes_wcs, axis)
                item[axis] = slice(pixel0, pixel1, None)
            elif cu.iter_isinstance((chunk, int, int)):
                item[axis] = slice(chunk[0], chunk[1], None)
            else:
                raise cu.CubeError(5, "Parameters must be of the same type")
            newdata = self.data[item].sum(axis)
        else:
            unit = chunk.unit if isinstance(chunk, u.Quantity) else None
            pixel = cu.convert_point(chunk, unit, self.axes_wcs, axis)
            item[axis] = pixel
            newdata = self.data[item]
        wcs_indices = [0, 1, 2, 3]
        wcs_indices.remove(3 - axis)
        newwcs = wu.reindex_wcs(self.axes_wcs, np.array(wcs_indices))
        if axis == 2 or axis == 3:
            newwcs = wu.add_celestial_axis(newwcs)
            newwcs.was_augmented = True
        cube = Cube(newdata, newwcs, meta=self.meta, **kwargs)
        return cube

    def convert_to_spectral_cube(self):
        """
        Converts this cube into a SpectralCube. It will only work if the cube
        has exactly three dimensions and one of those is a spectral axis.
        """
        if self.data.ndim == 4:
            raise cu.CubeError(4, "Too many dimensions: Can only convert a " +
                               "3D cube. Slice the cube before converting")
        if 'WAVE' not in self.axes_wcs.wcs.ctype:
            raise cu.CubeError(2, 'Spectral axis needed to create a spectrum')
        axis = 0 if self.axes_wcs.wcs.ctype[-1] == 'WAVE' else 1
        coordaxes = [1, 2] if axis == 0 else [0, 2]  # Non-spectral axes
        newwcs = wu.reindex_wcs(self.axes_wcs, np.arary(coordaxes))
        time_or_x_size = self.data.shape[coordaxes[1]]
        y_size = self.data.shape[coordaxes[0]]
        spectra = np.empty((time_or_x_size, y_size), dtype=Spectrum)
        for i in range(time_or_x_size):
            for j in range(y_size):
                spectra[i][j] = self.slice_to_spectrum(i, j)
        return SpectralCube(spectra, newwcs, self.meta)

    def time_axis(self):
        """
        Returns a numpy array containing the time values for the cube's time
        dimension, as well as the unit used.
        """
        if self.axes_wcs.wcs.ctype[-1] not in ['TIME', 'UTC']:
            raise cu.CubeError(1, 'No time axis present')
        delta = self.axes_wcs.wcs.cdelt[-1]
        crpix = self.axes_wcs.wcs.crpix[-1]
        crval = self.axes_wcs.wcs.crval[-1]
        start = crval - crpix * delta
        stop = start + len(self.data) * delta
        cunit = u.Unit(self.axes_wcs.wcs.cunit[-1])
        return np.linspace(start, stop, num=self.data.shape[0]) * cunit

    def wavelength_axis(self):
        """
        Returns a numpy array containing the frequency values for the cube's
        spectral dimension, as well as the axis's unit.
        """
        if 'WAVE' not in self.axes_wcs.wcs.ctype:
            raise cu.CubeError(2,
                               'No energy (wavelength, frequency) axis found')
        axis = 0 if self.axes_wcs.wcs.ctype[-1] == 'WAVE' else 1
        delta = self.axes_wcs.wcs.cdelt[-1 - axis]
        crpix = self.axes_wcs.wcs.crpix[-1 - axis]
        crval = self.axes_wcs.wcs.crval[-1 - axis]
        start = crval - crpix * delta
        stop = start + self.data.shape[axis] * delta
        cunit = u.Unit(self.axes_wcs.wcs.cunit[-1 - axis])
        return np.linspace(start, stop, num=self.data.shape[axis]) * cunit

    def _array_is_aligned(self):
        """
        Returns whether the wcs system and the array are well-aligned.
        """
        rot_matrix = self.axes_wcs.wcs.pc
        return np.allclose(rot_matrix, np.eye(self.axes_wcs.wcs.naxis))

    def __getitem__(self, item):
        if item is None or (isinstance(item, tuple) and None in item):
            raise IndexError("None indices not supported")
        pixels = cu.pixelize_slice(item, self.axes_wcs)
        if self.data.ndim == 3:
            return cu.getitem_3d(self, pixels)
        else:
            return cu.getitem_4d(self, pixels)


class CubeSequence(object):
    """
    Class representing list of cubes.

    Attributes
    ----------
    data_list : `list`
        List of cubes.

    meta : `dict` or None
        The header of the CubeSequence.

    common_axis: `int` or None
        The data axis which is common between the CubeSequence and the Cubes within.
        For example, if the Cubes are sequenced in chronological order and time is
        one of the zeroth axis of each Cube, then common_axis should be se to 0.
        This enables the option for the CubeSequence to be indexed as though it is
        one single Cube.
    """

    def __init__(self, data_list, meta=None, common_axis=None):
        if not all(isinstance(data, Cube) for data in data_list):
            raise ValueError("data list should be of cube object")
        self.data = data_list
        self.meta = meta
        self.common_axis = common_axis
        self.shape = tuple([len(data_list)] + list(data_list[0].shape))

    def __getitem__(self, item):
        if item is None or (isinstance(item, tuple) and None in item):
            raise IndexError("None indices not supported")
        return cu.get_cube_from_sequence(self, item)

    def animate(self, *args, **kwargs):
        i = ani.ImageAnimatorCubeSequence(self, *args, **kwargs)
        return i

    def index_as_cube(self, item):
        """
        Method to slice the cubesequence instance as a single cube
        """
        return cu.index_sequence_as_cube(self, item)

    def plot_x_slice(self, offset, **kwargs):
        """
        Plots an x-y graph at a certain specified wavelength onto the current
        axes. Keyword arguments are passed on to matplotlib.

        Parameters
        ----------
        offset: `int` or `float`
            The offset from the initial x value to plot. If it's an int it
            will plot slice n from the start; if it's a float then
            it will plot the closest x-distance. If the offset is out of range,
            it will plot the primary wavelength (offset 0)

        axes: `astropy.visualization.wcsaxes.core.WCSAxes` or None:
            The axes to plot onto. If None the current axes will be used.

        style: 'imshow' or 'pcolormesh'
            The style of plot to be used. Default is 'imshow'
        """
        cumul_cube_lengths = np.cumsum(np.array([c.shape[self.common_axis]
                                                 for c in self.data]))
        sequence_index, cube_index = cu._convert_cube_like_index_to_sequence_indices(
            offset, cumul_cube_lengths)
        plot = self[sequence_index].plot_x_slice(cube_index, **kwargs)
        return plot

    def plot_wavelength_slice(self, offset, **kwargs):
        """ 
        Plots an x-y graph at a certain specified wavelength onto the current
        axes. Keyword arguments are passed on to matplotlib.

        Parameters
        ----------
        offset: `int` or `float`
            The offset from the primary wavelength to plot. If it's an int it
            will plot the nth wavelength from the primary; if it's a float then
            it will plot the closest wavelength. If the offset is out of range,
            it will plot the primary wavelength (offset 0)

        axes: `astropy.visualization.wcsaxes.core.WCSAxes` or `None`:
            The axes to plot onto. If None the current axes will be used.

        style: 'imshow' or 'pcolormesh'
            The style of plot to be used. Default is 'imshow'
        """
        cumul_cube_lengths = np.cumsum(np.array([c.shape[self.common_axis]
                                                 for c in self.data]))
        sequence_index, cube_index = cu._convert_cube_like_index_to_sequence_indices(
            offset, cumul_cube_lengths)
        plot = self[sequence_index].plot_wavelength_slice(cube_index, **kwargs)
        return plot
