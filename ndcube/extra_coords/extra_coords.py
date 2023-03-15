import abc
from typing import Any, Tuple, Union, Iterable
from numbers import Integral
from functools import reduce, partial

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.wcsapi import BaseHighLevelWCS
from astropy.wcs.wcsapi.high_level_wcs_wrapper import HighLevelWCSWrapper
from astropy.wcs.wcsapi.wrappers.sliced_wcs import SlicedLowLevelWCS, sanitize_slices

from ndcube.utils.wcs import convert_between_array_and_pixel_axes
from ndcube.wcs.wrappers import CompoundLowLevelWCS, ResampledLowLevelWCS

from .table_coord import (BaseTableCoordinate, MultipleTableCoordinate, QuantityTableCoordinate,
                          SkyCoordTableCoordinate, TimeTableCoordinate)

__all__ = ['ExtraCoordsABC', 'ExtraCoords']


class ExtraCoordsABC(abc.ABC):
    """
    A representation of additional world coordinates associated with pixel axes.

    ExtraCoords can be initialised by either specifying a
    `~astropy.wcs.wcsapi.BaseLowLevelWCS` object and a ``mapping``, or it can be
    built up by specifying one or more lookup tables.

    Parameters
    ----------
    wcs
        The WCS specifying the extra coordinates.
    mapping
       The mapping between the array dimensions and pixel dimensions in the
       extra coords object. This is an iterable of ``(array_dimension, pixel_dimension)`` pairs
       of length equal to the number of pixel dimensions in the extra coords.

    """
    @abc.abstractmethod
    def add(self,
            name: Union[str, Iterable[str]],
            array_dimension: Union[int, Iterable[int]],
            lookup_table: Any,
            physical_types: Union[str, Iterable[str]] = None,
            **kwargs):
        """
        Add a coordinate to this `~ndcube.ExtraCoords` based on a lookup table.

        Parameters
        ----------
        name : `str` or sequence of `str`
            The name(s) for these world coordinate(s).
        array_dimension : `int` or `tuple` of `int`
            The array dimension(s), to which this lookup table corresponds.
        lookup_table : `object` or sequence of `object`
            The lookup table. A `~ndcube.extra_coords.BaseTableCoordinate` subclass or anything
            that can instantiate one, i.e. currently a `~astropy.time.Time`,
            `~astropy.coordinates.SkyCoord`, or a (sequence of) `~astropy.units.Quantity`.
        physical_types: `str` or iterable of `str`, optional
            Descriptor(s) of the :ref:`<dimensions and physical types <dimensions>`
            associated with each axis; length must match the number of dimensions in
            ``lookup_table``.
        """

    @abc.abstractmethod
    def keys(self) -> Iterable[str]:
        """
        The world axis names for all the coordinates in the extra coords.
        """

    @property
    @abc.abstractmethod
    def mapping(self) -> Iterable[Tuple[int, int]]:
        """
        The mapping between the array dimensions and pixel dimensions.

        This is an iterable of ``(array_dimension, pixel_dimension)`` pairs
        of length equal to the number of pixel dimensions in the extra coords.
        """

    @property
    @abc.abstractmethod
    def wcs(self) -> BaseHighLevelWCS:
        """
        A WCS object representing the world coordinates described by this ``ExtraCoords``.

        .. note::
            This WCS object does not map to the pixel dimensions of the data array
            in the `~ndcube.NDCube` object. It only includes pixel dimensions associated
            with the extra coordinates. For example, if there is only one extra coordinate
            associated with a single pixel dimension, this WCS will only have 1 pixel dimension,
            even if the `~ndcube.NDCube` object has a data array of 2-D or greater.
            Therefore using this WCS directly might lead to some confusing results.

        """

    @property
    @abc.abstractproperty
    def is_empty(self):
        """Return True if no extra coords present, else return False."""

    @abc.abstractmethod
    def __getitem__(self, item: Union[str, int, slice, Iterable[Union[str, int, slice]]]) -> "ExtraCoordsABC":
        """
        ExtraCoords can be sliced with either a string, or a numpy like slice.

        When sliced with a string it should return a new ExtraCoords object
        with only those coordinates with the given names. When sliced with a
        numpy array like slice it should return a new ExtraCoords with the
        slice applied. Supporting step is not required and "fancy indexing" is
        not supported.
        """


class ExtraCoords(ExtraCoordsABC):
    """
    A representation of additional world coordinates associated with pixel axes.

    ExtraCoords can be initialised by either specifying a
    `~astropy.wcs.wcsapi.BaseLowLevelWCS` object and a ``mapping``, or it can be
    built up by specifying one or more lookup tables.

    Parameters
    ----------
    wcs
        The WCS specifying the extra coordinates.
    mapping
       The mapping between the array dimensions and pixel dimensions in the
       extra coords object. This is an iterable of ``(array_dimension, pixel_dimension)`` pairs
       of length equal to the number of pixel dimensions in the extra coords.

    """

    def __init__(self, ndcube=None):
        super().__init__()

        # Setup private attributes
        self._wcs = None
        self._mapping = None

        # Lookup tables is a list of (pixel_dim, LookupTableCoord) to allow for
        # one pixel dimension having more than one lookup coord.
        self._lookup_tables = list()
        self._dropped_tables = list()

        # We need a reference to the parent NDCube
        self._ndcube = ndcube

    @classmethod
    def from_lookup_tables(cls, names, pixel_dimensions, lookup_tables, physical_types=None):
        """
        Construct a new ExtraCoords instance from lookup tables.

        This is a convenience wrapper around `ndcube.ExtraCoords.add` which does not
        expose all the options available in that method.

        Parameters
        ----------
        names : `tuple` of `str`
            The names of the world coordinates.
        pixel_dimensions : `tuple` of `int`
            The pixel dimensions (in the array) to which the ``lookup_tables``
            apply. Must be the same length as ``lookup_tables``.
        lookup_tables : iterable of `object`
            The lookup tables which specify the world coordinates for the ``pixel_dimensions``.
            Must be `~ndcube.extra_coords.BaseTableCoordinate` subclass instances or objects from
            which to instantiate them (see `ndcube.ExtraCoords.add`).
        physical_types: sequence of `str` or of sequences of `str`, optional
            Descriptors of the :ref:`dimensions`
            associated with each axis in the tables. Must be the same length as ``lookup_tables``;
            and length of each element must match the number of dimensions in corresponding
            ``lookup_tables[i]``.
            Defaults to `None`.

        Returns
        -------
        `ndcube.ExtraCoords`

        """
        if len(pixel_dimensions) != len(lookup_tables):
            raise ValueError(
                "The length of pixel_dimensions and lookup_tables must match."
            )

        if physical_types is None:
            physical_types = len(lookup_tables) * [physical_types]
        elif len(physical_types) != len(lookup_tables):
            raise ValueError("The number of physical types and lookup_tables must match.")

        extra_coords = cls()

        for name, pixel_dim, lookup_table, physical_type in zip(names, pixel_dimensions,
                                                                lookup_tables, physical_types):
            extra_coords.add(name, pixel_dim, lookup_table, physical_types=physical_type)

        return extra_coords

    def add(self, name, array_dimension, lookup_table, physical_types=None, **kwargs):
        # docstring in ABC

        if self._wcs is not None:
            raise ValueError(
                "Can not add a lookup_table to an ExtraCoords which was instantiated with a WCS object."
            )

        kwargs['names'] = [name] if not isinstance(name, (list, tuple)) else name

        if isinstance(lookup_table, BaseTableCoordinate):
            coord = lookup_table
        elif isinstance(lookup_table, Time):
            coord = TimeTableCoordinate(lookup_table, physical_types=physical_types, **kwargs)
        elif isinstance(lookup_table, SkyCoord):
            coord = SkyCoordTableCoordinate(lookup_table, physical_types=physical_types, **kwargs)
        elif isinstance(lookup_table, (list, tuple)):
            coord = QuantityTableCoordinate(*lookup_table, physical_types=physical_types, **kwargs)
        elif isinstance(lookup_table, u.Quantity):
            coord = QuantityTableCoordinate(lookup_table, physical_types=physical_types, **kwargs)
        else:
            raise TypeError(f"The input type {type(lookup_table)} isn't supported")

        self._lookup_tables.append((array_dimension, coord))

        # Sort the LUTs so that the mapping and the wcs are ordered in pixel dim order
        self._lookup_tables = list(sorted(self._lookup_tables,
                                          key=lambda x: x[0] if isinstance(x[0], Integral) else x[0][0]))

    @property
    def _name_lut_map(self):
        """
        Map of world names to the corresponding `.LookupTableCoord`
        """
        return {lut[1].wcs.world_axis_names: lut for lut in self._lookup_tables}

    def keys(self):
        # docstring in ABC
        if not self.wcs:
            return tuple()

        return tuple(self.wcs.world_axis_names) if self.wcs.world_axis_names else None

    @property
    def mapping(self):
        # docstring in ABC
        if self._mapping:
            return self._mapping

        # If mapping is not set but lookup_tables is empty then the extra
        # coords is empty, so there is no mapping.
        if not self._lookup_tables:
            return tuple()

        # The mapping is from the array index (position in the list) to the
        # pixel dimensions (numbers in the list)
        lts = [list([lt[0]] if isinstance(lt[0], Integral) else lt[0]) for lt in self._lookup_tables]
        converter = partial(convert_between_array_and_pixel_axes, naxes=len(self._ndcube.dimensions))
        pixel_indicies = [list(converter(np.array(ids))) for ids in lts]
        return tuple(reduce(list.__add__, pixel_indicies))

    @mapping.setter
    def mapping(self, mapping):
        if self._mapping is not None:
            raise AttributeError("Can't set mapping if a mapping has already been specified.")

        if self._lookup_tables:
            raise AttributeError(
                "Can't set mapping manually when ExtraCoords is built from lookup tables."
            )

        if self._wcs is not None:
            if not max(mapping) <= self._wcs.pixel_n_dim - 1:
                raise ValueError(
                    "Values in the mapping can not be larger than the number of pixel dimensions in the WCS."
                )

        self._mapping = mapping

    @property
    def wcs(self):
        # docstring in ABC
        if self._wcs is not None:
            return self._wcs

        if not self._lookup_tables:
            return None

        tcoords = set(lt[1] for lt in self._lookup_tables)
        # created a sorted list of unique items
        _tmp = set()  # a temporary set
        tcoords = [x[1] for x in self._lookup_tables if x[1] not in _tmp and _tmp.add(x[1]) is None]
        return MultipleTableCoordinate(*tcoords).wcs

    @wcs.setter
    def wcs(self, wcs):
        if self._wcs is not None:
            raise AttributeError(
                "Can't set wcs if a WCS has already been specified."
            )

        if self._lookup_tables:
            raise AttributeError(
                "Can't set wcs manually when ExtraCoords is built from lookup tables."
            )

        if self._mapping is not None:
            if not max(self._mapping) <= wcs.pixel_n_dim - 1:
                raise ValueError(
                    "Values in the mapping can not be larger than the number of pixel dimensions in the WCS."
                )

        self._wcs = wcs

    @property
    def is_empty(self):
        # docstring in ABC
        if not self._wcs and not self._lookup_tables:
            return True
        else:
            return False

    def _getitem_string(self, item):
        """
        Slice the Extracoords based on axis names.
        """

        for names, lut in self._name_lut_map.items():
            if item in names:
                new_ec = ExtraCoords(ndcube=self._ndcube)
                new_ec._lookup_tables = [lut]
                return new_ec

        raise KeyError(f"Can't find the world axis named {item} in this ExtraCoords object.")

    def _getitem_lookup_tables(self, item):
        """
        Apply an array slice to the lookup tables.

        Returns a new ExtraCoords object with modified lookup tables.
        """
        dropped_tables = set()
        new_lookup_tables = set()
        ndims = max([lut[0] if isinstance(lut[0], Integral) else max(lut[0])
                     for lut in self._lookup_tables]) + 1
        # Determine how many dimensions will be dropped by slicing below each dimension.
        if isinstance(item, Integral):
            n_dropped_dims = np.ones(ndims, dtype=int)
            item = tuple([item] + [slice(None)] * (ndims - 1))
        elif isinstance(item, slice):
            n_dropped_dims = np.zeros(ndims, dtype=int)
            item = tuple([item] + [slice(None)] * (ndims - 1))
        else:
            item = list(item) + [slice(None)] * (ndims - len(item))
            n_dropped_dims = np.cumsum([isinstance(i, Integral) for i in item])
        for lut_axis, lut in self._lookup_tables:
            lut_axes = (lut_axis,) if not isinstance(lut_axis, tuple) else lut_axis
            new_lut_axes = tuple(ax - n_dropped_dims[ax] for ax in lut_axes)
            lut_slice = tuple(item[i] for i in lut_axes)
            if isinstance(lut_slice, tuple) and len(lut_slice) == 1:
                lut_slice = lut_slice[0]

            sliced_lut = lut[lut_slice]

            if sliced_lut.is_scalar():
                dropped_tables.add(sliced_lut)
            else:
                new_lookup_tables.add((new_lut_axes, sliced_lut))
        new_extra_coords = type(self)()
        new_extra_coords._lookup_tables = list(new_lookup_tables)
        new_extra_coords._dropped_tables = list(dropped_tables)
        return new_extra_coords

    def _getitem_wcs(self, item):
        item = sanitize_slices(item, self.wcs.pixel_n_dim)

        # It's valid to slice down the EC such that there is nothing left,
        # which is not a valid way to slice the WCS
        if len(item) == self.wcs.pixel_n_dim and all(isinstance(i, Integral) for i in item):
            return type(self)()

        subwcs = self.wcs[item]

        new_mapping = [self.mapping[i] for i, subitem in enumerate(item) if not isinstance(subitem, Integral)]

        new_ec = type(self)()
        new_ec.wcs = subwcs
        new_ec.mapping = new_mapping
        return new_ec

    def __getitem__(self, item):
        # docstring in ABC
        if isinstance(item, str):
            return self._getitem_string(item)

        if self._wcs:
            return self._getitem_wcs(item)

        elif self._lookup_tables:
            return self._getitem_lookup_tables(item)

        # If we get here this object is empty, so just return an empty extra coords
        # This is done to simplify the slicing in NDCube
        return self

    @property
    def dropped_world_dimensions(self):
        """
        Return an APE-14 like representation of any sliced out world dimensions.
        """

        if self._wcs:
            if isinstance(self._wcs, SlicedLowLevelWCS):
                return self._wcs.dropped_world_dimensions

        if self._lookup_tables or self._dropped_tables:
            mtc = MultipleTableCoordinate(*[lt[1] for lt in self._lookup_tables])
            mtc._dropped_coords = self._dropped_tables

            return mtc.dropped_world_dimensions

        return dict()

    def resample(self, factor, offset=0, ndcube=None, **kwargs):
        """
        Resample all extra coords by given factors in array-index-space.

        One resample factor must be supplied for each array axis in array-axis order.

        Parameters
        ----------
        factor: `int`, `float`, or iterable thereof.
            The factor by which each array axis is resampled.
            If scalar, same factor is applied to all axes.
            Otherwise a factor for each axis must be provided.

        offset: `int`, `float`, or iterable therefore.
            The location on the underlying grid which corresponds
            to the zeroth element after resampling. If iterable, must have an entry
            for each dimension. If a scalar, the grid will be
            shifted by the same amount in all dimensions.

        ndcube: `~ndcube.NDCube`
            The NDCube instance with which the output ExtraCoords object is associated.

        kwargs
            All remaining kwargs are passed to `numpy.interp`.

        Returns
        -------
        new_ec: `~ndcube.ExtraCoords`
            A new ExtraCoords object holding the interpolated coords.
        """
        new_ec = type(self)(ndcube)
        if self.is_empty:
            return new_ec
        if self._ndcube is not None:
            cube_shape = self._ndcube.data.shape
            ndim = len(cube_shape)
        elif self._wcs is not None:
            ndim = self._wcs.pixel_n_dim
        else:
            raise NotImplementedError(
                "Resampling a lookup-table-based ExtraCoords not yet implemented. "
                "Please raise an issue at https://github.com/sunpy/ndcube/issues "
                "if you need this functionality")
        if np.isscalar(factor):
            factor = [factor] * ndim
        if len(factor) != ndim:
            raise ValueError(
                "factor must be scalar or an iterable with length equal to number of cube "
                f"dimensions: len(factor) = {len(factor)}; No. cube dimensions = {ndim}.")
        if np.isscalar(offset):
            offset = [offset] * ndim
        if len(offset) != ndim:
            raise ValueError(
                "offset must be scalar or an iterable with length equal to number of cube "
                f"dimensions: len(offset) = {len(offset)}; No. cube dimensions = {ndim}.")
        # If ExtraCoords object built on WCS, resample using WCS insfrastructure
        if self._wcs is not None:
            new_ec.wcs = HighLevelWCSWrapper(ResampledLowLevelWCS(self._wcs.low_level_wcs,
                                                                  factor, offset))
            return new_ec
        # Else interpolate the lookup table coordinates.
        factor = np.asarray(factor)
        new_grids = []
        for c, d, f in zip(offset, cube_shape, factor):
            x = np.arange(c, d+f, f)
            x = x[x <= d-1]
            new_grids.append(x)
        new_grids = np.array(new_grids, dtype=object)
        for array_axes, coord in self._lookup_tables:
            if np.isscalar(array_axes):
                new_coord = coord.interpolate(new_grids[array_axes], **kwargs)
            else:
                new_coord = coord.interpolate(*new_grids[np.asarray(array_axes)], **kwargs)
            new_ec.add(coord.names, array_axes, new_coord, physical_types=coord.physical_types)
        return new_ec

    @property
    def cube_wcs(self):
        """Produce a WCS that describes the associated NDCube with just the extra coords.

        For NDCube pixel axes without any extra coord, dummy axes are inserted.
        """
        wcses = [self.wcs]
        mapping = list(self.mapping)
        dummy_axes = self._cube_array_axes_without_extra_coords
        n_dummy_axes = len(dummy_axes)
        if n_dummy_axes > 0:
            dummy_wcs = WCS(naxis=n_dummy_axes)
            dummy_wcs.wcs.crpix = [1] * n_dummy_axes
            dummy_wcs.wcs.cdelt = [1] * n_dummy_axes
            dummy_wcs.wcs.crval = [0] * n_dummy_axes
            dummy_wcs.wcs.ctype = ["PIXEL"] * n_dummy_axes
            dummy_wcs.wcs.cunit = ["pix"] * n_dummy_axes
            wcses.append(dummy_wcs)
            mapping += list(dummy_axes)
        return CompoundLowLevelWCS(*wcses, mapping=mapping)

    @property
    def _cube_array_axes_without_extra_coords(self):
        """Return the array axes not associated with any extra coord."""
        return set(range(len(self._ndcube.dimensions))) - set(self.mapping)

    def __str__(self):
        classname = self.__class__.__name__
        elements = [f"{', '.join(table.names)} ({axes}) {table.physical_types}: {table}"
                    for axes, table in self._lookup_tables]
        length = len(classname) + 2 * len(elements) + sum(len(e) for e in elements)
        if length > np.get_printoptions()['linewidth']:
            joiner = ',\n ' + len(classname) * ' '
        else:
            joiner = ', '

        return f"{classname}({joiner.join(elements)})"

    def __repr__(self):
        return f"{object.__repr__(self)}\n{self}"
