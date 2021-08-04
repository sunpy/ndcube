import abc
from typing import Any, Tuple, Union, Iterable
from numbers import Integral
from functools import reduce, partial

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs.wcsapi import BaseHighLevelWCS
from astropy.wcs.wcsapi.wrappers.sliced_wcs import SlicedLowLevelWCS, sanitize_slices

from ndcube.utils.wcs import convert_between_array_and_pixel_axes

from .table_coord import (BaseTableCoordinate, MultipleTableCoordinate, QuantityTableCoordinate,
                          SkyCoordTableCoordinate, TimeTableCoordinate)

__all__ = ['ExtraCoords']


class ExtraCoordsABC(abc.ABC):
    """
    A representation of additional world coordinates associated with pixel axes.

    ExtraCoords can be initialised by either specifying a
    `~astropy.wcs.wcsapi.LowLevelWCS` object and a ``mapping``, or it can be
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
        Add a coordinate to this ``ExtraCoords`` based on a lookup table.

        Parameters
        ----------
        name : `str` or sequence of `str`
            The name(s) for these world coordinate(s).
        array_dimension : `int` or `tuple` of `int`
            The pixel dimension(s), in the array, to which this lookup table corresponds.
        lookup_table : `object` or sequence of `object`
            The lookup table. A `BaseTableCoordinate <.table_coord>` subclass or anything
            that can instantiate one, i.e. currently a `~astropy.time.Time`,
            `~astropy.coordinates.SkyCoord`, or a (sequence of) `~astropy.units.Quantity`.
        physical_types: `str` or iterable of `str`, optional
            Descriptor(s) of the `physical type <../data_classes.html#dimensions-and-physical-types>`_
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
            This WCS object does not map to the pixel dimensions of the array
            associated with the `.NDCube` object. It has the number of
            pixel dimensions equal to the number of inputs to the transforms to
            get the world coordinates (normally equal to the number of world
            coordinates). Therefore using this WCS directly might lead to some
            confusing results.

        """

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
    `~astropy.wcs.wcsapi.LowLevelWCS` object and a ``mapping``, or it can be
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

        This is a convience wrapper around `.add` which does not
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
            Must be `BaseTableCoordinate <.table_coord>` subclass instances or objects from
            which to instantiate them (see `.ExtraCoords.add`).
        physical_types: sequence of `str` or of sequences of `str`, optional
            Descriptors of the `physical types <../data_classes.html#dimensions-and-physical-types>`_
            associated with each axis in the tables. Must be the same length as ``lookup_tables``;
            and length of each element must match the number of dimensions in corresponding
            ``lookup_tables[i]``.

        Returns
        -------
        `ndcube.extra_coords.ExtraCoords`

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
                                          key=lambda x: x[0] if isinstance(x[0], int) else x[0][0]))

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
        for lut_axis, lut in self._lookup_tables:
            lut_axes = (lut_axis,) if not isinstance(lut_axis, tuple) else lut_axis
            lut_slice = tuple(item[i] for i in lut_axes) if isinstance(item, tuple) else item
            if isinstance(lut_slice, tuple) and len(lut_slice) == 1:
                lut_slice = lut_slice[0]

            sliced_lut = lut[lut_slice]

            if sliced_lut.is_scalar():
                dropped_tables.add(sliced_lut)
            else:
                new_lookup_tables.add((lut_axis, sliced_lut))

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

    def __str__(self):
        elements = [f"{', '.join(table.names)} ({axes}): {table}" for axes, table in self._lookup_tables]
        return f"ExtraCoords({', '.join(elements)})"

    def __repr__(self):
        return f"{object.__repr__(self)}\n{self}"
