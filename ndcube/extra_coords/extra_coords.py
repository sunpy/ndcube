import abc
import copy
from typing import Any, Tuple, Union, Iterable
from numbers import Integral
from functools import reduce

from astropy.wcs.wcsapi import BaseHighLevelWCS, BaseLowLevelWCS
from astropy.wcs.wcsapi.wrappers.sliced_wcs import sanitize_slices

from .lookup_table_coord import LookupTableCoord

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
    def __init__(self,
                 *,
                 wcs: BaseLowLevelWCS = None,
                 mapping: Iterable[Tuple[int, int]] = None):
        pass

    @abc.abstractmethod
    def add_coordinate(self,
                       name: str,
                       array_dimension: Union[int, Iterable[int]],
                       lookup_table: Any,
                       **kwargs):
        """
        Add a coordinate to this ``ExtraCoords`` based on a lookup table.

        Parameters
        ----------
        name
            The name for these world coordinate(s).
        array_dimension
            The pixel dimension(s), in the array, to which this lookup table corresponds.
        lookup_table
            The lookup table. Note, if this table is multi-dimensional it must
            (currently) be specified with its axes in world order, so
            transposed with respect to the data array.
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
    def __init__(self, *, wcs=None, mapping=None):
        if (wcs is None and mapping is not None) or (wcs is not None and mapping is None):
            raise ValueError("Either both WCS and mapping have to be specified together or neither.")

        super().__init__(wcs=wcs, mapping=mapping)

        # Setup private attributes
        self._wcs = None
        self._mapping = None
        # Lookup tables is a list of (pixel_dim, LookupTableCoord) to allow for
        # one pixel dimension having more than one lookup coord.
        self._lookup_tables = []

        # Set values using the setters for validation
        self.wcs = wcs
        self.mapping = mapping

    @classmethod
    def from_lookup_tables(cls, names, pixel_dimensions, lookup_tables):
        """
        Construct an ExtraCoords instance from lookup tables.

        This is a convience wrapper around `.add_coordinate` which does not
        expose all the options available in that method.

        Parameters
        ----------
        array_shape : `tuple` of `int`, optional
            The shape of the array.
        names : `tuple` of `str`
            The names of the world coordinates.
        pixel_dimensions : `tuple` of `int`
            The pixel dimensions (in the array) to which the ``lookup_tables``
            apply. Must be the same length as ``lookup_tables``.
        lookup_tables : `tuple` of `object`
            The lookup tables which specify the world coordinates for the ``pixel_dimensions``.

        Returns
        -------
        `ndcube.extra_coords.ExtraCoords`

        """
        if len(pixel_dimensions) != len(lookup_tables):
            raise ValueError(
                "The length of pixel_dimensions and lookup_tables must match."
            )

        extra_coords = cls()

        for name, pixel_dim, lookup_table in zip(names, pixel_dimensions, lookup_tables):
            extra_coords.add_coordinate(name, pixel_dim, lookup_table)

        return extra_coords

    def add_coordinate(self, name, array_dimension, lookup_table, **kwargs):
        # docstring in ABC

        if self._wcs is not None:
            raise ValueError(
                "Can not add a lookup_table to an ExtraCoords which was instantiated with a WCS object."
            )

        lutc = LookupTableCoord(lookup_table, names=name, **kwargs)
        self._lookup_tables.append((array_dimension, lutc))

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

        lts = [list([lt[0]] if isinstance(lt[0], Integral) else lt[0]) for lt in self._lookup_tables]
        return tuple(reduce(list.__add__, lts))

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

        lutcs = set(lt[1] for lt in self._lookup_tables)
        # created a sorted list of unique items
        _tmp = set()  # a temporary set
        lutcs = [x[1] for x in self._lookup_tables if x[1] not in _tmp and _tmp.add(x[1]) is None]
        out = copy.deepcopy(lutcs[0])
        for lut in lutcs[1:]:
            out = out & lut
        return out.wcs

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
                new_ec = ExtraCoords()
                new_ec._lookup_tables = [lut]
                return new_ec

        raise KeyError(f"Can't find the world axis named {item} in this ExtraCoords object.")

    def _getitem_lookup_tables(self, item):
        """
        Apply an array slice to the lookup tables.

        Returns a new ExtraCoords object with modified lookup tables.
        """
        new_lookup_tables = set()
        for lut_axis, lut in self._lookup_tables:
            lut_axes = (lut_axis,) if not isinstance(lut_axis, tuple) else lut_axis
            lut_slice = tuple(item[i] for i in lut_axes) if isinstance(item, tuple) else item

            sliced_lut = lut[lut_slice]
            if sliced_lut:
                new_lookup_tables.add((lut_axis, sliced_lut))

        new_extra_coords = type(self)()
        new_extra_coords._lookup_tables = tuple(new_lookup_tables)
        return new_extra_coords

    def _getitem_wcs(self, item):
        item = sanitize_slices(item, self.wcs.pixel_n_dim)

        # It's valid to slice down the EC such that there is nothing left,
        # which is not a valid way to slice the WCS
        if len(item) == self.wcs.pixel_n_dim and all(isinstance(i, Integral) for i in item):
            return type(self)()

        subwcs = self.wcs[item]

        new_mapping = [self.mapping[i] for i, subitem in enumerate(item) if not isinstance(subitem, Integral)]

        return type(self)(wcs=subwcs, mapping=new_mapping)

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
