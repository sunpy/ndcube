from typing import Union

import astropy.units as u
import gwcs.coordinate_frames as cf
from astropy.coordinates import SkyCoord
from astropy.time import Time

try:
    from functools import singledispatchmethod
except ImportError:
    from sunpy.util.functools import seconddispatch as singledispatchmethod

__all__ = ['LookupTableCoord', 'ExtraCoords']


class ExtraCoords:
    """
    A representation of additional world coordinates associated with pixel axes.

    ExtraCoords can be initialised by either specifying a
    `~astropy.wcs.wcsapi.LowLevelWCS` object and a ``mapping``, or it can be
    built up by specifying one or more lookup tables.

    Parameters
    ----------
    array_shape : `tuple` of `int`, optional
        The shape of the array, if not specified ``wcs.array_shape`` must be not `None`.
    wcs : `astropy.wcs.wcsapi.LowLevelWCS`
        The WCS specifying the extra coordinates.
    mapping : `tuple` of `int`
       The mapping between the array dimensions and pixel dimensions in the wcs.
    """
    def __init__(self, *, array_shape=None, wcs=None, mapping=None):
        if array_shape is not None or (wcs is not None and wcs.array_shape is not None):
            self.array_shape = array_shape or wcs.array_shape
        else:
            raise ValueError("Either array_shape or wcs.array_shape must be specified")

        self.array_ndim =  len(self.array_shape)

        # TODO: verify these mapping checks are correct
        if mapping is not None:
            if max(mapping) >= self.array_ndim:
                raise ValueError("The provided mapping tried to map to more pixel dimensions than `ndim`.")
        if wcs is not None:
            if not len(mapping) == wcs.pixel_n_dim:
                raise ValueError("The number of pixel dimensions in the WCS does not match the length of the mapping.")

        self._wcs = wcs
        self._mapping = mapping
        # Lookup tables is a list of (pixel_dim, LookupTableCoord) to allow for
        # one pixel dimension having more than one lookup coord.
        self._lookup_tables = []

    @classmethod
    def from_lookup_tables(cls, array_shape, pixel_dimensions, lookup_tables):
        """
        Construct an ExtraCoords instance from lookup tables.

        Parameters
        ----------
        array_shape : `tuple` of `int`, optional
            The shape of the array.
        pixel_dimensions : `tuple` of `int`
            The pixel dimensions (in the array) to which the ``lookup_tables``
            apply. Must be the same length as ``lookup_tables``.
        lookup_tables : `tuple` of `object`
            The lookup tables which specify the world coordinates for the ``pixel_dimensions``.

        Returns
        -------
        `ndcube.extra_coords.ExtraCoords`

        """
        extra_coords = cls(array_shape=array_shape)

        for pixel_dim, lookup_table in zip(pixel_dimensions, lookup_tables):
            extra_coords.add_coordinate(pixel_dim, lookup_table)

        return extra_coords

    def add_coordinate(self, pixel_dimension, lookup_table):
        """
        Add a coordinate to this ``ExtraCoords`` based on a lookup table.

        Parameters
        ----------
        pixel_dimension : `int`
            The pixel dimension (in the array) to which this lookup table corresponds.
        lookup_table : `object`
            The lookup table.
        """
        if self._wcs is not None:
            raise ValueError(
                "Can not add a lookup_table to an ExtraCoords which was instantiated with a WCS object."
            )

        self._lookup_tables.append((pixel_dimension, LookupTableCoord(lookup_table)))

    @property
    def mapping(self):
        """
        The mapping of the world dimensions in this ``ExtraCoords`` to pixel
        dimensions in the array.
        """
        if self._mapping:
            return self._mapping

        if not self._lookup_tables:
            return None

        return tuple(enumerate([lt[0] for lt in self._lookup_tables]))

    @mapping.setter
    def _mapping(self, mapping):
        if self._mapping is not None:
            raise AttributeError("Can't set mapping if a mapping has already been specified.")

        if self._lookup_tables:
            raise AttributeError(
                "Can't set mapping manually when ExtraCoords is built from lookup tables."
            )

        if self._wcs is not None:
            if len(mapping) != self.wcs.pixel_n_dim:
                raise ValueError("Mapping does not specify the same number of dimensions as the WCS.")

        self._mapping = mapping

    @property
    def wcs(self):
        """
        A WCS object representing the world coordinates described by this ``ExtraCoords``.

        .. note::
            This WCS object does not map to the pixel dimensions of the array
            associated with the `~ndcube.NDCube` object. It has the number of
            pixel dimensions equal to the number of inputs to the transforms to
            get the world coordinates (normally equal to the number of world
            coordinates). Therefore using this WCS directly might lead to some
            confusing results.

        """
        if self._wcs is not None:
            return self._wcs

        # This bit is complex need to build a compound model and a
        # CompoundFrame and gwcs here.
        raise NotImplementedError()

    @wcs.setter
    def _wcs(self, value):
        if self._wcs is not None:
            raise AttributeError(
                "Can't set wcs if a WCS has already been specified."
            )

        if self._lookup_tables:
            raise AttributeError(
                "Can't set mapping manually when ExtraCoords is built from lookup tables."
            )

        if self._mapping is not None:
            if len(self._mapping) != wcs.pixel_n_dim:
                raise ValueError(
                    "The WCS does not specify the same number of dimensions as described by the mapping."
                )

        self._wcs = wcs


class LookupTableCoord:
    """
    A class representing world coordinates described by a lookup table.

    This class takes an input in the form of a lookup table (can be many
    different array-like types) and holds the building blocks (transform and
    frame) to generate a `gwcs.WCS` object.

    Parameters
    ----------
    lookup_table : `object`
        The lookup table.
    """
    def __init__(self, lookup_table):
        self._model, self._frame = self.parse_lookup_table(lookup_table)

    @singledispatchmethod
    def _parse_lookup_table(self, lookup_table):
        """
        Generate an astropy model and gWCS frame from a lookup table.
        """
        raise NotImplementedError(f"Can not generate a lookup table from input of type {type(lookup_table)}")

    @_parse_lookup_table.register(Time)
    def _time_table(self, lookup_table):
        pass

    @_parse_lookup_table.register(SkyCoord)
    def _skycoord_table(self, lookup_table):
        pass

    @_parse_lookup_table.register(u.Quantity)
    def _quantity_table(self, lookup_table):
        pass

    @_parse_lookup_table.register(list)
    @_parse_lookup_table.register(tuple)
    def _list_table(self, lookup_table):
        pass
