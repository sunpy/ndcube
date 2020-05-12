import astropy.units as u
import gwcs
import gwcs.coordinate_frames as cf
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.modeling.models import tabular_model
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
            if len(mapping) == self.array_ndim:
                raise ValueError("The provided mapping tried to map to more pixel dimensions than `ndim`.")
        if wcs is not None:
            if not max(mapping) <= wcs.pixel_n_dim:
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

    This can be used as a way to generate gWCS objects based on lookup tables,
    however, it lacks some of the flexibility of doing this manually.

    Parameters
    ----------
    lookup_table : `object`
        The lookup table.
    """
    def __init__(self, lookup_table, mesh=False, names=None, physical_types=None, frame_type="auto"):
        self.model, self.frame = self._parse_lookup_table(lookup_table,
                                                          mesh=mesh,
                                                          names=names,
                                                          physical_types=physical_types,
                                                          frame_type=frame_type)

    @property
    def wcs(self):
        return gwcs.WCS(forward_transform=self.model,
                        input_frame=self._generate_generic_frame(self.model.n_inputs, u.pix),
                        output_frame=self.frame)

    @staticmethod
    def generate_tabular(lookup_table, interpolation='linear', points_unit=u.pix, **kwargs):
        if not isinstance(lookup_table, u.Quantity):
            raise TypeError("lookup_table must be a Quantity.")

        ndim = lookup_table.ndim
        TabularND = tabular_model(ndim, name=f"Tabular{ndim}D")

        # The integer location is at the centre of the pixel.
        points = [(np.arange(size) - 0) * points_unit for size in lookup_table.shape]

        kwargs = {
            'bounds_error': False,
            'fill_value': np.nan,
            'method': interpolation,
            **kwargs
            }

        return TabularND(points, lookup_table, **kwargs)

    @classmethod
    def _generate_compound_model(cls, *lookup_tables, mesh=True):
        """
        Takes a set of quantities and returns a ND compound model.
        """
        model = cls.generate_tabular(lookup_tables[0])
        for lt in lookup_tables[1:]:
            model = model & cls.generate_tabular(lt)

        return model

    @staticmethod
    def _generate_generic_frame(naxes, unit, names=None, physical_types=None):
        """
        Generate a simple frame, where all axes have the same type and unit.
        """
        axes_order = tuple(range(naxes))

        name = None
        axes_type = "CUSTOM"

        if isinstance(unit, (u.Unit, u.IrreducibleUnit)):
            unit = tuple([unit] * naxes)

        if all([u.m.is_equivalent(un) for un in unit]):
            axes_type = "SPATIAL"

        if all([u.pix.is_equivalent(un) for un in unit]):
            name = "DetectorFrame"
            axes_type = "PIXEL"

        axes_type = tuple([axes_type] * naxes)

        return cf.CoordinateFrame(naxes, axes_type, axes_order, unit=unit,
                                  axes_names=names, name=name, axis_physical_types=physical_types)

    @singledispatchmethod
    @classmethod
    def _parse_lookup_table(cls, lookup_table, **kwargs):
        raise NotImplementedError(f"Can not generate a lookup table from input of type {type(lookup_table)}.")

    @_parse_lookup_table.register(Time)
    @classmethod
    def from_time(cls, lookup_table, mesh=False, names=None, physical_types=None, **kwargs):
        if mesh:
            raise ValueError("Can not use mesh=True with Time objects.")

    @_parse_lookup_table.register(SkyCoord)
    @classmethod
    def from_skycoord(cls, lookup_table, mesh=False, names=None, physical_types=None, **kwargs):
        pass

    @_parse_lookup_table.register(list)
    @_parse_lookup_table.register(tuple)
    @_parse_lookup_table.register(u.Quantity)
    @classmethod
    def _from_quantity(cls, lookup_table, mesh=False, names=None, physical_types=None, frame_type="auto"):
        naxes = 1
        if isinstance(lookup_table, (list, tuple)):
            if not all((isinstance(x, u.Quantity) for x in lookup_table)):
                raise TypeError("Can only parse a list or tuple of u.Quantity objects.")

            naxes = len(lookup_table)
            try:
                combined = u.Quantity(lookup_table)
                unit = combined.unit
            except u.UnitsError:
                unit = tuple(lt.unit for lt in lookup_table)

            model = cls._generate_compound_model(*lookup_table, mesh=mesh)

        else:
            unit = lookup_table.unit

            model = cls.generate_tabular(lookup_table)

        frame = cls._generate_generic_frame(naxes, unit, names, physical_types)

        if frame_type == "spectral":
            if not isinstance(lookup_table, u.Quantity):
                raise TypeError("Can not generate a spectral frame with more than one lookup table")
            if not unit.is_equivalent(u.Hz, equivalencies=u.spectral()):
                raise u.UnitsError("The provided quantity is not compatible with a spectral type.")
            frame = cf.SpectralFrame(unit=(unit,), axes_names=names, axis_physical_types=physical_types)

        return model, frame
