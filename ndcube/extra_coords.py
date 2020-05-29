import astropy.units as u
import gwcs
import gwcs.coordinate_frames as cf
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.modeling import models
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
    lookup_tables : `object`
        The lookup tables. If more than one lookup table is specified, it
        should represent one physical coordinate type, i.e "spatial". They must
        all be the same type, shape and unit.
    """
    def __init__(self, *lookup_tables, mesh=True, names=None, physical_types=None):
        lt0 = lookup_tables[0]
        if not all(isinstance(lt, type(lt0)) for lt in lookup_tables):
            raise TypeError("All lookup tables must be the same type")

        if not all(lt0.shape == lt.shape for lt in lookup_tables):
            raise ValueError("All lookup tables must have the same shape")

        type_map = {
            u.Quantity: self._from_quantity,
            Time: self._from_time,
            SkyCoord: self._from_skycoord
        }
        model, frame = type_map[type(lt0)](lookup_tables,
                                           mesh=mesh,
                                           names=names,
                                           physical_types=physical_types)
        self.models = [model]
        self.frames = [frame]

    @property
    def model(self):
        model = self.models[0]
        for m2 in self.models[1:]:
            model = model & m2
        return model

    @property
    def frame(self):
        if len(self.frames) == 1:
            return self.frames[0]
        else:
            return cf.CompoundFrame(self.frames)

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
        if len(points) == 1:
            points = points[0]

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

        if mesh:
            return model

        # If we are not meshing the inputs duplicate the inputs across all models
        mapping = list(range(lookup_tables[0].ndim)) * len(lookup_tables)
        return models.Mapping(mapping) | model

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
            name = "PixelFrame"
            axes_type = "PIXEL"

        axes_type = tuple([axes_type] * naxes)

        return cf.CoordinateFrame(naxes, axes_type, axes_order, unit=unit,
                                  axes_names=names, name=name, axis_physical_types=physical_types)

    def _from_time(self, lookup_tables, mesh=False, names=None, physical_types=None, **kwargs):
        if len(lookup_tables) > 1:
            raise ValueError("Can only parse one time lookup table.")

        time = lookup_tables[0]
        deltas = (time[1:] - time[0]).to(u.s)
        deltas = deltas.insert(0, 0)
        model, _ = self._from_quantity((deltas,))
        frame = cf.TemporalFrame(time[0], unit=u.s, axes_names=names, name="TemporalFrame")
        return model, frame

    def _from_skycoord(self, lookup_tables, mesh=False, names=None, physical_types=None, **kwargs):
        if len(lookup_tables) > 1:
            raise ValueError("Can only parse one SkyCoord lookup table.")

        sc = lookup_tables[0]
        components = tuple(getattr(sc.data, comp) for comp in sc.data.components)
        model, _ = self._from_quantity(components, mesh=mesh)
        ref_frame = sc.frame.replicate_without_data()
        units = list(c.unit for c in components)
        frame = cf.CelestialFrame(reference_frame=ref_frame,
                                  unit=units,
                                  axes_names=names,
                                  axis_physical_types=physical_types,
                                  name="CelestialFrame")
        return model, frame


    def _from_spectral(self, lookup_tables, mesh=False, names=None, physical_types=None, **kwargs):
        pass

    def _from_quantity(self, lookup_tables, mesh=False, names=None, physical_types=None):
        if len(lookup_tables) > 1:
            unit = lookup_tables[0].unit
            if not all((isinstance(x, u.Quantity) for x in lookup_tables)):
                raise TypeError("Can only parse a list or tuple of u.Quantity objects.")
            if not all(lt.unit.is_equivalent(unit) for lt in lookup_tables):
                raise u.UnitsError("All lookup tables must have equivalent units.")

            combined = u.Quantity(lookup_tables)
            unit = combined.unit

            model = self._generate_compound_model(*lookup_tables, mesh=mesh)

        else:
            unit = lookup_tables[0].unit

            model = self.generate_tabular(lookup_tables[0])

        frame = self._generate_generic_frame(len(lookup_tables), unit, names, physical_types)

        return model, frame
