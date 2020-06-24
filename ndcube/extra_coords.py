import copy
from numbers import Number, Integral
from functools import reduce
from collections import defaultdict
from collections.abc import Sequence

import astropy.units as u
import gwcs
import gwcs.coordinate_frames as cf
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.modeling import models
from astropy.modeling.models import tabular_model
from astropy.time import Time
from astropy.wcs.wcsapi.sliced_low_level_wcs import sanitize_slices

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
    wcs : `astropy.wcs.wcsapi.LowLevelWCS`
        The WCS specifying the extra coordinates.
    mapping : `tuple` of `int`
       The mapping between the array dimensions and pixel dimensions in the wcs.

    """
    def __init__(self, *, wcs=None, mapping=None):
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
    def from_lookup_tables(cls, names, pixel_dimensions, lookup_tables):
        """
        Construct an ExtraCoords instance from lookup tables.

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
        """
        Add a coordinate to this ``ExtraCoords`` based on a lookup table.

        Parameters
        ----------
        name : `str`
            The name for this coordinate(s).
        array_dimension : `int`
            The pixel dimension, in the array, to which this lookup table corresponds.
        lookup_table : `object`
            The lookup table.
        """
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
        return {lut[1].wcs.world_axis_names: lut for lut in self._lookup_tables}

    def keys(self):
        keys = []
        for key in self._name_lut_map.keys():
            for k in key:
                keys.append(k)
        return tuple(keys)

    @property
    def mapping(self):
        """
        The mapping of the world dimensions in this ``ExtraCoords`` to pixel
        dimensions in the array.
        """
        if self._mapping:
            return self._mapping

        if not self._lookup_tables:
            return tuple()

        lts = [list([lt[0]] if isinstance(lt[0], Integral) else lt[0]) for lt in self._lookup_tables]
        return tuple(reduce(list.__add__, lts))

    @mapping.setter
    def _set_mapping(self, mapping):
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
    def _set_wcs(self, value):
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
        axis_shifts = defaultdict(lambda: models.Identity(1))
        for lut_axis, lut in self._lookup_tables:
            lut_axes = (lut_axis,) if not isinstance(lut_axis, tuple) else lut_axis
            for item_axis, sub_item in enumerate(item):
                # This slice does not apply to this lookup table
                if item_axis not in lut_axes:
                    continue

                if isinstance(sub_item, slice):
                    if sub_item.start is None:
                        new_lookup_tables.add((lut_axis, lut))
                        continue

                    axis_shifts[item_axis] = models.Shift(sub_item.start * u.pix)
                    new_lookup_tables.add((lut_axis, lut))

                elif isinstance(sub_item, int):
                    # Drop the lut
                    continue

                else:
                    raise ValueError(f"A slice of type {type(sub_item)} for axis {item_axis} is not supported.")

        # Apply any offsets to the models in the lookup tables
        lookup_tables = list(new_lookup_tables)
        new_lookup_tables = []
        for i, (lut_axes, lut) in enumerate(lookup_tables):
            # Append shift model to front of chain.
            if len(lut.models) != 1:
                raise NotImplementedError("PANIC")

            if isinstance(lut_axes, int):
                if lut_axes in axis_shifts:
                    lut = copy.deepcopy(lut)
                    lut.models = [axis_shifts[lut_axes] | lut.models[0]]

            if isinstance(lut_axes, tuple):
                if any((l in axis_shifts for l in lut_axes)):
                    shift = axis_shifts[lut_axes[0]]
                    for axis in lut_axes[1:]:
                        shift = shift & axis_shifts[axis]
                    lut = copy.deepcopy(lut)
                    lut.models = [shift | lut.models[0]]

            new_lookup_tables.append((lut_axes, lut))

        new_extra_coords = type(self)()
        new_extra_coords._lookup_tables = new_lookup_tables
        return new_extra_coords

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._getitem_string(item)

        # item = sanitize_slices(item)
        if self._lookup_tables:
            return self._getitem_lookup_tables(item)

        elif self._wcs:
            raise NotImplementedError("PICNIC")


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

    def __str__(self):
        return f"{self.frames=} {self.models=}"

    def __repr__(self):
        return f"{object.__repr__(self)}\n{self}"

    def __and__(self, other):
        if not isinstance(other, LookupTableCoord):
            raise TypeError(
                "Can only concatenate LookupTableCoord objects with other LookupTableCoord objects.")

        new_lutc = copy.copy(self)
        new_lutc.models += other.models
        new_lutc.frames += other.frames

        # We must now re-index the frames so that they align with the composite frame
        ind = 0
        for f in new_lutc.frames:
            new_ind = ind + f.naxes
            f._axes_order = tuple(range(ind, new_ind))
            ind = new_ind

        return new_lutc

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
            return cf.CompositeFrame(self.frames)

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
        model = self._model_from_quantity((deltas,))
        frame = cf.TemporalFrame(time[0], unit=u.s, axes_names=names, name="TemporalFrame")
        return model, frame

    def _from_skycoord(self, lookup_tables, mesh=False, names=None, physical_types=None, **kwargs):
        if len(lookup_tables) > 1:
            raise ValueError("Can only parse one SkyCoord lookup table.")

        sc = lookup_tables[0]
        components = tuple(getattr(sc.data, comp) for comp in sc.data.components)
        model = self._model_from_quantity(components, mesh=mesh)
        ref_frame = sc.frame.replicate_without_data()
        units = list(c.unit for c in components)
        # TODO: Currently this limits you to 2D due to gwcs#120
        frame = cf.CelestialFrame(reference_frame=ref_frame,
                                  unit=units,
                                  axes_names=names,
                                  axis_physical_types=physical_types,
                                  name="CelestialFrame")
        return model, frame

    def _model_from_quantity(self, lookup_tables, mesh=False):
        if len(lookup_tables) > 1:
            if not all((isinstance(x, u.Quantity) for x in lookup_tables)):
                raise TypeError("Can only parse a list or tuple of u.Quantity objects.")

            return self._generate_compound_model(*lookup_tables, mesh=mesh)

        return self.generate_tabular(lookup_tables[0])

    def _from_quantity(self, lookup_tables, mesh=False, names=None, physical_types=None):
        if not all(lt.unit.is_equivalent(lt[0].unit) for lt in lookup_tables):
            raise u.UnitsError("All lookup tables must have equivalent units.")

        unit = u.Quantity(lookup_tables).unit

        model = self._model_from_quantity(lookup_tables, mesh=mesh)
        frame = self._generate_generic_frame(len(lookup_tables), unit, names, physical_types)

        return model, frame
