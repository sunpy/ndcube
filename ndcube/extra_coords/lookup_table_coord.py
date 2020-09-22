import copy

import astropy.units as u
import gwcs
import gwcs.coordinate_frames as cf
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.modeling import models
from astropy.modeling.models import tabular_model
from astropy.time import Time

__all__ = ['LookupTableCoord']


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
