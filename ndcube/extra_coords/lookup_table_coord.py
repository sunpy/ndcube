import abc
from numbers import Integral

import astropy.units as u
import gwcs
import gwcs.coordinate_frames as cf
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.modeling import models
from astropy.modeling.models import tabular_model
from astropy.time import Time
from astropy.wcs.wcsapi.wrappers.sliced_wcs import sanitize_slices

__all__ = ['LookupTableCoord']


def _generate_generic_frame(naxes, unit, names=None, physical_types=None):
    """
    Generate a simple frame, where all axes have the same type and unit.
    """
    axes_order = tuple(range(naxes))

    name = None
    axes_type = "CUSTOM"

    if isinstance(unit, (u.Unit, u.IrreducibleUnit, u.CompositeUnit)):
        unit = tuple([unit] * naxes)

    if all([u.m.is_equivalent(un) for un in unit]):
        axes_type = "SPATIAL"

    if all([u.pix.is_equivalent(un) for un in unit]):
        name = "PixelFrame"
        axes_type = "PIXEL"

    axes_type = tuple([axes_type] * naxes)

    return cf.CoordinateFrame(naxes, axes_type, axes_order, unit=unit,
                              axes_names=names, name=name, axis_physical_types=physical_types)


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
    mesh: `bool`
        If `True` treat the inputs as coordinate vectors per axis. Has the
        equivalent effect of passing the inputs to `numpy.meshgrid` but without
        generating the arrays.
    names: iterable of `str`, optional
        The world coordinate names to be passed to the `gwcs` coordinate frame.
    physical_types: iterable of `str`, optional
        The world axis physical types, to be passed to the gWCS frames. This
        can be used to override the defaults generated by `gwcs`.

    Attributes
    ----------
    delayed_models: `list` of `.DelayedLookupTable`
        A list of all the lookup tables contained by this object.
    frames: `list` of `gwcs.coordinate_frames.CoordinateFrame`
        A list of all the gwcs coordinate frame objects corresponding to the
        lookup tables.

    Notes
    -----

    The constructor to this class expects a single coordinate to be passed,
    i.e. one coordinate frame (e.g. Celestial), although this coordinate could be built
    from multiple lookup tables.

    The implementation of this class, however, allows the representation of
    many coordinate frames in a single instance. This is primarily to allow the
    ``&`` operator to work between instances of this class, and therefore build
    up a multi-dimensional WCS object.

    If you wish to build a multi-dimensional `.LookupTableCoord` the correct
    approach is to construct two instances and then join them into a new
    combined instance with the ``&`` operator.
    """

    def __init__(self, *lookup_tables, mesh=True, names=None, physical_types=None):
        self._lookup_tables = []
        self._dropped_world_dimensions = None

        if lookup_tables:
            lt0 = lookup_tables[0]
            if not all(isinstance(lt, type(lt0)) for lt in lookup_tables):
                raise TypeError("All lookup tables must be the same type")

            type_map = {
                u.Quantity: QuantityTableCoordinate,
                SkyCoord: SkyCoordTableCoordinate,
                Time: TimeTableCoordinate,
            }

            self._lookup_tables.append(type_map[type(lt0)](*lookup_tables,
                                                           mesh=mesh,
                                                           names=names,
                                                           physical_types=physical_types))

    def __str__(self):
        return f"LookupTableCoord(tables={self._lookup_tables})"

    def __repr__(self):
        return f"{object.__repr__(self)}\n{self}"

    def __and__(self, other):
        if not isinstance(other, LookupTableCoord):
            raise TypeError(
                "Can only concatenate LookupTableCoord objects with other LookupTableCoord objects.")

        new_lutc = type(self)()
        new_lutc._lookup_tables = self._lookup_tables + other._lookup_tables

        return new_lutc

    @staticmethod
    def _append_frame_to_dropped_dimensions(dropped_world_dimensions, frame):
        if "world_axis_object_classes" not in dropped_world_dimensions:
            dropped_world_dimensions["world_axis_object_classes"] = dict()

        wao_classes = frame._world_axis_object_classes
        wao_components = frame._world_axis_object_components

        dropped_world_dimensions["world_axis_names"].append(frame.axes_names)
        dropped_world_dimensions["world_axis_physical_types"].append(frame.world_axis_physical_types)
        dropped_world_dimensions["world_axis_units"].append(frame.world_axis_units)
        dropped_world_dimensions["world_axis_object_components"].append(wao_components)
        dropped_world_dimensions["world_axis_object_classes"].update(dict(
            filter(
                lambda x: x[0] == wao_components[0][0], wao_classes.items()
            )
        ))
        dropped_world_dimensions["serialized_classes"] = False

        return dropped_world_dimensions

    def __getitem__(self, item):
        """
        Apply a given slice to all the lookup tables stored in this object.

        If no lookup tables remain after slicing `None` is returned.
        """
        item = sanitize_slices(item, self.model.n_inputs)

        ind = 0
        new_coords = []
        for coord in self._lookup_tables:
            n_axes = coord.n_inputs

            # Extract the parts of the slice that correspond to this model
            sub_items = tuple(item[i] for i in range(ind, ind + n_axes))
            ind += n_axes

            # If all the slice elements are ints then we are dropping this model
            if not all(isinstance(it, Integral) for it in sub_items):
                new = coord[sub_items]
                if new is None:
                    raise ValueError("You have no power here...")
                new_coords.append(new)

        if not new_coords:
            return

        # Return a new instance with the smaller tables
        new_lutc = type(self)()
        new_lutc._lookup_tables = new_coords
        return new_lutc

    @property
    def model(self):
        model = self._lookup_tables[0].generate_model()
        for m2 in self._lookup_tables[1:]:
            model = model & m2.generate_model()
        return model

    @property
    def frame(self):
        if len(self._lookup_tables) == 1:
            return self._lookup_tables[0].generate_frame()
        else:
            frames = [t.generate_frame() for t in self._lookup_tables]

            # We now have to set the axes_order of all the frames so that we
            # have one consistent WCS with the correct number of pixel
            # dimensions.
            ind = 0
            for f in frames:
                new_ind = ind + f.naxes
                f._axes_order = tuple(range(ind, new_ind))
                ind = new_ind

            return cf.CompositeFrame(frames)

    @property
    def wcs(self):
        return gwcs.WCS(forward_transform=self.model,
                        input_frame=_generate_generic_frame(self.model.n_inputs, u.pix),
                        output_frame=self.frame)

    @property
    def dropped_word_dimensions(self):
        return self._dropped_world_dimensions or {}


class BaseTableCoordinate(abc.ABC):
    """
    A Base LookupTable contains a single lookup table coordinate.

    This can be multi-dimensional, to support use cases for coupled dimensions,
    such as SkyCoord, or a 3D grid of distances where three 1D lookup tables
    are supplied for each of the axes. The upshot of this is that each
    BaseLookupTable has only one gWCS frame.

    The contrasts with LookupTableCoord which can contain multiple physical
    coordinates, meaning it can have multiple gWCS frames.
    """
    def __init__(self, *tables, mesh=False, names=None, physical_types=None):
        self.table = tables
        self.mesh = mesh
        self.names = names
        self.physical_types = physical_types

    @property
    @abc.abstractmethod
    def n_inputs(self):
        pass

    @abc.abstractmethod
    def __getitem__(self, item):
        pass

    @staticmethod
    def generate_tabular(lookup_table, interpolation='linear', points_unit=u.pix, **kwargs):
        """
        Generate a Tabular model class and instance.
        """
        if not isinstance(lookup_table, u.Quantity):
            raise TypeError("lookup_table must be a Quantity.")  # pragma: no cover

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

        t = TabularND(points, lookup_table, **kwargs)

        # TODO: Remove this when there is a new gWCS release
        # Work around https://github.com/spacetelescope/gwcs/pull/331
        t.bounding_box = None

        return t

    @classmethod
    def generate_compound_model(cls, *lookup_tables, mesh=True):
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

    def model_from_quantity(self, lookup_tables, mesh=False):
        if len(lookup_tables) > 1:
            return self.generate_compound_model(*lookup_tables, mesh=mesh)

        return self.generate_tabular(lookup_tables[0])

    @abc.abstractmethod
    def generate_frame(self):
        """
        Generate the Frame for this LookupTable.
        """

    @abc.abstractmethod
    def generate_model(self):
        """
        Generate the Astropy Model for this LookupTable.
        """


class QuantityTableCoordinate(BaseTableCoordinate):
    def __init__(self, *tables, mesh=False, names=None, physical_types=None):
        if not all([isinstance(t, u.Quantity) for t in tables]):
            raise TypeError("All Tables must be astropy Quantity objects")
        if not all([t.unit.is_equivalent(tables[0].unit) for t in tables]):
            raise u.UnitsError("All lookup tables must have equivalent units.")

        if isinstance(names, str):
            names = [names]

        if names is not None and len(names) != len(tables):
            raise ValueError("The number of names should match the number of world dimensions")

        self.unit = tables[0].unit

        super().__init__(*tables, mesh=mesh, names=names, physical_types=physical_types)

    @property
    def n_inputs(self):
        return len(self.table)

    def __getitem__(self, item):
        if isinstance(item, (slice, Integral)):
            item = (item,)
        if not (len(item) == len(self.table) or len(item) == self.table[0].ndim):
            raise ValueError("Can not slice with incorrect length")

        tables = []
        names = []
        physical_types = []

        if self.mesh:
            for i, (ele, table) in enumerate(zip(item, self.table)):
                # if isinstance(ele, Integral):
                #     continue
                tables.append(table[ele])
                if self.names:
                    names.append(self.names[i])
                if self.physical_types:
                    physical_types.append(self.physical_types[i])
        else:
            for i, table in enumerate(self.table):
                tables.append(table[item])
                if self.names:
                    names.append(self.names[i])
                if self.physical_types:
                    physical_types.append(self.physical_types[i])

        names = names or None
        physical_types = physical_types or None

        return type(self)(*tables, mesh=self.mesh, names=names, physical_types=physical_types)

    def generate_frame(self):
        """
        Generate the Frame for this LookupTable.
        """
        return _generate_generic_frame(len(self.table), self.unit, self.names, self.physical_types)

    def generate_model(self):
        """
        Generate the Astropy Model for this LookupTable.
        """
        return self.model_from_quantity(self.table, self.mesh)


class SkyCoordTableCoordinate(BaseTableCoordinate):
    def __init__(self, *tables, mesh=False, names=None, physical_types=None):
        if not len(tables) == 1 and isinstance(tables[0], SkyCoord):
            raise TypeError("SkyCoordLookupTable can only be constructed from a single SkyCoord object")
        if names is not None and len(names) != 2:
            raise ValueError("The number of names must equal two one for lat one for lon.")

        self._was_meshed = False
        sc = tables[0]

        if mesh:
            components = tuple(getattr(sc.data, comp) for comp in sc.data.components)
            units = [c.unit for c in components]
            coords = [u.Quantity(mesh, unit=unit) for mesh, unit in zip(np.meshgrid(*components), units)]
            sc = SkyCoord(*coords, frame=sc)
            mesh = False
            self._was_meshed = True  # An internal flag to know if we meshed the input

        super().__init__(sc, mesh=mesh, names=names, physical_types=physical_types)
        self.table = self.table[0]

    @property
    def n_inputs(self):
        return self.table.ndim

    def __getitem__(self, item):
        if not (isinstance(item, (slice, Integral)) or len(item) == self.table.ndim):
            raise ValueError("Can not slice with incorrect length")

        return type(self)(self.table[item], mesh=False, names=self.names, physical_types=self.physical_types)

    def generate_frame(self):
        """
        Generate the Frame for this LookupTable.
        """
        sc = self.table
        components = tuple(getattr(sc.data, comp) for comp in sc.data.components)
        ref_frame = sc.frame.replicate_without_data()
        units = list(c.unit for c in components)

        names = self.names
        if self.names and len(self.names) != 2:
            names = None

        # TODO: Currently this limits you to 2D due to gwcs#120
        return cf.CelestialFrame(reference_frame=ref_frame,
                                 unit=units,
                                 axes_names=names,
                                 axis_physical_types=self.physical_types,
                                 name="CelestialFrame")

    def generate_model(self):
        """
        Generate the Astropy Model for this LookupTable.
        """
        sc = self.table
        components = tuple(getattr(sc.data, comp) for comp in sc.data.components)
        return self.model_from_quantity(components, mesh=self.mesh)


class TimeTableCoordinate(BaseTableCoordinate):
    def __init__(self, *tables, mesh=False, names=None, physical_types=None):
        if mesh:
            # Override the default, mesh is meaningless when the length of the
            # table is one anyway.
            mesh = False

        if not len(tables) == 1 and isinstance(tables[0], Time):
            raise TypeError("TimeLookupTable can only be constructed from a single Time object")

        if names is not None and not(isinstance(names, str) or len(names) != 1):
            raise ValueError("A Time coordinate can only have one name")

        if isinstance(names, str):
            names = [names]

        super().__init__(*tables, mesh=mesh, names=names, physical_types=physical_types)
        self.table = self.table[0]

    @property
    def n_inputs(self):
        return 1

    def __getitem__(self, item):
        if not (isinstance(item, (slice, Integral)) or len(item) == 1):
            raise ValueError("Can not slice with incorrect length")

        return type(self)(self.table[item], mesh=self.mesh, names=self.names, physical_types=self.physical_types)

    def generate_frame(self):
        """
        Generate the Frame for this LookupTable.
        """
        return cf.TemporalFrame(self.table[0], unit=u.s, axes_names=self.names, name="TemporalFrame")

    def generate_model(self):
        """
        Generate the Astropy Model for this LookupTable.
        """
        time = self.table
        deltas = (time[1:] - time[0]).to(u.s)
        deltas = deltas.insert(0, 0 * u.s)

        return self.model_from_quantity((deltas,), mesh=False)
