import abc
import copy
from numbers import Integral
from collections import defaultdict

import astropy.units as u
import gwcs
import gwcs.coordinate_frames as cf
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.modeling import models
from astropy.modeling.models import tabular_model
from astropy.time import Time

__all__ = ['TimeTableCoordinate', 'SkyCoordTableCoordinate', 'QuantityTableCoordinate']


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


def _generate_tabular(lookup_table, interpolation='linear', points_unit=u.pix, **kwargs):
    """
    Generate a Tabular model class and instance.
    """
    if not isinstance(lookup_table, u.Quantity):
        raise TypeError("lookup_table must be a Quantity.")  # pragma: no cover

    ndim = lookup_table.ndim
    if ndim == 0:
        lookup_table = u.Quantity([lookup_table])
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


def _generate_compound_model(*lookup_tables, mesh=True):
    """
    Takes a set of quantities and returns a ND compound model.
    """
    model = _generate_tabular(lookup_tables[0])
    for lt in lookup_tables[1:]:
        model = model & _generate_tabular(lt)

    if mesh:
        return model

    # If we are not meshing the inputs duplicate the inputs across all models
    mapping = list(range(lookup_tables[0].ndim)) * len(lookup_tables)
    return models.Mapping(mapping) | model


def _model_from_quantity(lookup_tables, mesh=False):
    if len(lookup_tables) > 1:
        return _generate_compound_model(*lookup_tables, mesh=mesh)

    return _generate_tabular(lookup_tables[0])


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
        self.names = names if not isinstance(names, str) else [names]
        self.physical_types = physical_types if not isinstance(physical_types, str) else [physical_types]
        self._dropped_world_dimensions = defaultdict(list)
        self._dropped_world_dimensions["world_axis_object_classes"] = dict()

    @abc.abstractmethod
    def __getitem__(self, item):
        pass  # pragma: no cover

    def __and__(self, other):
        if not isinstance(other, BaseTableCoordinate):
            return NotImplemented

        if isinstance(other, MultipleTableCoordinate):
            # By returning NotImplemented here we trigger python calling
            # __rand__ on LookupTableCoord, which will work if other is a
            # BaseTableCoordinate but fail otherwise
            return NotImplemented

        return MultipleTableCoordinate(self, other)

    def __str__(self):
        return str(self.table)

    def __repr__(self):
        return f"{object.__repr__(self)}\n{self}"

    @property
    @abc.abstractmethod
    def n_inputs(self):
        """
        Number of pixel dimensions in this table.
        """

    @abc.abstractmethod
    def is_scalar(self):
        """
        Return a boolean if this coordinate is a scalar.

        This is used by `.MultipleTableCoordinate` and `.ExtraCoords` to know
        if the dimension has been "dropped".
        """

    @property
    @abc.abstractmethod
    def frame(self):
        """
        Generate the Frame for this LookupTable.
        """

    @property
    @abc.abstractmethod
    def model(self):
        """
        Generate the Astropy Model for this LookupTable.
        """

    @property
    def wcs(self):
        """
        A gWCS object representing all the coordinates.
        """
        model = self.model
        return gwcs.WCS(forward_transform=model,
                        input_frame=_generate_generic_frame(model.n_inputs, u.pix),
                        output_frame=self.frame)

    @property
    def dropped_world_dimensions(self):
        return self._dropped_world_dimensions


class QuantityTableCoordinate(BaseTableCoordinate):
    """
    A lookup table made up of N Quantity objects.

    This class can either be instantiated with N ND arrays (i.e. the output of
    `numpy.meshgrid`) or N 1D arrays (i.e. the input to `numpy.meshgrid`).

    Notes
    -----
    The reason for supporting both the input and output of meshgrid is that
    meshgrid isn't called when ``mesh=True``, the "meshing" is done in the gWCS
    layer.
    """
    def __init__(self, *tables, mesh=False, names=None, physical_types=None):
        if not all([isinstance(t, u.Quantity) for t in tables]):
            raise TypeError("All tables must be astropy Quantity objects")
        if not all([t.unit.is_equivalent(tables[0].unit) for t in tables]):
            raise u.UnitsError("All tables must have equivalent units.")

        if isinstance(names, str):
            names = [names]
        if isinstance(physical_types, str):
            physical_types = [physical_types]

        if names is not None and len(names) != len(tables):
            raise ValueError("The number of names should match the number of world dimensions")
        if physical_types is not None and len(physical_types) != len(tables):
            raise ValueError("The number of physical types should match the number of world dimensions")

        self.unit = tables[0].unit

        super().__init__(*tables, mesh=mesh, names=names, physical_types=physical_types)

    def _slice_table(self, i, table, item, new_components, whole_slice):
        # If mesh is True then we can drop a dimension
        if isinstance(item, Integral) and (
                isinstance(whole_slice, tuple) and
                not(all(isinstance(i, Integral) for i in whole_slice))):
            dwd = new_components["dropped_world_dimensions"]
            dwd["value"].append(table[item])
            dwd["world_axis_names"].append(self.names[i] if self.names else None)
            dwd["world_axis_physical_types"].append(self.frame.axis_physical_types[i])
            dwd["world_axis_units"].append(table.unit.to_string())
            dwd["world_axis_object_components"].append((f"quantity{i}", 0, "value"))
            dwd["world_axis_object_classes"].update({f"quantity{i}": (u.Quantity, tuple(), {"unit", table.unit.to_string()})})
            return

        new_components["tables"].append(table[item])
        if self.names:
            new_components["names"].append(self.names[i])
        if self.physical_types:
            new_components["physical_types"].append(self.physical_types[i])

    def __getitem__(self, item):
        if isinstance(item, (slice, Integral)):
            item = (item,)
        if not (len(item) == len(self.table) or len(item) == self.table[0].ndim):
            raise ValueError("Can not slice with incorrect length")

        new_components = defaultdict(list)
        new_components["dropped_world_dimensions"] = copy.deepcopy(self._dropped_world_dimensions)

        if self.mesh:
            for i, (ele, table) in enumerate(zip(item, self.table)):
                self._slice_table(i, table, ele, new_components, whole_slice=item)
        else:
            for i, table in enumerate(self.table):
                self._slice_table(i, table, item, new_components, whole_slice=item)

        names = new_components["names"] or None
        physical_types = new_components["physical_types"] or None

        ret_table = type(self)(*new_components["tables"], mesh=self.mesh, names=names, physical_types=physical_types)
        ret_table._dropped_world_dimensions = new_components["dropped_world_dimensions"]
        return ret_table

    @property
    def n_inputs(self):
        return len(self.table)

    def is_scalar(self):
        return all(t.shape == tuple() for t in self.table)

    @property
    def frame(self):
        """
        Generate the Frame for this LookupTable.
        """
        return _generate_generic_frame(len(self.table), self.unit, self.names, self.physical_types)

    @property
    def model(self):
        """
        Generate the Astropy Model for this LookupTable.
        """
        return _model_from_quantity(self.table, self.mesh)


class SkyCoordTableCoordinate(BaseTableCoordinate):
    """
    A lookup table created from a `~astropy.coordinates.SkyCoord`.

    If mesh is `True` in this class then `numpy.meshgrid` *is* called when the
    class is constructed, this is to allow slicing operations on the tables
    which make the length of the dimensions different.
    """
    def __init__(self, *tables, mesh=False, names=None, physical_types=None):
        if not len(tables) == 1 and isinstance(tables[0], SkyCoord):
            raise ValueError("SkyCoordLookupTable can only be constructed from a single SkyCoord object")

        if isinstance(names, str):
            names = [names]
        if names is not None and len(names) != 2:
            raise ValueError("The number of names must equal two for a SkyCoord table.")
        if physical_types is not None and len(physical_types) != 2:
            raise ValueError("The number of physical types must equal two for a SkyCoord table.")

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

    def is_scalar(self):
        return self.table.shape == tuple()

    def __getitem__(self, item):
        if not (isinstance(item, (slice, Integral)) or len(item) == self.table.ndim):
            raise ValueError("Can not slice with incorrect length")

        return type(self)(self.table[item],
                          mesh=False,
                          names=self.names,
                          physical_types=self.physical_types)

    @property
    def frame(self):
        """
        Generate the Frame for this LookupTable.
        """
        sc = self.table
        components = tuple(getattr(sc.data, comp) for comp in sc.data.components)
        ref_frame = sc.frame.replicate_without_data()
        units = list(c.unit for c in components)

        # TODO: Currently this limits you to 2D due to gwcs#120
        return cf.CelestialFrame(reference_frame=ref_frame,
                                 unit=units,
                                 axes_names=self.names,
                                 axis_physical_types=self.physical_types,
                                 name="CelestialFrame")

    @property
    def model(self):
        """
        Generate the Astropy Model for this LookupTable.
        """
        sc = self.table
        components = tuple(getattr(sc.data, comp) for comp in sc.data.components)
        return _model_from_quantity(components, mesh=self.mesh)


class TimeTableCoordinate(BaseTableCoordinate):
    """
    A lookup table based on a `~astropy.time.Time`, will always be one dimensional.
    """
    def __init__(self, *tables, names=None, physical_types=None, reference_time=None):
        if not len(tables) == 1 and isinstance(tables[0], Time):
            raise ValueError("TimeLookupTable can only be constructed from a single Time object.")

        if isinstance(names, str):
            names = [names]
        if isinstance(physical_types, str):
            physical_types = [physical_types]

        if names is not None and len(names) != 1:
            raise ValueError("A Time coordinate can only have one name.")
        if physical_types is not None and len(physical_types) != 1:
            raise ValueError("A Time coordinate can only have one physical type.")

        super().__init__(*tables, mesh=False, names=names, physical_types=physical_types)
        self.table = self.table[0]
        self.reference_time = reference_time or self.table[0]

    def __getitem__(self, item):
        if not (isinstance(item, (slice, Integral)) or len(item) == 1):
            raise ValueError("Can not slice with incorrect length")

        return type(self)(self.table[item],
                          names=self.names,
                          physical_types=self.physical_types,
                          reference_time=self.reference_time)

    @property
    def n_inputs(self):
        return 1  # The time table has to be one dimensional

    def is_scalar(self):
        return self.table.shape == tuple()

    @property
    def frame(self):
        """
        Generate the Frame for this LookupTable.
        """
        return cf.TemporalFrame(self.reference_time,
                                unit=u.s,
                                axes_names=self.names,
                                name="TemporalFrame")

    @property
    def model(self):
        """
        Generate the Astropy Model for this LookupTable.
        """
        time = self.table
        deltas = (time - self.reference_time).to(u.s)

        return _model_from_quantity((deltas,), mesh=False)


class MultipleTableCoordinate(BaseTableCoordinate):
    """
    A Holder for multiple multiple `.BaseTableCoordinate` objects.

    This class allows the generation of a gWCS from many `.BaseTableCoordinate`
    objects.

    Parameters
    ----------
    lookup_tables : `BaseTableCoordinate`
        One or more lookup table coordinate classes to combine into a gWCS
        object.

    Notes
    -----
    The most useful method of constructing a ``LookupTableCoord`` class is to
    combine multiple instances of `.BaseTableCoordinate` with the ``&``
    operator.
    """
    def __init__(self, *table_coordinates):
        if not all(isinstance(lt, BaseTableCoordinate) and
                   not(isinstance(lt, MultipleTableCoordinate)) for lt in table_coordinates):
            raise TypeError("All arguments must be BaseTableCoordinate instances, such as QuantityTableCoordinate, "
                            "and not instances of MultipleTableCoordinate.")
        self._table_coords = list(table_coordinates)
        self._dropped_coords = list()

    def __str__(self):
        return f"MultipleTableCoordinate(tables=[{', '.join([str(t) for t in self._table_coords])}])"

    def __and__(self, other):
        if not isinstance(other, BaseTableCoordinate):
            return NotImplemented

        if isinstance(other, MultipleTableCoordinate):
            others = other._table_coords
        else:
            others = [other]

        return type(self)(*(self._table_coords + others))

    def __rand__(self, other):
        # This method should never be called if the left hand operand is a MultipleTableCoordinate
        if not isinstance(other, BaseTableCoordinate) or isinstance(other, MultipleTableCoordinate):
            return NotImplemented

        return type(self)(*([other] + self._table_coords))

    def __getitem__(self, item):
        if isinstance(item, (slice, Integral)):
            item = (item,)

        if not len(item) == self.n_inputs:
            raise ValueError(
                f"length of the slice ({len(item)}) must match the number of coordinates {self.n_inputs}")

        new_tables = []
        dropped_tables = []
        i = 0
        for table in self._table_coords:
            tslice = item[i:i+table.n_inputs]
            i += table.n_inputs
            new_table = table[tslice]
            if new_table.is_scalar():
                dropped_tables.append(new_table)
            else:
                new_tables.append(new_table)

        new = MultipleTableCoordinate(*new_tables)
        new._dropped_coords = dropped_tables
        return new

    @property
    def n_inputs(self):
        return sum(t.n_inputs for t in self._table_coords)

    def is_scalar(self):
        return False

    @property
    def model(self):
        """
        The combined astropy model for all the lookup tables.
        """
        model = self._table_coords[0].model
        for m2 in self._table_coords[1:]:
            model = model & m2.model
        return model

    @property
    def frame(self):
        """
        The gWCS coordinate frame for all the lookup tables.
        """
        if len(self._table_coords) == 1:
            return self._table_coords[0].frame
        else:
            frames = [t.frame for t in self._table_coords]

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
    def dropped_world_dimensions(self):
        dropped_world_dimensions = defaultdict(list)
        dropped_world_dimensions["world_axis_object_classes"] = dict()

        # Combine the dicts on the tables with our dict
        for lutc in self._table_coords:
            for key, value in lutc.dropped_world_dimensions.items():
                if key == "world_axis_object_classes":
                    dropped_world_dimensions[key].update(value)
                else:
                    dropped_world_dimensions[key] += value

        # Generate the dict from the dropped tables
        for dropped in self._dropped_coords:
            wao_classes = dropped.frame._world_axis_object_classes
            wao_components = dropped.frame._world_axis_object_components

            for i in range(dropped.frame.naxes):
                dropped_world_dimensions["world_axis_names"].append(dropped.frame.axes_names[i] or None)
                dropped_world_dimensions["world_axis_physical_types"].append(dropped.frame.axis_physical_types[i] or None)
                dropped_world_dimensions["world_axis_units"].append(dropped.frame.unit[i].to_string())
                dropped_world_dimensions["world_axis_object_components"].append(wao_components[i])
                dropped_world_dimensions["world_axis_object_classes"].update(dict(
                    filter(
                        lambda x: x[0] == wao_components[i][0], wao_classes.items()
                    )
                ))
                coord = dropped.frame.coordinate_to_quantity(dropped.table)
                if dropped.frame.naxes > 1 or (isinstance(coord, (tuple, list)) and len(coord) == 1):
                    coord = coord[i]
                dropped_world_dimensions["value"].append(coord)

        return dropped_world_dimensions
