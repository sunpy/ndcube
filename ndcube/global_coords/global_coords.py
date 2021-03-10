import copy
from collections import OrderedDict, defaultdict
from collections.abc import Mapping

from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.wcs.wcsapi.high_level_api import default_order
from astropy.wcs.wcsapi.utils import deserialize_class

from ndcube.utils.wcs import validate_physical_types

__all__ = ['GlobalCoords']


class GlobalCoords(Mapping):
    """
    A structured representation of coordinate information applicable to a whole NDCube.

    Parameters
    ----------
    ndcube : `.NDCube`, optional
        The parent ndcube for this object. Used to extract global coordinates
        from the wcs and extra coords of the ndcube. If not specified only
        coordinates explicitly added will be shown.
    """
    def __init__(self, ndcube=None):
        super().__init__()
        self._ndcube = ndcube
        self._internal_coords = OrderedDict()

    @staticmethod
    def _convert_dropped_to_internal(dropped_dimensions):
        """
        Convert the `~astropy.wcs.wcsapi.wrappers.SlicedLowLevelWCS` style
        ``dropped_world_dimensions`` dictionary to the GlobalCoords internal
        representation.
        """
        # Most of this method is adapted from
        # astropy.wcs.wcsapi.high_level_wcs.HighLevelWCSMixin.pixel_to_world

        new_internal_coords = {}

        world = dropped_dimensions.pop("value")
        components = dropped_dimensions.pop("world_axis_object_components")
        classes = dropped_dimensions.pop("world_axis_object_classes")

        # Deserialize classes
        if dropped_dimensions.get("serialized_classes", False):
            classes_new = {}
            for key, value in classes.items():
                classes_new[key] = deserialize_class(value, construct=False)
            classes = classes_new

        args = defaultdict(list)
        kwargs = defaultdict(dict)

        for i, (key, attr, _) in enumerate(components):
            if isinstance(attr, str):
                kwargs[key][attr] = world[i]
            else:
                while attr > len(args[key]) - 1:
                    args[key].append(None)
                args[key][attr] = world[i]

        # key is the unique names of the classes in the order they appear in components
        for key in default_order(components):
            key_ele = [i for i, components in enumerate(components) if components[0] == key]
            physical_types = [dropped_dimensions["world_axis_physical_types"][i] for i in key_ele]
            # Use name if it's set, drop back to physical type if not
            names = tuple([dropped_dimensions["world_axis_names"][i] or
                           dropped_dimensions["world_axis_physical_types"][i] for i in key_ele])

            # convert lists to strings if a single coordinate
            physical_types = physical_types[0] if len(physical_types) == 1 else tuple(physical_types)
            names = names[0] if len(set(names)) == 1 else names

            klass, ar, kw, *rest = classes[key]
            if len(rest) == 0:
                klass_gen = klass
            elif len(rest) == 1:
                klass_gen = rest[0]
            else:
                raise ValueError("Tuples in world_axis_object_classes should have length 3 or 4")

            high_level_object = klass_gen(*args[key], *ar, **kwargs[key], **kw)

            # Special case SkyCoord to get a pretty name
            if isinstance(high_level_object, SkyCoord):
                names = high_level_object.name

            new_internal_coords[names] = (physical_types, high_level_object)

        return new_internal_coords

    @property
    def _all_coords(self):
        """
        A dynamic dictionary of all global coordinates, stored here or derived
        from the ndcube object.
        """
        if self._ndcube is None:
            return self._internal_coords

        all_coords = {**self._internal_coords}

        if hasattr(self._ndcube.wcs.low_level_wcs, "dropped_world_dimensions"):
            dropped_world = copy.deepcopy(self._ndcube.wcs.low_level_wcs.dropped_world_dimensions)
            if dropped_world:
                wcs_dropped = self._convert_dropped_to_internal(dropped_world)
                all_coords.update(wcs_dropped)

        ec_dropped = self._ndcube.extra_coords.dropped_world_dimensions
        if "value" in ec_dropped:
            all_coords.update(self._convert_dropped_to_internal(ec_dropped))

        return all_coords

    def add(self, name, physical_type, coord):
        """
        Add a new coordinate to the collection.

        Parameters
        ----------
        name : `str`
            The name for the coordinate.
        physical_type : `str`
            An IOVA UCD1+ physical type description for the coordinate
            (http://www.ivoa.net/documents/latest/UCDlist.html). If no matching UCD
            type exists, this can instead be ``"custom:xxx"``, where ``xxx`` is an
            arbitrary string. If not known, can be `None`.
        coord : `object`
            The object describing the coordinate value, for example a
            `~astropy.units.Quantity` or a `~astropy.coordinates.SkyCoord`.
        """
        if name in self._internal_coords.keys():
            raise ValueError("coordinate with same name already exists: "
                             f"{name}: {self._internal_coords[name]}")

        # Ensure the physical type is valid
        validate_physical_types((physical_type,))

        self._internal_coords[name] = (physical_type, coord)

    def remove(self, name):
        """
        Remove a coordinate from the collection.
        """
        del self._internal_coords[name]

    @property
    def physical_types(self):
        """
        A mapping of names to physical types for each coordinate.
        """
        return dict((name, value[0]) for name, value in self._all_coords.items())

    def filter_by_physical_type(self, physical_type):
        """
        Filter this object to coordinates with a given physical type.

        Parameters
        ----------
        physical_type: `str`
            The physical type to filter by.

        Returns
        -------
        `.GlobalCoords`
            A new object storing just the coordinates with the given physical type.
        """
        gc = GlobalCoords()
        gc._internal_coords = dict(filter(lambda x: x[1][0] == physical_type, self._all_coords.items()))
        return gc

    def __getitem__(self, item):
        """
        Index the collection by a name.
        """
        if item not in self._all_coords:
            for key, value in self._all_coords.items():
                if isinstance(key, tuple) and item in key:
                    return value[1]

        return self._all_coords[item][1]

    def __iter__(self):
        """
        Iterate over the collection.
        """
        return iter(self._all_coords)

    def __len__(self):
        """
        Establish the length of the collection.
        """
        return len(self._all_coords)

    def __str__(self):
        elements = [f"{name}: {repr(coord)}" for name, coord in self.items()]
        return f"GlobalCoords({', '.join(elements)})"

    def __repr__(self):
        return f"{object.__repr__(self)}\n{str(self)}"
