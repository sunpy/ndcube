from collections import OrderedDict
from collections.abc import Mapping

__all__ = ['GlobalCoords']


class GlobalCoords(Mapping):
    """
    A structured representation of coordinate information applicable to a whole NDCube.
    """
    def __init__(self):
        """
        Init method.
        """
        super().__init__()
        self._internal_coords = OrderedDict()

    @property
    def _all_coords(self):
        """
        Establish _all_coords as a property returning some _internal_coords.
        """
        return self._internal_coords

    def add(self, name, physical_type, coords):
        """
        Add a new coordinate to the collection.
        """
        if name in self._internal_coords.keys():
            raise ValueError("coordinate with same name already exists: "
                             f"{name}: {self._internal_coords[name]}")
        self._internal_coords[name] = (physical_type, coords)

    def remove(self, name):
        """
        Remove a coordinate from the collection
        """
        del self._internal_coords[name]

    @property
    def names(self):
        """
        A tuple of all the names or keys.
        """
        return tuple(self._all_coords.keys())

    @property
    def physical_types(self):
        """
        A tuple of all physical types, one per coordinate.
        """
        return dict((key, value[0]) for key, value in self.items())

    @property
    def coords(self):
        return dict((key, value[1]) for key, value in self.items())

def get_physical_type(self, name):
    """Return the physical type of a specific coordinate."""
    return self._all_coords[name][0]

def get_coord(self, name):
    """Return value of a specific coordinate."""
    return self._all_coords[name][1]
    def __getitem__(self, item):
        """
        Index the collection by a name.
        """
        return self._all_coords[item]

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
        names_and_coords = zip(self.names, self._all_coords.values())
        return f"GlobalCoords({[(name, coord) for name, coord in names_and_coords]})"

    def __repr__(self):
        return f"{object.__repr__(self)}\n{str(self)}"
