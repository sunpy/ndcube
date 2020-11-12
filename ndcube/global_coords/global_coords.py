from collections import OrderedDict
from collections.abc import Mapping

__all__ = ['GlobalCoords']


class GlobalCoords(Mapping):
    """
    A structured representation of coordinate information applicable to a whole NDCube.
    """
    def __init__(self, ndcube):
        """
        Init method.
        """
        super().__init__()
        self._ndcube = ndcube

        # Set values using the setters for validation
        self._internal_coords = OrderedDict()

    @property
    def _all_coords(self):
        """
        Establish the _all_coords property with an _internal_coords.
        """
        return self._internal_coords

    def add(self, name, physical_type, coords):
        """
        Add a new coordinate to the collection.
        """
        self._internal_coords[str(name)] = (name, physical_type, coords)

    def remove(self, name):
        """
        Remove a coordinate from the collection
        """
        del self._internal_coords[str(name)]

    @property
    def names(self):
        """
        A mutable list of all the names or keys.
        """
        return list(self._internal_coords.keys())

    @property
    def physical_types(self):
        """
        A mutable list of all physical types, one per coordinate.
        """
        physical_types_list = []
        for item in self._internal_coords.values():
            physical_types_list.append(item[1])
        return physical_types_list

    @property
    def coords(self):
        """
        A mutable list of all coords, one per coordinate.
        """
        coords_list = []
        for item in self._internal_coords.values():
            coords_list.append(item[2])
        return coords_list

    def __getitem__(self, item):
        """
        Index the collection by a name.
        """
        return self._internal_coords[item]

    def __iter__(self):
        """
        Iterate over the collection.
        """
        return iter(self._internal_coords)

    def __len__(self):
        """
        Establish the length of the collection.
        """
        return len(self._internal_coords)
