from collections.abc import Mapping

from ndcube import NDCube

__all__ = ['GlobalCoords']


class GlobalCoords(Mapping):
    """
    A structured representation of coordinate information applicable to a whole NDCube.
    """
    def __init__(self, ndcube):
        super().__init__()
        self._ndcube = ndcube

        # Set values using the setters for validation
        self._internal_coords = {}
        
    @property
    def _all_coords(self):
        return self._internal_coords

    def add(self, name, physical_type, coords):
        """
        Add a new coordinate to the collection.
        """
        self.mapping[name] = [name, physical_type, coords]

    def remove(self, name):
        """
        Remove a coordinate from the collection
        """
        del self.mapping[name]

    @property
    def names(self):
        """
        A mutable list of all the names or keys.
        """
        return [*self.mapping.keys()]

    @property
    def physical_types(self):
        """
        A mutable list of all physical types, one per coordinate.
        """
        return [*self.mapping[1]]

    def __getitem__(self, item):
        """
        Index the collection by a name.
        """
        return self.mapping[item]

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)
