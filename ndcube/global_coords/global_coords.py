from collections.abc import Mapping

from ndcube import NDCube

__all__ = ['GlobalCoords']


class GlobalCoords(Mapping):
    """
    A structured representation of coordinate information applicable to a whole NDCube.
    """
    def __init__(self, NDCube):
        super().__init__()

        # Set values using the setters for validation
        self.mapping = {}

    def add(self, name, physical_type, gcoords):
        """
        Add a new coordinate to the collection.
        """
        try:
            self.mapping[name] = (name, physical_type, gcoords)
        except TypeError:
            self.mapping[name] = (physical_type, physical_type, gcoords)

    def remove(self, name):
        """
        Remove a coordinate from the collection
        """
        del self.mapping[name]

    @property
    def names(self):
        """
        A tuple of all the names or keys.
        """
        return tuple([*self.mapping.keys()])

    @property
    def physical_types(self):
        """
        A tuple of all physical types, one per coordinate.
        """
        return tuple(*self.mapping[1])

    def keys(self):
        """
        A set-like of all names in this collection.
        """
        return set(self.mapping.keys())

    def values(self):
        """
        A set-like of all values in this collection
        """
        return set(self.mapping.values())

    def __getitem__(self, item):
        """
        Index the collection by a name.
        """
        return self.mapping[item]

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)
