from collections import OrderedDict
from collections.abc import Mapping

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

    @property
    def _all_coords(self):
        """
        A dynamic dictionary of all global coordinates, stored here or derived
        from the ndcube object.
        """
        return self._internal_coords

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
        return f"GlobalCoords({[(name, coord) for name, coord in self.items()]})"

    def __repr__(self):
        return f"{object.__repr__(self)}\n{str(self)}"
