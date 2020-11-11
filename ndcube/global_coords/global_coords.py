from collections.abc import Mapping

import astropy.units as u
from astropy.modeling import models
from astropy.wcs.wcsapi import BaseHighLevelWCS, BaseLowLevelWCS
from astropy.wcs.wcsapi.sliced_low_level_wcs import sanitize_slices

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

    def add(self, name, physical_type):
        """
        Add a new coordinate to the collection.
        """
        if len(self.mapping) > 1:
            self.mapping[name] = wcs.world_axis_name or physical_type

    def remove(self, name):
        """
        Remove a coordinate from the collection
        """
        if len(self.mapping) > 1:
            del self.mapping[name]

    @property
    def names(self):
        """
        A tuple of all the names.
        """
        if len(self.mapping) >= 1:
            return [*self.mapping]

    @property
    def physical_types(self):
        """
        A tuple of all physical types, one per coordinate.
        """

    def keys(self):
        """
        A set-like of all names in this collection.
        """

    def values(self):
        """
        A set-like of all values in this collection
        """

    def __getitem__(self, item):
        """
        Index the collection by a name.
        """
        return self.mapping[item]

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)
