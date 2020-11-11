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
        self.wcs = {}

    def add(self, name, physical_type):
        """
        Add a new coordinate to the collection.
        """
        if len(self.wcs) > 1:
            self.wcs[name] = wcs.world_axis_name or physical_type

    def remove(self, name):
        """
        Remove a coordinate from the collection
        """
        if len(self.wcs) > 1:
            del self.wcs[name]

    @property
    def names(self):
        """
        A tuple of all the names.
        """

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

    def __iter__(self):
        pass
