import inspect
from functools import wraps

import astropy.units as u
from astropy.wcs.wcsapi import BaseHighLevelWCS

__all__ = ['sanitise_wcs', 'unique_sorted']


def unique_sorted(iterable):
    """
    Return unique values in the order they are first encountered in the iterable.
    """
    lookup = set()  # a temporary lookup set
    return [ele for ele in iterable if ele not in lookup and lookup.add(ele) is None]


def sanitise_wcs(func):
    """
    A wrapper for NDCube methods to sanitise the wcs argument.

    This decorator is only designed to be used on methods of NDCube.

    It will find the wcs argument, keyword or positional and if it is None, set
    it to `self.wcs`.
    It will then verify that the WCS has a matching number of pixel dimensions
    to the dimensionality of the array. It will finally verify that the object
    passed is a HighLevelWCS object, or an ExtraCoords object.
    """
    # This needs to be here to prevent a circular import
    from ndcube.extra_coords import ExtraCoords

    @wraps(func)
    def wcs_wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        params = sig.bind(*args, **kwargs)
        wcs = params.arguments.get('wcs', None)
        self = params.arguments['self']

        if wcs is None:
            wcs = self.wcs

        if not isinstance(wcs, ExtraCoords):
            if not wcs.pixel_n_dim == self.data.ndim:
                raise ValueError("The supplied WCS must have the same number of "
                                 "pixel dimensions as the NDCube object. "
                                 "If you specified `cube.extra_coords.wcs` "
                                 "please just pass `cube.extra_coords`.")

        if not isinstance(wcs, (BaseHighLevelWCS, ExtraCoords)):
            raise TypeError("wcs argument must be a High Level WCS or an ExtraCoords object.")

        params.arguments['wcs'] = wcs

        return func(*params.args, **params.kwargs)

    return wcs_wrapper


def sanitize_corners(*corners):
    """Sanitize corner inputs to NDCube crop methods."""
    corners = [list(corner) if isinstance(corner, (tuple, list)) else [corner]
               for corner in corners]
    n_coords = [len(corner) for corner in corners]
    if len(set(n_coords)) != 1:
        raise ValueError("All corner inputs must have same number of coordinate objects. "
                         f"Lengths of corner objects: {n_coords}")
    return corners


def convert_quantities_to_units(coords, units):
    """Converts a sequence of Quantities to units used in the WCS.

    Non-Quantity types in the sequence are allowed and ignored.

    Parameters
    ----------
    coords: iterable of `astropy.units.Quantity` or `None`
        The coordinates to be converted.

    units: iterable of `astropy.units.Unit` or `str`
        The units to which the coordinates should be converted.

    Returns
    -------
    converted_coords: iterable of `astropy.units.Quantity` or `None`
        The coordinates converted to the units.
        Non-quantity types remain.
    """
    return [coord.to(unit) if isinstance(coord, u.Quantity) else coord
            for coord, unit in zip(coords, units)]
