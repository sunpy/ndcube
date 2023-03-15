import astropy.units as u

__all__ = ['unique_sorted', 'convert_quantities_to_units']


def unique_sorted(iterable):
    """
    Return unique values in the order they are first encountered in the iterable.
    """
    lookup = set()  # a temporary lookup set
    return [ele for ele in iterable if ele not in lookup and lookup.add(ele) is None]


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
