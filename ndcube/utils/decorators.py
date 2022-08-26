"""
NDCube specific decorators
"""
from functools import wraps

from astropy.nddata import NDData
import astropy.units as u

__all__ = ['check_arithmetic_compatibility']


def check_arithmetic_compatibility(func):
    """
    A decorator to check if an arithmetic operation can
    be performed between a map instance and some other operation.
    """
    @wraps(func)
    def inner(instance, value):
        # This is explicit because it is expected that users will try to do this. This raises
        # a different error because it is expected that this type of operation will be supported
        # in future releases.
        if isinstance(value, NDData):
            return NotImplemented
        try:
            # We want to support operations between numbers and array-like objects. This includes
            # floats, ints, lists (of the aforementioned), arrays, quantities. This test acts as
            # a proxy for these possible inputs. If it can be cast to a unitful quantity, we can
            # do arithmetic with it. Broadcasting or unit mismatches are handled later in the
            # actual operations by numpy and astropy respectively.
            _ = u.Quantity(value, copy=False)
        except TypeError:
            return NotImplemented
        return func(instance, value)
    return inner
