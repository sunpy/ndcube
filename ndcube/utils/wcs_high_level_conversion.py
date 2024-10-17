"""
This module provides helpers introduced in https://github.com/astropy/astropy/pull/11950.
"""
from functools import wraps

import astropy.wcs.wcsapi.high_level_api as _hlvl
from astropy.utils.decorators import deprecated

__all__ = ['values_to_high_level_objects', 'high_level_objects_to_values']


@deprecated("2.2.3", "Please use astropy.wcs.wcsapi.high_level_api.values_to_high_level_objects", warning_type=DeprecationWarning)
@wraps(_hlvl.values_to_high_level_objects)
def values_to_high_level_objects(*args, **kwargs):
    """
    Convert low level values into high level objects.

    This function uses the information in ``wcs.world_axis_object_classes`` and
    ``wcs.world_axis_object_components`` to convert low level "values"
    `~.Quantity` objects, to high level objects (such as `~.SkyCoord`).

    This is used in `.HighLevelWCSMixin.pixel_to_world`, but provided as a
    separate function for use in other places where needed.

    Parameters
    ----------
    *world_values: object
        Low level, "values" representations of the world coordinates.

    low_level_wcs: `.BaseLowLevelWCS`
        The WCS object to use to interpret the coordinates.
    """
    return _hlvl.values_to_high_level_objects(*args, **kwargs)


@deprecated("2.2.3", "Please use astropy.wcs.wcsapi.high_level_api.high_level_objects_to_values", warning_type=DeprecationWarning)
@wraps(_hlvl.high_level_objects_to_values)
def high_level_objects_to_values(*args, **kwargs):
    """
    Convert the input high level object to low level values.

    This function uses the information in ``wcs.world_axis_object_classes`` and
    ``wcs.world_axis_object_components`` to convert the high level objects
    (such as `~.SkyCoord`) to low level "values" `~.Quantity` objects.

    This is used in `.HighLevelWCSMixin.world_to_pixel`, but provided as a
    separate function for use in other places where needed.

    Parameters
    ----------
    *world_objects: object
        High level coordinate objects.

    low_level_wcs: `.BaseLowLevelWCS`
        The WCS object to use to interpret the coordinates.
    """
    return _hlvl.high_level_objects_to_values(*args, **kwargs)
