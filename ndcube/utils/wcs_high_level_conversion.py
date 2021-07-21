"""
This module provides helpers introduced in https://github.com/astropy/astropy/pull/11950.
"""
from collections import OrderedDict, defaultdict

from astropy.wcs.wcsapi.high_level_api import default_order, rec_getattr

__all__ = ['values_to_high_level_objects', 'high_level_objects_to_values']

try:
    from astropy.wcs.wcsapi.high_level_api import high_level_objects_to_values, values_to_high_level_objects

except ImportError:

    def high_level_objects_to_values(*world_objects, low_level_wcs):
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
        # Cache the classes and components since this may be expensive
        serialized_classes = low_level_wcs.world_axis_object_classes
        components = low_level_wcs.world_axis_object_components

        # Deserialize world_axis_object_classes using the default order
        classes = OrderedDict()
        for key in default_order(components):
            if low_level_wcs.serialized_classes:
                classes[key] = deserialize_class(serialized_classes[key],
                                                 construct=False)
            else:
                classes[key] = serialized_classes[key]

        # Check that the number of classes matches the number of inputs
        if len(world_objects) != len(classes):
            raise ValueError("Number of world inputs ({}) does not match "
                             "expected ({})".format(len(world_objects), len(classes)))

        # Determine whether the classes are uniquely matched, that is we check
        # whether there is only one of each class.
        world_by_key = {}
        unique_match = True
        for w in world_objects:
            matches = []
            for key, (klass, *_) in classes.items():
                if isinstance(w, klass):
                    matches.append(key)
            if len(matches) == 1:
                world_by_key[matches[0]] = w
            else:
                unique_match = False
                break

        # If the match is not unique, the order of the classes needs to match,
        # whereas if all classes are unique, we can still intelligently match
        # them even if the order is wrong.

        objects = {}

        if unique_match:

            for key, (klass, args, kwargs, *rest) in classes.items():

                if len(rest) == 0:
                    klass_gen = klass
                elif len(rest) == 1:
                    klass_gen = rest[0]
                else:
                    raise ValueError("Tuples in world_axis_object_classes should have length 3 or 4")

                # FIXME: For now SkyCoord won't auto-convert upon initialization
                # https://github.com/astropy/astropy/issues/7689
                from astropy.coordinates import SkyCoord
                if isinstance(world_by_key[key], SkyCoord):
                    if 'frame' in kwargs:
                        objects[key] = world_by_key[key].transform_to(kwargs['frame'])
                    else:
                        objects[key] = world_by_key[key]
                else:
                    objects[key] = klass_gen(world_by_key[key], *args, **kwargs)

        else:

            for ikey, key in enumerate(classes):

                klass, args, kwargs, *rest = classes[key]

                if len(rest) == 0:
                    klass_gen = klass
                elif len(rest) == 1:
                    klass_gen = rest[0]
                else:
                    raise ValueError("Tuples in world_axis_object_classes should have length 3 or 4")

                w = world_objects[ikey]
                if not isinstance(w, klass):
                    raise ValueError("Expected the following order of world "
                                     "arguments: {}".format(', '.join([k.__name__ for (k, _, _) in classes.values()])))

                # FIXME: For now SkyCoord won't auto-convert upon initialization
                # https://github.com/astropy/astropy/issues/7689
                from astropy.coordinates import SkyCoord
                if isinstance(w, SkyCoord):
                    if 'frame' in kwargs:
                        objects[key] = w.transform_to(kwargs['frame'])
                    else:
                        objects[key] = w
                else:
                    objects[key] = klass_gen(w, *args, **kwargs)

        # We now extract the attributes needed for the world values
        world = []
        for key, _, attr in components:
            if callable(attr):
                world.append(attr(objects[key]))
            else:
                world.append(rec_getattr(objects[key], attr))

        return world

    def values_to_high_level_objects(*world_values, low_level_wcs):
        """
        Convert low level values into high level objects.

        This function uses the information in ``wcs.world_axis_object_classes`` and
        ``wcs.world_axis_object_components`` to convert the high level objects
        (such as `~.SkyCoord`) to low level "values" `~.Quantity` objects.

        This is used in `.HighLevelWCSMixin.pixel_to_world`, but provided as a
        separate function for use in other places where needed.

        Parameters
        ----------
        *world_values: object
            Low level, "values" representations of the world coordinates.

        low_level_wcs: `.BaseLowLevelWCS`
            The WCS object to use to interpret the coordinates.
        """
        # Cache the classes and components since this may be expensive
        components = low_level_wcs.world_axis_object_components
        classes = low_level_wcs.world_axis_object_classes

        # Deserialize classes
        if low_level_wcs.serialized_classes:
            classes_new = {}
            for key, value in classes.items():
                classes_new[key] = deserialize_class(value, construct=False)
            classes = classes_new

        args = defaultdict(list)
        kwargs = defaultdict(dict)

        for i, (key, attr, _) in enumerate(components):
            if isinstance(attr, str):
                kwargs[key][attr] = world_values[i]
            else:
                while attr > len(args[key]) - 1:
                    args[key].append(None)
                args[key][attr] = world_values[i]

        result = []

        for key in default_order(components):
            klass, ar, kw, *rest = classes[key]
            if len(rest) == 0:
                klass_gen = klass
            elif len(rest) == 1:
                klass_gen = rest[0]
            else:
                raise ValueError("Tuples in world_axis_object_classes should have length 3 or 4")
            result.append(klass_gen(*args[key], *ar, **kwargs[key], **kw))

        return result
