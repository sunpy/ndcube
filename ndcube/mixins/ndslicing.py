
from astropy.nddata.mixins.ndslicing import NDSlicingMixin
from astropy.wcs.wcsapi.wrappers.sliced_wcs import sanitize_slices

__all__ = ['NDCubeSlicingMixin']


class NDCubeSlicingMixin(NDSlicingMixin):
    # Inherit docstring from parent class
    __doc__ = NDSlicingMixin.__doc__

    def __getitem__(self, item):
        """
        Override the parent class method to explicitly catch `None` indices.

        This method calls ``_slice`` and then constructs a new object
        using the kwargs returned by ``_slice``.
        """
        if item is None or (isinstance(item, tuple) and None in item):
            raise IndexError("None indices not supported")

        item = tuple(sanitize_slices(item, len(self.dimensions)))
        sliced_cube = super().__getitem__(item)

        sliced_cube._global_coords._internal_coords = self.global_coords._internal_coords
        sliced_cube._extra_coords = self.extra_coords[item]

        return sliced_cube
