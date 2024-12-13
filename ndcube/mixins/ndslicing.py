
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

        # If cube has a sliceable metadata, remove it and handle it separately.
        # This is to prevent the shapes of the data and metadata getting out of
        # sync part way through the slicing process.
        meta_is_sliceable = False
        if hasattr(self.meta, "__ndcube_can_slice__") and self.meta.__ndcube_can_slice__:
            meta_is_sliceable = True
            meta = self.meta
            self.meta = None

        # Slice cube.
        item = tuple(sanitize_slices(item, len(self.shape)))
        sliced_cube = super().__getitem__(item)
        if meta_is_sliceable:
            self.meta = meta  # Add unsliced meta back onto unsliced cube.

        # Add sliced coords back onto sliced cube.
        sliced_cube._global_coords._internal_coords = self.global_coords._internal_coords
        sliced_cube._extra_coords = self.extra_coords[item]

        # If metadata sliceable, slice and add back onto sliced cube.
        if meta_is_sliceable:
            sliced_cube.meta = meta.slice[item]

        return sliced_cube
