
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

        if isinstance(item, tuple) and Ellipsis in item:
            if item.count(Ellipsis) > 1:
                raise IndexError("An index can only have a single ellipsis ('...')")
            expanded_item = []
            for i in item:
                if i is Ellipsis:
                    expanded_item.extend([slice(None)] * (len(self.shape) - len(item) + 1))
                else:
                    expanded_item.append(i)
            item = tuple(expanded_item)

        # Slice cube.
        item = tuple(sanitize_slices(item, len(self.shape)))
        # Resolve negative/out-of-range bounds, which the WCS slicer does not (astropy#15557).
        # Drop the step: sanitize_slices only allows 1, which nested WCS slicing rejects.
        item = tuple(slice(*i.indices(n)[:2]) if isinstance(i, slice) else range(n)[i]
                     for i, n in zip(item, self.shape))
        if 0 not in self.shape and any(isinstance(i, slice) and i.start >= i.stop for i in item):
            raise IndexError("Slicing would give a length-0 axis, which a WCS cannot describe.")
        # If cube has a sliceable metadata, remove it and handle it separately.
        # This is to prevent the shapes of the data and metadata getting out of
        # sync part way through the slicing process.
        meta_is_sliceable = False
        if hasattr(self.meta, "__ndcube_can_slice__") and self.meta.__ndcube_can_slice__:
            meta_is_sliceable = True
            meta = self.meta
            self.meta = None

        try:
            sliced_cube = super().__getitem__(item)
        finally:
            if meta_is_sliceable:
                self.meta = meta  # Add unsliced meta back onto unsliced cube.

        # Add sliced coords back onto sliced cube.
        sliced_cube._global_coords._internal_coords = self.global_coords._internal_coords
        sliced_cube._extra_coords = self.extra_coords[item]

        # If metadata sliceable, slice and add back onto sliced cube.
        if meta_is_sliceable:
            sliced_cube.meta = meta.slice[item]

        return sliced_cube
