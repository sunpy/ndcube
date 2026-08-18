
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
        sliced_cube = super().__getitem__(item)
        if meta_is_sliceable:
            self.meta = meta  # Add unsliced meta back onto unsliced cube.

        # Add sliced coords back onto sliced cube.
        sliced_cube._global_coords._internal_coords = self.global_coords._internal_coords
        sliced_cube._extra_coords = self.extra_coords[item]

        # If metadata sliceable, slice and add back onto sliced cube.
        if meta_is_sliceable:
            sliced_cube.meta = meta.slice[item]

        self._slice_custom_state(sliced_cube, item)

        return sliced_cube

    def _slice_custom_state(self, sliced_cube, item):
        """
        Update custom subclass state on a newly sliced cube.

        Called at the end of ``__getitem__``, after the data, WCS, coords and
        metadata of ``sliced_cube`` have been set.

        Subclasses carrying extra state that tracks the data axes (for example
        a list of per-frame WCS objects) should override this method instead of
        ``__getitem__``, mutating ``sliced_cube`` in place.  The default
        implementation does nothing.

        Parameters
        ----------
        sliced_cube : `~ndcube.NDCube`
            The new cube produced by slicing this one.

        item : `tuple`
            The sanitized slice item: one entry per data axis of the original
            cube, containing only `int` and `slice` objects.  Any ellipsis has
            already been expanded and missing trailing axes filled with
            ``slice(None)``.
        """
