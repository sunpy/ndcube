
from typing import TYPE_CHECKING, Any

from astropy.nddata.mixins.ndslicing import NDSlicingMixin
from astropy.wcs.wcsapi.wrappers.sliced_wcs import sanitize_slices

if TYPE_CHECKING:
    from ndcube.extra_coords.extra_coords import ExtraCoordsABC
    from ndcube.global_coords import GlobalCoordsABC

__all__ = ['NDCubeSlicingMixin']


class NDCubeSlicingMixin(NDSlicingMixin):  # type: ignore[misc]
    # Inherit docstring from parent class
    __doc__ = NDSlicingMixin.__doc__

    # Attributes expected to be provided by the class this is mixed into
    # (e.g. NDCubeBase, which combines this with astropy.nddata.NDData and NDCubeABC).
    if TYPE_CHECKING:
        meta: Any
        shape: tuple[int, ...]
        global_coords: GlobalCoordsABC
        extra_coords: ExtraCoordsABC

    def __getitem__(self, item: Any) -> Any:
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
        meta: Any = None
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
        sliced_cube._global_coords._internal_coords = self.global_coords._internal_coords  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
        sliced_cube._extra_coords = self.extra_coords[item]

        # If metadata sliceable, slice and add back onto sliced cube.
        if meta_is_sliceable:
            sliced_cube.meta = meta.slice[item]

        return sliced_cube
