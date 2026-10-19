import numbers

import numpy as np

from astropy.nddata.mixins.ndslicing import NDSlicingMixin
from astropy.utils.shapes import simplify_basic_index

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

        # Make all items iterable
        items = [item if isinstance(item, (tuple, list)) else (item,)]
        # Ensure all items are valid input
        if not all(isinstance(i, (numbers.Integral, slice)) or i is Ellipsis for i in items):
            raise IndexError("This slice is invalid, only integer or range slices are supported.")

        ################################################################################
        # TODO: Remove when astropy>7.2.3
        ################################################################################
        # Resolve negative and out-of-range bounds against the cube's shape, as numpy does for the data.
        # Before astropy 7.2.3/8.0.2 the WCS slicer takes negative values as offsets from 0 (astropy/astropy#15557),
        # and since then it raises for them if the WCS has no array_shape.
        try:
            item = simplify_basic_index(item, shape=self.shape)
        except np.exceptions.AxisError as error:
            raise IndexError(f"index {error.axis} is out of bounds for an axis of size {error.ndim}") from None

        # Drop the step: nested WCS slicing rejects an explicit step before astropy 7.2.3/8.0.2.
        item = tuple(slice(i.start, i.stop) if isinstance(i, slice) else i for i in item)
        if 0 not in self.shape and any(isinstance(i, slice) and i.start >= i.stop for i in item):
            raise IndexError("Slicing would give a length-0 axis, which a WCS cannot describe.")
        ################################################################################

        # Error early if a step is passed
        if any(isinstance(i, slice) and i.step not in (None, 1) for i in item):
            raise IndexError("Slicing WCS with a step is not supported.")

        # If cube has a sliceable metadata, remove it and handle it separately.
        # This is to prevent the shapes of the data and metadata getting out of
        # sync part way through the slicing process.
        if meta_is_sliceable := (hasattr(self.meta, "__ndcube_can_slice__") and self.meta.__ndcube_can_slice__):
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
