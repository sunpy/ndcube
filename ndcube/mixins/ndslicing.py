
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

        return super().__getitem__(item)

    def _slice(self, item):
        """Construct a set of keyword arguments to initialise a new (sliced)
        instance of the class. This method is called in
        `astropy.nddata.mixins.NDSlicingMixin.__getitem__`.

        This method extends the `~astropy.nddata.mixins.NDSlicingMixin` method
        to add support for  ``extra_coords`` and overwrites the astropy
        handling of wcs slicing.

        Parameters
        ----------
        item : slice
            The slice passed to ``__getitem__``. Note that the item parameter corresponds
            to numpy ordering, keeping with the convention for NDCube.

        Returns
        -------
        dict :
            Containing all the attributes after slicing - ready to
            use them to create ``self.__class__.__init__(**kwargs)`` in
            ``__getitem__``.
        """

        item = tuple(sanitize_slices(item, len(self.dimensions)))
        kwargs = super()._slice(item)

        # Store the original dimension of NDCube object before slicing
        len(self.dimensions)

        kwargs['extra_coords'] = self.extra_coords[item]

        return kwargs
