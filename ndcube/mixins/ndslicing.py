import copy

from astropy.nddata.mixins.ndslicing import NDSlicingMixin

from ndcube import utils

__all__ = ['NDCubeSlicingMixin']


class NDCubeSlicingMixin(NDSlicingMixin):
    # Inherit docstring from parent class
    __doc__ = NDSlicingMixin.__doc__

    def _slice_wcs(self, item):
        """
        Override parent class method so we disable the wcs slicing on
        `astropy.nddata.mixins.NDSlicingMixin`.
        """
        return None

    def __getitem__(self, item):
        """
        Override the parent class method to explicitly catch `None` indices.

        This method calls ``_slice`` and then constructs a new object
        using the kwargs returned by ``_slice``.
        """
        # Abort slicing if the data is a single scalar.
        if self.data.shape == ():
            raise TypeError('scalars cannot be sliced.')

        # Let the other methods handle slicing.
        kwargs, dropped_coords = self._slice(item)
        new_cube = self.__class__(**kwargs)

        # Add any WCS coords whose axes are missing after slicing to extra_coords.
        if new_cube._extra_coords_wcs_axis is not None:
            new_cube._extra_coords_wcs_axis.update(dropped_coords)
        elif dropped_coords != {}:
            new_cube._extra_coords_wcs_axis = dropped_coords

        return new_cube

    def _slice(self, item):
        """
        Construct a set of keyword arguments to initialise a new (sliced)
        instance of the class. This method is called in
        `astropy.nddata.mixins.NDSlicingMixin.__getitem__`.
        This method extends the `~astropy.nddata.mixins.NDSlicingMixin`
        method to add support for ``missing_axes`` and ``extra_coords``
        and overwrites the astropy handling of wcs slicing.
        """
        kwargs = super()._slice(item)

        wcs, missing_axes, dropped_coords = self._slice_wcs_missing_axes(item)
        kwargs['wcs'] = wcs
        kwargs['missing_axes'] = missing_axes
        kwargs['extra_coords'] = self._slice_extra_coords(item, missing_axes)

        return kwargs, dropped_coords

    def _slice_wcs_missing_axes(self, item):
        # here missing axis is reversed as the item comes already in the reverse order
        # of the input
        return utils.wcs._wcs_slicer(self.wcs, copy.deepcopy(self.missing_axes), item)

    def _slice_extra_coords(self, item, missing_axes):
        if self.extra_coords is None:
            new_extra_coords_dict = None
        else:
            old_extra_coords = self.extra_coords
            extra_coords_keys = list(old_extra_coords.keys())
            new_extra_coords = copy.deepcopy(self._extra_coords_wcs_axis)
            for ck in extra_coords_keys:
                axis_ck = old_extra_coords[ck]["axis"]
                if isinstance(item, (slice, int)):
                    if axis_ck == 0:
                        new_extra_coords[ck]["value"] = new_extra_coords[ck]["value"][item]
                if isinstance(item, tuple):
                    try:
                        slice_item_extra_coords = item[axis_ck]
                        new_extra_coords[ck]["value"] = \
                            new_extra_coords[ck]["value"][slice_item_extra_coords]
                    except IndexError:
                        pass
                    except TypeError:
                        pass
                new_extra_coords_dict = utils.cube.convert_extra_coords_dict_to_input_format(
                    new_extra_coords, missing_axes)
        return new_extra_coords_dict
