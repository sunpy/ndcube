import copy

from astropy.nddata.mixins.ndslicing import NDSlicingMixin
from astropy.wcs.wcsapi import SlicedLowLevelWCS
from astropy.wcs.wcsapi.sliced_low_level_wcs import sanitize_slices

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

        This method calls ``_slice`` and then constructs a new object using the
        kwargs returned by ``_slice``.
        """
        if item is None or (isinstance(item, tuple) and None in item):
            raise IndexError("None indices not supported")

        return super().__getitem__(item)

    def _slice(self, item):
        """
        Construct a set of keyword arguments to initialise a new (sliced)
        instance of the class. This method is called in
        `astropy.nddata.mixins.NDSlicingMixin.__getitem__`.

        This method extends the `~astropy.nddata.mixins.NDSlicingMixin` method
        to add support for ``missing_axes`` and ``extra_coords`` and overwrites
        the astropy handling of wcs slicing.
        """

        item = tuple(sanitize_slices(item, len(self.dimensions)))
        kwargs = super()._slice(item)

        # Store the original dimension of the wcs object before slicing
        prev_dim = self.high_level_wcs.low_level_wcs.world_n_dim

        # Sanitize the input arguments
        wcs = self._slice_wcs(item)
        print('ITEM {0}'.format(item))

        # Set the kwargs values
        kwargs['wcs'] = wcs
        kwargs['missing_axes'] = None
        kwargs['extra_coords'] = self._slice_extra_coords(item, wcs._pixel_keep, prev_dim)
        return kwargs

    def _slice_wcs(self, item):

        # Returns the sliced WCS object
        return SlicedLowLevelWCS(self.wcs, item)


    def _slice_extra_coords(self, item, pixel_keep, naxes):

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
            new_extra_coords, pixel_keep, naxes)
        return new_extra_coords_dict
