# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from astropy.nddata import NDIOMixin, CCDData
from astropy.io import fits, registry
from astropy import units as u
from astropy.nddata.nduncertainty import (
    StdDevUncertainty, NDUncertainty, VarianceUncertainty, InverseVariance)

_known_uncertainties = (StdDevUncertainty, VarianceUncertainty, InverseVariance)
_unc_name_to_cls = {cls.__name__: cls for cls in _known_uncertainties}
_unc_cls_to_name = {cls: cls.__name__ for cls in _known_uncertainties}


__all__ = ['NDCubeIOMixin']

# This file is written to define a mixin class to support
# write option in ndcube. Uses `astropy.io.registry`
# to register the write function


def flatten(lst):
    """Helper function to flatten a list of dictionary,
       resulted out of the form :
       {Key1:Value1, {Key2:Value2, Key3:Value3}}


    Parameters
    ----------
    lst : list
        dictionary of dictionary

    Returns
    -------
    list
        list of flattened dictionary
    Notes
    -----
    Format of the output:
    [{Key1:Value1, Key2:Value2, Key3:Value3}]
    """

    res1 = list()
    for entry in lst:
        for key, value in entry.items():
            res = dict()
            res['name'] = key
            for k,v in value.items():
                res[k] = v
        res1.append(res)
    return res1



def _insert_in_metadata_fits_safe(meta, key, value):
    """Helper function to insert key-value pair into metadata in a way that
       FITS can serialize

    Parameters
    ----------
    key : str
        Key to be inserted in the dictionary
    value : str or None
        Value to be inserted

    Notes
    -----
    This addresses a shortcoming of the FITS standard. There are length
        restrictions on both the ``key`` (8 characters) and ``value`` (72
        characters) in the FITS standard. There is a convention for handling
        long keywords and a convention for handling long values, but the
        two conventions cannot be used at the same time.

        This addresses that case by checking the length of the ``key`` and
        ``value`` and, if necessary, shortening the key.
    Reference
    ---------
    This helper method is taken from `astropy.nddata.cddata`. Not imported
    as this is a private method, subject to change

    """

    if len(key) > 8 and len(value) > 72:
        short_name = key[:8]
        meta['HIERARCH {0}'.format(key.upper())] = (short_name, "Shortened name for {}".format(key))
        meta[short_name] = value
    else:
        meta[key] = value


class NDCubeIOMixin(NDIOMixin):

    # Inherit docstring from parent class
    __doc__ = NDIOMixin.__doc__

    MISSING = 'MISNG{0}'
    EXTRA_COORDS_LABEL = 'EXTCR{0}'

    # Here we create a read and write function to support I/O of NDCube files
    # Currently read option is not supported
    # Here we delegate the write method to save the files to different helper functions
    # For different formats, we can define different helper functions, and based on the file type
    # we can call them. Currently only FITS is supported.


    def to_hdulist(self, hdu_mask='MASK', hdu_uncertainty='UNCERT'):
        """Create a HDUList from a ImageHDU object

        Parameters
        ----------
        hdu_mask, hdu_uncertainty : str, optional
            If it is a string append this attribute to the HDUList as
            `astropy.io.fits.ImageHDU` with the string as the extension name.
            Default is `MASK` for hdu_mask, `UNCERT` for uncertainty and `None`
            for flags.


        Raises
        ------
        ValueError
            - If `self._mask` is set but not a `numpy.ndarray`.
            - If `self._uncertainty` is set but not a astropy uncertainty type
            - If `self._uncertainty` is set but has another unit than `self.__data`.

        Returns
        -------
        hdulist:
            `astropy.io.fits.HDUList`
        """


        # ----------------------------------HDU0------------------------------
        # Create a copy of the meta data to avoid changing of the header of data
        if self._meta is not None:
            header = fits.Header(self._meta.copy())

        else:
            header = fits.Header()
        # Create a FITS Safe header from the meta data

        # Cross-check if the metadata is fits safe, if not then it is corrected
        for k, v in header.items():
            _insert_in_metadata_fits_safe(header, k, v)

        if self._wcs:
            # Create a header for a given wcs object
            # Hard-Coded relax parameter to write all
            # recognized informal extensions of the WCS standard.
            wcs_header = self._wcs.to_header(relax=True)
            header.extend(wcs_header, useblanks=False, update=True)

        # Create a FITS header for storing missing_axis
        # MISSING0 : 1 if axis 1 is missing
        # other keywords are skipped, if the axis is present
        # The missing axis are expected to be present, as it is a bool
        header0 = fits.Header()
        for index, axis in enumerate(self.missing_axis):
            if axis:
                header0[MISSING.format(index)] = 1
            else:
                header0[MISSING.format(index)] = 0

        for k, v in header0.items():
            _insert_in_metadata_fits_safe(header0, k, v)

        header.extend(header0, useblanks=False, update=True)

        # PrimaryHDU list contains only meta, missing_axis and wcs as headers, no data
        hdus = [fits.PrimaryHDU(data=None, header=header)]

        #------------------------------HDU0----------------------------------

        #------------------------------HDU1----------------------------------
        # Header for unit
        if self._unit:
            header_unit = fits.Header()
            if self._unit is not u.dimensionless_unscaled:
                header_unit['bunit'] = self._unit.to_string()
            hdus.append(fits.ImageHDU(self._data, header_unit, name='DATA'))

        #------------------------------HDU1----------------------------------

        #------------------------------HDU2----------------------------------
        # Store the uncertainty
        if self._uncertainty is not None:
            
            # Set the initial header, and the type of uncertainty
            hdr_uncertainty = fits.Header()
            hdr_uncertainty['UTYPE'] = self._uncertainty.uncertainty_type

            # Set the data of the uncertainty and header
            hduUncert = fits.ImageHDU(self._uncertainty.array, hdr_uncertainty, name=hdu_uncertainty)
            hdus.append(hduUncert)
        #----------------------------HDU2--------------------------------

        #----------------------------HDU3------------------------------
        # Store the mask
        if self._mask is not None:
            # Always assuming that the mask is a np.ndarray
            # by checking that it has a shape
            if not isinstance(self.__mask, np.ndarray):
                raise ValueError('Only a numpy.ndarray mask can be saved.')
            hduMask = fits.ImageHDU(self._mask.astype(np.uint8), name=hdu_mask)
            hdus.append(hduMask)

        #----------------------------HDU3-------------------------------
        # Store the extra_coords
        if self._extra_coords_wcs_axis is not None:

            # Set up the data
            flattened_list_of_dict = flatten(self._extra_coords_wcs_axis)

            # We convert the list of dictionary to pandas dataframe
            # and then to numpy.ndarray
            dframe = pd.DataFrame(flattened_list_of_dict)
            extra_coords = dframe.values

            # Set up the header
            header3 = fits.Header()
            for index, _, value in enumerate(extra_coords):
                header3[EXTRA_COORDS_LABEL.format(index)] = value

            # Make sure all the keywords are FITS safe
            for k, v in header3.items():
                _insert_in_metadata_fits_safe(header3, k, v)

            # Setting up the Header
            hdu_extra_coords = fits.Header()
            hdu_extra_coords.extend(header3, useblanks=False, update=True)

            hdus.append(fits.TableHDU(data=extra_coords, header=hdu_extra_coords, name='EXTRA_COORDS'))
        #--------------------------HDU3---------------------------------

        hdulist = fits.HDUList(hdus)
        return hdulist
