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
        list of dictionary
    
    Returns
    -------
    list
        list of flattened list
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
        meta['short_name'] = value
    else:
        meta[key] = value


class NDCubeIOMixin(NDIOMixin):

    # Inherit docstring from parent class
    __doc__ = NDIOMixin.__doc__

    MISSING = 'MISNG{0}'
    EXTRA_COORDS = 'EXTCR{0}'

    # Here we create a read and write function to support I/O of NDCube files
    # Currently read option is not supported
    # Here we delegate the write method to save the files to different helper functions
    # For different formats, we can define different helper functions, and based on the file type
    # we can call them. Currently only FITS is supported.

    # Here we create a constructor for a mixin class, which is quite rare for a mixin
    # Since the data of the NDCube needs to be used in this mixin, the
    # constructor seems to be useful

    def __init__(self, data, wcs, uncertainty, mask, meta,
                 unit, copy, extra_coords=None, missing_axis=None, **kwargs):
        
        super().__init__(data, wcs, uncertainty, mask, meta,
                 unit, copy, **kwargs)

        self.__data = data
        self.__wcs = wcs
        self.__meta = meta
        self.__uncertainty = uncertainty
        self.__mask = mask
        self.__unit = unit
        self.__missing_axis = missing_axis
        self.__extra_coords = extra_coords



    def to_hdu(self, hdu_mask='MASK', hdu_uncertainty='UNCERT', key_uncertainty_type='UTYPE'):
        """Create a HDUList from a ImageHDU object
        
        Parameters
        ----------
        hdu_mask, hdu_uncertainty : str, optional
            If it is a string append this attribute to the HDUList as 
            `astropy.io.fits.ImageHDU` with the string as the extension name.
            Default is `MASK` for hdu_mask, `UNCERT` for uncertainty and `None`
            for flags.

        key_uncertainty_type : str, optional
            The header key name for the class name of the uncertainty (if any)
            that is used to store the uncertainty type in the uncertainty hdu.
        
        Raises
        ------
        ValueError
            - If `self.__mask` is set but not a `numpy.ndarray`.
            - If `self.__uncertainty` is set but not a astropy uncertainty type
            - If `self.__uncertainty` is set but has another unit than `self.__data`.
        
        Returns
        -------
        hdulist:
            `astropy.io.fits.HDUList`
        """


        # ----------------------------------HDU0------------------------------
        # Create a copy of the meta data to avoid changing of the header of data
        if self.__meta is not None:
            header = fits.Header(self.__meta.copy())

        else:
            header = fits.Header()
        # Create a FITS Safe header from the meta data

        # Cross-check if the metadata is fits safe, if not then it is corrected
        for k, v in header.items():
            _insert_in_metadata_fits_safe(header, k, v)

        if self.__wcs:
            # Create a header for a given wcs object
            # Hard-Coded relax parameter to write all 
            # recognized informal extensions of the WCS standard.
            wcs_header = self.__wcs.to_header(relax=True)
            header.extend(wcs_header, useblanks=False, update=True)

        # Create a FITS header for storing missing_axis
        # MISSING0 : 1 if axis 1 is missing
        # other keywords are skipped, if the axis is present
        if self.__missing_axis:
            header0 = fits.Header()
            for index, axis in enumerate(self.__missing_axis):
                if axis:
                    header0[MISSING.format(index)] = 1
                else:
                    continue

            for k, v in header0.items():
                _insert_in_metadata_fits_safe(header0, k, v)
            
            header.extend(header0, useblanks=False, update=True)


        # PrimaryHDU list contains only meta, missing_axis and wcs as headers, no data
        hdus = [fits.PrimaryHDU(data=None, header=header)]
        
        #------------------------------HDU0----------------------------------
        
        #------------------------------HDU1----------------------------------
        # Header for unit 
        if self.__unit:
            header_unit = fits.Header()
            if self.__unit is not u.dimensionless_unscaled:
                header_unit['bunit'] = self.__unit.to_string()
            hdus.append(fits.ImageHDU(self.__data, header_unit, name='UNIT'))

        #------------------------------HDU1----------------------------------

        #------------------------------HDU2----------------------------------
        # Store the uncertainty
        if self.__uncertainty is not None:
            # NOTE: Comments copied from `astropy.nddata.ccdata`, displayed here for reference
            # We need to save some kind of information which uncertainty was
            # used so that loading the HDUList can infer the uncertainty type.
            # No idea how this can be done so only allow StdDevUncertainty.
            uncertainty_cls =self.__uncertainty.__class__
            if uncertainty_cls not in _known_uncertainties:
                raise ValueError('only uncertainties of type {} can be saved.'
                                .format(_known_uncertainties))
            uncertainty_name = _unc_cls_to_name[uncertainty_cls]

            hdr_uncertainty = fits.Header()
            hdr_uncertainty[key_uncertainty_type] = uncertainty_name

            # Assuming uncertainty is an StdDevUncertainty save just the array
            # this might be problematic if the Uncertainty has a unit differing
            # from the data so abort for different units. This is important for
            # astropy > 1.2
            if (hasattr(self.__uncertainty, 'unit') and
                    self.__uncertainty.unit is not None):
                if not _uncertainty_unit_equivalent_to_parent(
                        uncertainty_cls, self.__uncertainty.unit, self.__unit):
                    raise ValueError(
                        'saving uncertainties with a unit that is not '
                        'equivalent to the unit from the data unit is not '
                        'supported.')
            hduUncert = fits.ImageHDU(self.__uncertainty, hdr_uncertainty, name='UNCERT')
            hdus.append(hduUncert)
        #----------------------------HDU2--------------------------------
        
        #----------------------------HDU3------------------------------
        # Store the mask
        if self.__mask is not None:
            # Always assuming that the mask is a np.ndarray
            # by checking that it has a shape
            if not hasattr(self.__mask, 'shape'):
                raise ValueError('only a numpy.ndarray mask can be saved.')
            hduMask = fits.ImageHDU(self.__mask.astype(np.uint8), name=hdu_mask)
            hdus.append(hduMask)

        #----------------------------HDU3-------------------------------
        # Store the extra_coords
        if self.__extra_coords is not None:

            # Set up the data
            flattened_list_of_dict = flatten(self.__extra_coords)
            
            # We convert the list of dictionary to pandas dataframe
            # and then to numpy.ndarray
            dframe = pd.DataFrame(flattened_list_of_dict)
            extra_coords = dframe.values

            # Set up the header
            header3 = fits.Header()
            for index, _, value in enumerate(extra_coords):
                header3[EXTRA_COORDS.format(index)] = value
            
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

        