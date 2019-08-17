# -*- coding: utf-8 -*-

"""
Helpers for testing ndcube.
"""
import unittest

import numpy as np
import astropy.modeling.projections as projections
import astropy.modeling.rotations as rotations
import gwcs as GWCS

from astropy import units as u
from sunpy.coordinates.frames import Helioprojective
import gwcs.coordinate_frames as cf
from numpy.testing import assert_equal
from astropy.modeling import models
from astropy.modeling.models import Identity, Multiply, Shift

from ndcube import utils
from astropy.wcs.wcsapi.fitswcs import SlicedFITSWCS, SlicedLowLevelWCS
from astropy.wcs.wcsapi.sliced_low_level_wcs import sanitize_slices

__all__ = ['assert_extra_coords_equal',
           'assert_metas_equal',
           'assert_cubes_equal',
           'assert_cubesequences_equal',
           'assert_wcs_are_equal']


def assert_extra_coords_equal(test_input, extra_coords):
    assert test_input.keys() == extra_coords.keys()
    for key in list(test_input.keys()):
        assert test_input[key]['axis'] == extra_coords[key]['axis']
        assert (test_input[key]['value'] == extra_coords[key]['value']).all()


def assert_metas_equal(test_input, expected_output):
    assert test_input.keys() == expected_output.keys()
    for key in list(test_input.keys()):
        assert test_input[key] == expected_output[key]


def assert_cubes_equal(test_input, expected_cube):
    unit_tester = unittest.TestCase()
    assert type(test_input) == type(expected_cube)
    assert np.all(test_input.mask == expected_cube.mask)
    assert_wcs_are_equal(test_input.wcs, expected_cube.wcs)
    assert test_input.uncertainty.array.shape == expected_cube.uncertainty.array.shape
    # assert test_input.world_axis_physical_types == expected_cube.world_axis_physical_types
    assert all(test_input.dimensions.value == expected_cube.dimensions.value)
    assert test_input.dimensions.unit == expected_cube.dimensions.unit
    assert_extra_coords_equal(test_input.extra_coords, expected_cube.extra_coords)


def assert_cubesequences_equal(test_input, expected_sequence):
    assert type(test_input) == type(expected_sequence)
    assert_metas_equal(test_input.meta, expected_sequence.meta)
    assert test_input._common_axis == expected_sequence._common_axis
    for i, cube in enumerate(test_input.data):
        assert_cubes_equal(cube, expected_sequence.data[i])


def assert_wcs_are_equal(wcs1, wcs2):
    """
    Assert function for testing two wcs object.
    Used in testing NDCube.
    Also checks if both the wcs objects are instance
    of `SlicedLowLevelWCS`
    """

    # Check the APE14 attributes of both the WCS
    assert wcs1.pixel_n_dim == wcs2.pixel_n_dim
    assert wcs1.world_n_dim == wcs2.world_n_dim
    # assert wcs1.array_shape == wcs2.array_shape
    assert wcs1.pixel_shape == wcs2.pixel_shape
    # assert wcs1.world_axis_physical_types == wcs2.world_axis_physical_types
    assert wcs1.world_axis_units == wcs2.world_axis_units
    assert_equal(wcs1.axis_correlation_matrix, wcs2.axis_correlation_matrix)
    assert wcs1.world_axis_object_components == wcs2.world_axis_object_components
    assert wcs1.pixel_bounds == wcs2.pixel_bounds
    assert repr(wcs1) == repr(wcs2)

def create_sliced_wcs(wcs, item, dim):
    """
    Creates a sliced `SlicedFITSWCS` object from the given slice item
    """

    # Sanitize the slices
    item = sanitize_slices(item, dim)
    return SlicedFITSWCS(wcs, item)

def convert_fits_to_gwcs(fitswcs):
	"""Helper function to return a corresponding gWCS object from a fits-wcs
	
	Parameters
	----------
	fitswcs : `astropy.wcs.WCS`
		The Astropy fits-wcs object
	
	Note
	----
	This function assumes that the following order of elements
	are added in the FITS-WCS object - (WCS Ordering)

	1. WAVE/TIME
	2. HPLT-TAN
	3. HPLN-TAN
	or

	1. WAVE
	2. TIME
	3. HPLT-TAN
	4. HPLN-TAN
	or

	1. HPLT-TAN
	2. HPLN-TAN

	Any other order might not result a correct gWCS object / raise Error
	"""



	# Get the number of axis and ctypes/crval/crpix of fitswcs
	naxis = fitswcs.pixel_n_dim
	fctypes = fitswcs.wcs.ctype
	fcunit = fitswcs.wcs.cunit
	
	# breakpoint()
	# If naxis is 1, raise an Error
	if(naxis == 1):
		raise ValueError("The dimension of the FITS-WCS should be greater than 1!")

	# Define the Model for celestial coordinates

	#  Helioprojective frame
	sky_frame = cf.CelestialFrame(axes_order=(naxis-2, naxis-1), name='helioprojective',
								reference_frame=Helioprojective(obstime="2018-01-01"))

	# Case 1 : Only celestial axes are present
	if(naxis == 2):

		# Detector frame 
		detector_frame = cf.CoordinateFrame(name="detector", naxes=2,
									axes_order=(0, 1),
									axes_type=("pixel", "pixel"),
									axes_names=("x", "y"),
									unit=(u.pix, u.pix))
		# Get the transformation
		trans = get_celestial_transformation(fitswcs)

		return GWCS.wcs.WCS(forward_transform=trans, output_frame=sky_frame, input_frame=detector_frame)

	# Case 2: Celestial and one more axes is present
	elif(naxis == 3):

		# Check whether 3rd dimension is WAVE/TIME
		if(fctypes[-3] == 'WAVE'):

			# Get the transformation of wave and sky
			trans_wave = get_external_transformation(fitswcs)
			trans_sky = get_celestial_transformation(fitswcs)

			# Stitch the model together
			trans = trans_sky & trans_wave

			# Define the frame for WAVE
			wave_frame = cf.SpectralFrame(axes_order=(0, ), unit=u.Unit(fcunit[-3]), axes_names=("wavelength",))
			
			# Stitch the WAVE and CELESTIAL frame
			frame = cf.CompositeFrame([sky_frame, wave_frame])

			detector_frame = cf.CoordinateFrame(name="detector", naxes=3,
                                        axes_order=(0, 1, 2),
                                        axes_type=("pixel", "pixel", "pixel"),
                                        axes_names=("x", "y", "z"), unit=(u.pix, u.pix, u.pix))
			
			return GWCS.wcs.WCS(forward_transform=trans, output_frame=frame, input_frame=detector_frame)

		if(fctypes[-3] == 'TIME'):

			# Get the transformation of time and sky
			trans_time = get_external_transformation(fitswcs)
			trans_sky = get_celestial_transformation(fitswcs)

			# Stitch the model together
			trans = trans_sky & trans_time

			# Define the frame for TIME
			time_frame = cf.TemporalFrame(axes_order=(0, ), unit=u.Unit(fcunit[-3]))

			# Stitch the TIME and CELESTIAL
			frame = cf.CompositeFrame([sky_frame, time_frame])

			detector_frame = cf.CoordinateFrame(name="detector", naxes=3,
                                        axes_order=(0, 1, 2),
                                        axes_type=("pixel", "pixel", "pixel"),
                                        axes_names=("x", "y", "s"), unit=(u.pix, u.pix, u.pix))

			return GWCS.wcs.WCS(forward_transform=trans, output_frame=frame, input_frame=detector_frame)
	
	# Case 3: Celestial and two more axis are present
	elif(naxis == 4):

		# Get the transformation of time/wave/sky
		trans_sky = get_celestial_transformation(fitswcs)
		trans_time, trans_wave = get_external_transformation(fitswcs)
		
		# Stitch the model together
		trans = trans_sky & trans_wave & trans_time

		# Define the frame for TIME/WAVE
		wave_frame = cf.SpectralFrame(axes_order=(1, ), unit=u.Unit(fcunit[-4]))
		time_frame = cf.TemporalFrame(axes_order=(0, ), unit=u.Unit(fcunit[-3]))

		# Stitch the TIME/WAVE/CELESTIAL
		frame = cf.CompositeFrame([sky_frame, wave_frame, time_frame])
		
		detector_frame = cf.CoordinateFrame(name="detector", naxes=4,
                                        axes_order=(0, 1, 2, 3),
                                        axes_type=("pixel", "pixel", "pixel", "pixel"),
                                        axes_names=("x", "y", "z", "s"),
										unit=(u.pix, u.pix, u.pix, u.pix))

		return GWCS.wcs.WCS(forward_transform=trans, output_frame=frame, input_frame=detector_frame)
	
	# Case 4: When the naxis > 3
	else:
		raise ValueError("Currently more than 4 dimensions of FITSWCS conversion to gWCS not supported!")

    
def get_celestial_transformation(fitswcs):
	"""Returns the celestial transformation
	
	Parameters
	----------
	fitwcs : `astropy.wcs.WCS`
		Fits wcs object
	"""
	projection_dict = {
        'TAN': projections.Pix2Sky_TAN(),
        'SIN': projections.Pix2Sky_SIN()
    }

	# naxis = fitswcs.pixel_n_dim
	fctypes = fitswcs.wcs.ctype
	fcrval = fitswcs.wcs.crval
	fcrpix = fitswcs.wcs.crpix
	fcunit = fitswcs.wcs.cunit

	tptype = fctypes[-1][-3:]

	# Shift the x/y coordinates by CRPIX
	shift_by_crpix = models.Shift(fcrpix[-1]*u.pix) & models.Shift(fcrpix[-2]*u.pix)

	# Add the PC matrix if present
	if fitswcs.wcs.has_pc():
		pcmatrix = np.array(fitswcs.wcs.cdelt)[-2:] * fitswcs.wcs.pc[-2:,-2:]

	# Rotation / Projection / Rotation using CRVAL
	rotation = projections.AffineTransformation2D(pcmatrix)
	projection_pipe = projection_dict[tptype]
	celestial_rotation = rotations.RotateNative2Celestial(fcrval[-1], fcrval[-2], 180.)
	
	# The Final Transformation
	trans = shift_by_crpix | rotation | projection_pipe | celestial_rotation
	
	return trans

def get_external_transformation(fitswcs):
	"""Returns the transformation of Time/Wave dimension
	Parameters
	----------
	fitwcs : `astropy.wcs.WCS`
		Fits wcs object
	"""

	naxis = fitswcs.pixel_n_dim
	fctypes = fitswcs.wcs.ctype
	fcrval = fitswcs.wcs.crval
	fcrpix = fitswcs.wcs.crpix * u.pix
	fcunit = fitswcs.wcs.cunit
	fcdelt = fitswcs.wcs.cdelt

	# Currently only wave and time as an external dimension
	# is supported

	# Case 1: Only one extra dimension is present
	# Either wave / time
	if(naxis == 3):
		
		shift = Shift(fcrpix[-3])
		scale = Multiply(fcdelt[-3]* (u.Unit(fcunit[-3]) / u.pix))

		return (shift | scale | Identity(1))
	
	# Case  2: There are 2 extra dimension is present
	elif(naxis == 4):

		result_tuple = list()

		shift = Shift(fcrpix[-3])
		scale = Multiply(fcdelt[-3]* (u.Unit(fcunit[-3]) / u.pix))

		result_tuple.append(shift | scale | Identity(1))

		shift = Shift(fcrpix[-4])
		scale = Multiply(fcdelt[-4]* (u.Unit(fcunit[-4]) / u.pix))

		result_tuple.append(shift | scale | Identity(1))
		
		return tuple(result_tuple)
	else:
		raise ValueError(f'Transformation of WCS object with {naxis} dimensions not supported!')


def create_ndcube(wcs_ndcube, gwcs_ndcube, slice_item):
	"""This function returns a tuple of NDCube created
	from a wcs and gwcs based NDCube object. This function
	also slices	the NDCube object.

	Parameters
	----------
	slice_item : `slice`
		The slice of the NDCube
	"""

	wcs_slice = wcs_ndcube[slice_item] if wcs_ndcube else None
	gwcs_slice = gwcs_ndcube[slice_item] if gwcs_ndcube else None

	return (wcs_slice, gwcs_slice)
