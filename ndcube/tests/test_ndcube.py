# -*- coding: utf-8 -*-
'''
Tests for NDCube
'''
from collections import namedtuple
import datetime

import pytest
import sunpy.map
import numpy as np
import astropy.units as u

from ndcube import NDCube, NDCubeOrdered
from ndcube.utils.wcs import WCS, _wcs_slicer
from ndcube.tests import helpers

# sample data for tests
# TODO: use a fixture reading from a test file. file TBD.
ht = {'CTYPE3': 'HPLT-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.5, 'CRPIX3': 0, 'CRVAL3': 0, 'NAXIS3': 2,
      'CTYPE2': 'WAVE    ', 'CUNIT2': 'Angstrom', 'CDELT2': 0.2, 'CRPIX2': 0, 'CRVAL2': 0,
      'NAXIS2': 3,
      'CTYPE1': 'TIME    ', 'CUNIT1': 'min', 'CDELT1': 0.4, 'CRPIX1': 0, 'CRVAL1': 0, 'NAXIS1': 4}
wt = WCS(header=ht, naxis=3)

data = np.array([[[1, 2, 3, 4], [2, 4, 5, 3], [0, -1, 2, 3]],
                 [[2, 4, 5, 1], [10, 5, 2, 2], [10, 3, 3, 0]]])

hm = {'CTYPE1': 'WAVE    ', 'CUNIT1': 'Angstrom', 'CDELT1': 0.2, 'CRPIX1': 0, 'CRVAL1': 10,
      'NAXIS1': 4,
      'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5,
      'NAXIS2': 3,
      'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 2}
wm = WCS(header=hm, naxis=3)

h_disordered = {
    'CTYPE1': 'TIME    ', 'CUNIT1': 'min', 'CDELT1': 0.4, 'CRPIX1': 0, 'CRVAL1': 0, 'NAXIS1': 2,
    'CTYPE2': 'WAVE    ', 'CUNIT2': 'Angstrom', 'CDELT2': 0.2, 'CRPIX2': 0, 'CRVAL2': 10,
    'NAXIS2': 4,
    'CTYPE3': 'HPLT-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.5, 'CRPIX3': 2, 'CRVAL3': 0.5,
    'NAXIS3': 3,
    'CTYPE4': 'HPLN-TAN', 'CUNIT4': 'deg', 'CDELT4': 0.4, 'CRPIX4': 2, 'CRVAL4': 1, 'NAXIS4': 2}
w_disordered = WCS(header=h_disordered, naxis=4)

data_disordered = np.zeros((2, 3, 4, 2))
data_disordered[:, :, :, 0] = data
data_disordered[:, :, :, 1] = data


h_ordered = {
    'CTYPE1': 'HPLN-TAN', 'CUNIT1': 'deg', 'CDELT1': 0.4, 'CRPIX1': 2, 'CRVAL1': 1, 'NAXIS1': 2,
    'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5,
    'NAXIS2': 3,
    'CTYPE3': 'WAVE    ', 'CUNIT3': 'Angstrom', 'CDELT3': 0.2, 'CRPIX3': 0, 'CRVAL3': 10,
    'NAXIS3': 4,
    'CTYPE4': 'TIME    ', 'CUNIT4': 'min', 'CDELT4': 0.4, 'CRPIX4': 0, 'CRVAL4': 0, 'NAXIS4': 2}
w_ordered = WCS(header=h_ordered, naxis=4)

data_ordered = np.zeros((2, 4, 3, 2))
data_ordered[0] = data.transpose()
data_ordered[1] = data.transpose()

h_rotated = {'CTYPE1': 'HPLN-TAN', 'CUNIT1': 'arcsec', 'CDELT1': 0.4, 'CRPIX1': 0,
             'CRVAL1': 0, 'NAXIS1': 5,
             'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'arcsec', 'CDELT2': 0.5, 'CRPIX2': 0,
             'CRVAL2': 0, 'NAXIS2': 5,
             'CTYPE3': 'Time    ', 'CUNIT3': 'seconds', 'CDELT3': 0.3, 'CRPIX3': 0,
             'CRVAL3': 0, 'NAXIS3': 2,
             'PC1_1': 0.714963912964, 'PC1_2': -0.699137151241, 'PC1_3': 0.0,
             'PC2_1': 0.699137151241, 'PC2_2': 0.714963912964, 'PC2_3': 0.0,
             'PC3_1': 0.0, 'PC3_2': 0.0, 'PC3_3': 1.0}
w_rotated = WCS(header=h_rotated, naxis=3)

data_rotated = np.array([[[1, 2, 3, 4, 6], [2, 4, 5, 3, 1], [0, -1, 2, 4, 2], [3, 5, 1, 2, 0]],
                         [[2, 4, 5, 1, 3], [1, 5, 2, 2, 4], [2, 3, 4, 0, 5], [0, 1, 2, 3, 4]]])

mask_cubem = data > 0
mask_cube = data >= 0
uncertaintym = data
uncertainty = np.sqrt(data)

mask_disordered = data_disordered > 0
uncertainty_disordered = data_disordered

mask_ordered = data_ordered > 0
uncertainty_ordered = data_ordered

cubem = NDCube(
    data,
    wm,
    mask=mask_cubem,
    uncertainty=uncertaintym,
    extra_coords=[('time', 0, u.Quantity(range(data.shape[0]), unit=u.pix)),
                  ('hello', 1, u.Quantity(range(data.shape[1]), unit=u.pix)),
                  ('bye', 2, u.Quantity(range(data.shape[2]), unit=u.pix))])

cube_disordered_inputs = (
    data_disordered, w_disordered, mask_disordered, uncertainty_disordered,
    [('spam', 0, u.Quantity(range(data_disordered.shape[0]), unit=u.pix)),
     ('hello', 1, u.Quantity(range(data_disordered.shape[1]), unit=u.pix)),
     ('bye', 2, u.Quantity(range(data_disordered.shape[2]), unit=u.pix))])
cube_disordered = NDCube(cube_disordered_inputs[0], cube_disordered_inputs[1],
                         mask=cube_disordered_inputs[2], uncertainty=cube_disordered_inputs[3],
                         extra_coords=cube_disordered_inputs[4])

cube_ordered = NDCubeOrdered(
    data_ordered,
    w_ordered,
    mask=mask_ordered,
    uncertainty=uncertainty_ordered,
    extra_coords=[('spam', 3, u.Quantity(range(data_disordered.shape[0]), unit=u.pix)),
                  ('hello', 2, u.Quantity(range(data_disordered.shape[1]), unit=u.pix)),
                  ('bye', 1, u.Quantity(range(data_disordered.shape[2]), unit=u.pix))])

cube = NDCube(
    data,
    wt,
    mask=mask_cube,
    uncertainty=uncertainty,
    missing_axis=[False, False, False, True],
    extra_coords=[('time', 0, u.Quantity(range(data.shape[0]), unit=u.pix)),
                  ('hello', 1, u.Quantity(range(data.shape[1]), unit=u.pix)),
                  ('bye', 2, u.Quantity(range(data.shape[2]), unit=u.pix))])

cubet = NDCube(
    data,
    wm,
    mask=mask_cubem,
    uncertainty=uncertaintym,
    extra_coords=[('time', 0, np.array([datetime.datetime(2000, 1, 1)+datetime.timedelta(minutes=i)
                                        for i in range(data.shape[0])])),
                  ('hello', 1, u.Quantity(range(data.shape[1]), unit=u.pix)),
                  ('bye', 2, u.Quantity(range(data.shape[2]), unit=u.pix))])

cube_rotated = NDCube(
    data_rotated,
    w_rotated,
    mask=mask_cube,
    uncertainty=uncertainty,
    missing_axis=[False, False, False],
    extra_coords=[('time', 0, u.Quantity(range(data_rotated.shape[0]), unit=u.pix)),
                  ('hello', 1, u.Quantity(range(data_rotated.shape[1]), unit=u.pix)),
                  ('bye', 2, u.Quantity(range(data_rotated.shape[2]), unit=u.pix))])

@pytest.mark.parametrize(
    "test_input,expected,mask,wcs,uncertainty,dimensions,world_axis_physical_types,extra_coords",
    [(cubem[:, 1],
      NDCube,
      mask_cubem[:, 1],
      _wcs_slicer(wm, [False, False, False], (slice(None, None, None), 1)),
      data[:, 1],
      u.Quantity((2, 4), unit=u.pix),
      ('custom:pos.helioprojective.lon', 'em.wl'),
      {'bye': {'axis': 1, 'value': u.Quantity(range(int(cubem.dimensions[2].value)), unit=u.pix)},
       'hello': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'time': {'axis': 0, 'value': u.Quantity(range(int(cubem.dimensions[0].value)), unit=u.pix)}}
      ),
     (cubem[:, 0:2],
      NDCube,
      mask_cubem[:, 0:2],
      _wcs_slicer(wm, [False, False, False], (slice(None, None, None), slice(0, 2, None))),
      data[:, 0:2],
      u.Quantity((2, 2, 4), unit=u.pix),
      ('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat', 'em.wl'),
      {'bye': {'axis': 2, 'value': u.Quantity(range(int(cubem.dimensions[2].value)), unit=u.pix)},
       'hello': {'axis': 1, 'value': u.Quantity(range(2), unit=u.pix)},
       'time': {'axis': 0, 'value': u.Quantity(range(int(cubem.dimensions[0].value)), unit=u.pix)}}
      ),
     (cubem[:, :],
      NDCube,
      mask_cubem[:, :],
      _wcs_slicer(wm, [False, False, False], (slice(None, None, None), slice(None, None, None))),
      data[:, :],
      u.Quantity((2, 3, 4), unit=u.pix),
      ('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat', 'em.wl'),
      {'time': {'axis': 0, 'value': u.Quantity(range(int(cubem.dimensions[0].value)), unit=u.pix)},
       'hello': {'axis': 1, 'value': u.Quantity(range(int(cubem.dimensions[1].value)), unit=u.pix)},
       'bye': {'axis': 2, 'value': u.Quantity(range(int(cubem.dimensions[2].value)), unit=u.pix)}}
      ),
     (cubem[1, 1],
      NDCube,
      mask_cubem[1, 1],
      _wcs_slicer(wm, [False, False, False], (1, 1)),
      data[1, 1],
      u.Quantity((4, ), unit=u.pix),
      tuple(['em.wl']),
      {'time': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'hello': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'bye': {'axis': 0, 'value': u.Quantity(range(int(cubem.dimensions[2].value)), unit=u.pix)}}
      ),
     (cubem[1, 0:2],
      NDCube,
      mask_cubem[1, 0:2],
      _wcs_slicer(wm, [False, False, False], (1, slice(0, 2, None))),
      data[1, 0:2],
      u.Quantity((2, 4), unit=u.pix),
      ('custom:pos.helioprojective.lat', 'em.wl'),
      {'time': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'hello': {'axis': 0, 'value': u.Quantity(range(2), unit=u.pix)},
       'bye': {'axis': 1, 'value': u.Quantity(range(int(cubem.dimensions[2].value)), unit=u.pix)}}
      ),
     (cubem[1, :],
      NDCube,
      mask_cubem[1, :],
      _wcs_slicer(wm, [False, False, False], (1, slice(None, None, None))),
      data[1, :],
      u.Quantity((3, 4), unit=u.pix),
      ('custom:pos.helioprojective.lat', 'em.wl'),
      {'time': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'hello': {'axis': 0, 'value': u.Quantity(range(int(cubem.dimensions[1].value)), unit=u.pix)},
       'bye': {'axis': 1, 'value': u.Quantity(range(int(cubem.dimensions[2].value)), unit=u.pix)}}
      ),
     (cube[:, 1],
      NDCube,
      mask_cube[:, 1],
      _wcs_slicer(wt, [True, False, False, False], (slice(None, None, None), 1)),
      uncertainty[:, 1],
      u.Quantity((2, 4), unit=u.pix),
      ('custom:pos.helioprojective.lat', 'time'),
      {'time': {'axis': 0, 'value': u.Quantity(range(int(cube.dimensions[0].value)), unit=u.pix)},
       'hello': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'bye': {'axis': 1, 'value': u.Quantity(range(int(cube.dimensions[2].value)), unit=u.pix)}}
      ),
     (cube[:, 0:2],
      NDCube,
      mask_cube[:, 0:2],
      _wcs_slicer(wt, [True, False, False, False], (slice(None, None, None), slice(0, 2, None))),
      uncertainty[:, 0:2],
      u.Quantity((2, 2, 4), unit=u.pix),
      ('custom:pos.helioprojective.lat', 'em.wl', 'time'),
      {'time': {'axis': 0, 'value': u.Quantity(range(int(cube.dimensions[0].value)), unit=u.pix)},
       'hello': {'axis': 1, 'value': u.Quantity(range(2), unit=u.pix)},
       'bye': {'axis': 2, 'value': u.Quantity(range(int(cube.dimensions[2].value)), unit=u.pix)}}
      ),
     (cube[:, :],
      NDCube,
      mask_cube[:, :],
      _wcs_slicer(wt, [True, False, False, False],
                  (slice(None, None, None), slice(None, None, None))),
      uncertainty[:, :],
      u.Quantity((2, 3, 4), unit=u.pix),
      ('custom:pos.helioprojective.lat', 'em.wl', 'time'),
      {'time': {'axis': 0, 'value': u.Quantity(range(int(cube.dimensions[0].value)), unit=u.pix)},
       'hello': {'axis': 1, 'value': u.Quantity(range(int(cube.dimensions[1].value)), unit=u.pix)},
       'bye': {'axis': 2, 'value': u.Quantity(range(int(cube.dimensions[2].value)), unit=u.pix)}}
      ),
     (cube[1, 1],
      NDCube,
      mask_cube[1, 1],
      _wcs_slicer(wt, [True, False, False, False], (1, 1)),
      uncertainty[1, 1],
      u.Quantity((4, ), unit=u.pix),
      tuple(['time']),
      {'time': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'hello': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'bye': {'axis': 0, 'value': u.Quantity(range(int(cube.dimensions[2].value)), unit=u.pix)}}
      ),
     (cube[1, 0:2],
      NDCube,
      mask_cube[1, 0:2],
      _wcs_slicer(wt, [True, False, False, False], (1, slice(0, 2, None))),
      uncertainty[1, 0:2],
      u.Quantity((2, 4), unit=u.pix),
      ('em.wl', 'time'),
      {'time': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'hello': {'axis': 0, 'value': u.Quantity(range(2), unit=u.pix)},
       'bye': {'axis': 1, 'value': u.Quantity(range(int(cube.dimensions[2].value)), unit=u.pix)}}
      ),
     (cube[1, :],
      NDCube,
      mask_cube[1, :],
      _wcs_slicer(wt, [True, False, False, False], (1, slice(0, 2, None))),
      uncertainty[1, :],
      u.Quantity((3, 4), unit=u.pix),
      ('em.wl', 'time'),
      {'time': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'hello': {'axis': 0, 'value': u.Quantity(range(int(cube.dimensions[1].value)), unit=u.pix)},
       'bye': {'axis': 1, 'value': u.Quantity(range(int(cube.dimensions[2].value)), unit=u.pix)}}
      )])
def test_slicing_second_axis(test_input, expected, mask, wcs, uncertainty,
                             dimensions, world_axis_physical_types, extra_coords):
    assert isinstance(test_input, expected)
    assert np.all(test_input.mask == mask)
    helpers.assert_wcs_are_equal(test_input.wcs, wcs[0])
    assert test_input.missing_axis == wcs[1]
    assert test_input.uncertainty.array.shape == uncertainty.shape
    assert np.all(test_input.dimensions.value == dimensions.value)
    assert test_input.dimensions.unit == dimensions.unit
    assert test_input.world_axis_physical_types == world_axis_physical_types
    helpers.assert_extra_coords_equal(test_input.extra_coords, extra_coords)


@pytest.mark.parametrize(
    "test_input,expected,mask,wcs,uncertainty,dimensions,world_axis_physical_types,extra_coords",
    [(cubem[1],
      NDCube,
      mask_cubem[1],
      _wcs_slicer(wm, [False, False, False], 1),
      data[1],
      u.Quantity((3, 4), unit=u.pix),
      ('custom:pos.helioprojective.lat', 'em.wl'),
      {'time': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'hello': {'axis': 0, 'value': u.Quantity(range(int(cubem.dimensions[1].value)), unit=u.pix)},
       'bye': {'axis': 1, 'value': u.Quantity(range(int(cubem.dimensions[2].value)), unit=u.pix)}}
      ),
     (cubem[0:2],
      NDCube,
      mask_cubem[0:2],
      _wcs_slicer(wm, [False, False, False], slice(0, 2, None)),
      data[0:2],
      u.Quantity((2, 3, 4), unit=u.pix),
      ('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat', 'em.wl'),
      {'time': {'axis': 0, 'value': u.Quantity(range(2), unit=u.pix)},
       'hello': {'axis': 1, 'value': u.Quantity(range(int(cubem.dimensions[1].value)), unit=u.pix)},
       'bye': {'axis': 2, 'value': u.Quantity(range(int(cubem.dimensions[2].value)), unit=u.pix)}}
      ),
     (cubem[:],
      NDCube,
      mask_cubem[:],
      _wcs_slicer(wm, [False, False, False], slice(None, None, None)),
      data[:],
      u.Quantity((2, 3, 4), unit=u.pix),
      ('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat', 'em.wl'),
      {'time': {'axis': 0, 'value': u.Quantity(range(int(cubem.dimensions[0].value)), unit=u.pix)},
       'hello': {'axis': 1, 'value': u.Quantity(range(int(cubem.dimensions[1].value)), unit=u.pix)},
       'bye': {'axis': 2, 'value': u.Quantity(range(int(cubem.dimensions[2].value)), unit=u.pix)}}
      ),
     (cube[1],
      NDCube,
      mask_cube[1],
      _wcs_slicer(wt, [True, False, False, False], 1),
      uncertainty[1],
      u.Quantity((3, 4), unit=u.pix),
      ('em.wl', 'time'),
      {'time': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'hello': {'axis': 0, 'value': u.Quantity(range(int(cube.dimensions[1].value)), unit=u.pix)},
       'bye': {'axis': 1, 'value': u.Quantity(range(int(cube.dimensions[2].value)), unit=u.pix)}}
      ),
     (cube[0:2],
      NDCube,
      mask_cube[0:2],
      _wcs_slicer(wt, [True, False, False, False], slice(0, 2, None)),
      uncertainty[0:2],
      u.Quantity((2, 3, 4), unit=u.pix),
      ('custom:pos.helioprojective.lat', 'em.wl', 'time'),
      {'time': {'axis': 0, 'value': u.Quantity(range(2), unit=u.pix)},
       'hello': {'axis': 1, 'value': u.Quantity(range(int(cube.dimensions[1].value)), unit=u.pix)},
       'bye': {'axis': 2, 'value': u.Quantity(range(int(cube.dimensions[2].value)), unit=u.pix)}}
      ),
     (cube[:],
      NDCube,
      mask_cube[:],
      _wcs_slicer(wt, [True, False, False, False], slice(None, None, None)),
      uncertainty[:],
      u.Quantity((2, 3, 4), unit=u.pix),
      ('custom:pos.helioprojective.lat', 'em.wl', 'time'),
      {'time': {'axis': 0, 'value': u.Quantity(range(int(cube.dimensions[0].value)), unit=u.pix)},
       'hello': {'axis': 1, 'value': u.Quantity(range(int(cube.dimensions[1].value)), unit=u.pix)},
       'bye': {'axis': 2, 'value': u.Quantity(range(int(cube.dimensions[2].value)), unit=u.pix)}}
      )])
def test_slicing_first_axis(test_input, expected, mask, wcs, uncertainty,
                            dimensions, world_axis_physical_types, extra_coords):
    assert isinstance(test_input, expected)
    assert np.all(test_input.mask == mask)
    helpers.assert_wcs_are_equal(test_input.wcs, wcs[0])
    assert test_input.missing_axis == wcs[1]
    assert test_input.uncertainty.array.shape == uncertainty.shape
    assert np.all(test_input.dimensions.value == dimensions.value)
    assert test_input.dimensions.unit == dimensions.unit
    assert test_input.world_axis_physical_types == world_axis_physical_types
    helpers.assert_extra_coords_equal(test_input.extra_coords, extra_coords)


@pytest.mark.parametrize(
    "test_input,expected,mask,wcs,uncertainty,dimensions,world_axis_physical_types,extra_coords",
    [(cubem[:, :, 1],
      NDCube,
      mask_cubem[:, :, 1],
      _wcs_slicer(wm, [False, False, False],
                  (slice(None, None, None), slice(None, None, None), 1)),
      data[:, :, 1],
      u.Quantity((2, 3), unit=u.pix),
      ('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat'),
      {'time': {'axis': 0, 'value': u.Quantity(range(int(cubem.dimensions[0].value)), unit=u.pix)},
       'hello': {'axis': 1, 'value': u.Quantity(range(int(cube.dimensions[1].value)), unit=u.pix)},
       'bye': {'axis': None, 'value': u.Quantity(1, unit=u.pix)}}
      ),
     (cubem[:, :, 0:2],
      NDCube,
      mask_cubem[:, :, 0:2],
      _wcs_slicer(wm, [False, False, False],
                  (slice(None, None, None), slice(None, None, None), slice(0, 2, None))),
      data[:, :, 0:2],
      u.Quantity((2, 3, 2), unit=u.pix),
      ('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat', 'em.wl'),
      {'time': {'axis': 0, 'value': u.Quantity(range(int(cubem.dimensions[0].value)), unit=u.pix)},
       'hello': {'axis': 1, 'value': u.Quantity(range(int(cubem.dimensions[1].value)), unit=u.pix)},
       'bye': {'axis': 2, 'value': u.Quantity(range(2), unit=u.pix)}}
      ),
     (cubem[:, :, :],
      NDCube,
      mask_cubem[:, :, :],
      _wcs_slicer(wm, [False, False, False],
                  (slice(None, None, None), slice(None, None, None), slice(None, None, None))),
      data[:, :, :],
      u.Quantity((2, 3, 4), unit=u.pix),
      ('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat', 'em.wl'),
      {'time': {'axis': 0, 'value': u.Quantity(range(int(cubem.dimensions[0].value)), unit=u.pix)},
       'hello': {'axis': 1, 'value': u.Quantity(range(int(cubem.dimensions[1].value)), unit=u.pix)},
       'bye': {'axis': 2, 'value': u.Quantity(range(int(cubem.dimensions[2].value)), unit=u.pix)}}
      ),
     (cubem[:, 1, 1],
      NDCube,
      mask_cubem[:, 1, 1],
      _wcs_slicer(wm, [False, False, False], (slice(None, None, None), 1, 1)),
      data[:, 1, 1],
      u.Quantity((2, ), unit=u.pix),
      tuple(['custom:pos.helioprojective.lon']),
      {'time': {'axis': 0, 'value': u.Quantity(range(int(cubem.dimensions[0].value)), unit=u.pix)},
       'hello': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'bye': {'axis': None, 'value': u.Quantity(1, unit=u.pix)}}
      ),
     (cubem[:, 1, 0:2],
      NDCube,
      mask_cubem[:, 1, 0:2],
      _wcs_slicer(wm, [False, False, False], (slice(None, None, None), 1, slice(0, 2, None))),
      data[:, 1, 0:2],
      u.Quantity((2, 2), unit=u.pix),
      ('custom:pos.helioprojective.lon', 'em.wl'),
      {'time': {'axis': 0, 'value': u.Quantity(range(int(cubem.dimensions[0].value)), unit=u.pix)},
       'hello': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'bye': {'axis': 1, 'value': u.Quantity(range(2), unit=u.pix)}}
      ),
     (cubem[:, 1, :],
      NDCube,
      mask_cubem[:, 1, :],
      _wcs_slicer(wm, [False, False, False], (slice(None, None, None), 1, slice(None, None, None))),
      data[:, 1, :],
      u.Quantity((2, 4), unit=u.pix),
      ('custom:pos.helioprojective.lon', 'em.wl'),
      {'time': {'axis': 0, 'value': u.Quantity(range(int(cubem.dimensions[0].value)), unit=u.pix)},
       'hello': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'bye': {'axis': 1, 'value': u.Quantity(range(int(cubem.dimensions[2].value)), unit=u.pix)}}
      ),
     (cubem[1, :, 1],
      NDCube,
      mask_cubem[1, :, 1],
      _wcs_slicer(wm, [False, False, False], (1, slice(None, None, None), 1)),
      data[1, :, 1],
      u.Quantity((3, ), unit=u.pix),
      tuple(['custom:pos.helioprojective.lat']),
      {'time': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'hello': {'axis': 0, 'value': u.Quantity(range(int(cubem.dimensions[1].value)), unit=u.pix)},
       'bye': {'axis': None, 'value': u.Quantity(1, unit=u.pix)}}
      ),
     (cubem[1, :, 0:2],
      NDCube,
      mask_cubem[1, :, 0:2],
      _wcs_slicer(wm, [False, False, False], (1, slice(None, None, None), slice(0, 2, None))),
      data[1, :, 0:2],
      u.Quantity((3, 2), unit=u.pix),
      ('custom:pos.helioprojective.lat', 'em.wl'),
      {'time': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'hello': {'axis': 0, 'value': u.Quantity(range(int(cubem.dimensions[1].value)), unit=u.pix)},
       'bye': {'axis': 1, 'value': u.Quantity(range(2), unit=u.pix)}}
      ),
     (cubem[1, :, :],
      NDCube,
      mask_cubem[1, :, :],
      _wcs_slicer(wm, [False, False, False], (1, slice(None, None, None), slice(None, None, None))),
      data[1, :, :],
      u.Quantity((3, 4), unit=u.pix),
      ('custom:pos.helioprojective.lat', 'em.wl'),
      {'time': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'hello': {'axis': 0, 'value': u.Quantity(range(int(cubem.dimensions[1].value)), unit=u.pix)},
       'bye': {'axis': 1, 'value': u.Quantity(range(int(cubem.dimensions[2].value)), unit=u.pix)}}
      ),
     (cubem[1, 1, 1],
      NDCube,
      mask_cubem[1, 1, 1],
      _wcs_slicer(wm, [False, False, False], (1, 1, 1)),
      data[1, 1, 1],
      u.Quantity((), unit=u.pix),
      (),
      {'time': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'hello': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'bye': {'axis': None, 'value': u.Quantity(1, unit=u.pix)}}
      ),
     (cubem[1, 1, 0:2],
      NDCube,
      mask_cubem[1, 1, 0:2],
      _wcs_slicer(wm, [False, False, False], (1, 1, slice(0, 2, None))),
      data[1, 1, 0:2],
      u.Quantity((2, ), unit=u.pix),
      tuple(['em.wl']),
      {'time': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'hello': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'bye': {'axis': 0, 'value': u.Quantity(range(2), unit=u.pix)}}
      ),
     (cubem[1, 1, :],
      NDCube,
      mask_cubem[1, 1, :],
      _wcs_slicer(wm, [False, False, False], (1, 1, slice(None, None, None))),
      data[1, 1, :],
      u.Quantity((4, ), unit=u.pix),
      tuple(['em.wl']),
      {'time': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'hello': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'bye': {'axis': 0, 'value': u.Quantity(range(int(cubem.dimensions[2].value)), unit=u.pix)}}
      ),
     (cube[:, :, 1],
      NDCube,
      mask_cube[:, :, 1],
      _wcs_slicer(wt, [True, False, False, False],
                  (slice(None, None, None), slice(None, None, None), 1)),
      uncertainty[:, :, 1],
      u.Quantity((2, 3), unit=u.pix),
      ('custom:pos.helioprojective.lat', 'em.wl'),
      {'time': {'axis': 0, 'value': u.Quantity(range(int(cube.dimensions[0].value)), unit=u.pix)},
       'hello': {'axis': 1, 'value': u.Quantity(range(int(cube.dimensions[1].value)), unit=u.pix)},
       'bye': {'axis': None, 'value': u.Quantity(1, unit=u.pix)}}
      ),
     (cube[:, :, 0:2],
      NDCube,
      mask_cube[:, :, 0:2],
      _wcs_slicer(wt, [True, False, False, False],
                  (slice(None, None, None), slice(None, None, None), slice(0, 2, None))),
      uncertainty[:, :, 0:2],
      u.Quantity((2, 3, 2), unit=u.pix),
      ('custom:pos.helioprojective.lat', 'em.wl', 'time'),
      {'time': {'axis': 0, 'value': u.Quantity(range(int(cube.dimensions[0].value)), unit=u.pix)},
       'hello': {'axis': 1, 'value': u.Quantity(range(int(cube.dimensions[1].value)), unit=u.pix)},
       'bye': {'axis': 2, 'value': u.Quantity(range(2), unit=u.pix)}}
      ),
     (cube[:, :, :],
      NDCube,
      mask_cube[:, :, :],
      _wcs_slicer(wt, [True, False, False, False],
                  (slice(None, None, None), slice(None, None, None), slice(None, None, None))),
      uncertainty[:, :, :],
      u.Quantity((2, 3, 4), unit=u.pix),
      ('custom:pos.helioprojective.lat', 'em.wl', 'time'),
      {'time': {'axis': 0, 'value': u.Quantity(range(int(cube.dimensions[0].value)), unit=u.pix)},
       'hello': {'axis': 1, 'value': u.Quantity(range(int(cube.dimensions[1].value)), unit=u.pix)},
       'bye': {'axis': 2, 'value': u.Quantity(range(int(cube.dimensions[2].value)), unit=u.pix)}}
      ),
     (cube[:, 1, 1],
      NDCube,
      mask_cube[:, 1, 1],
      _wcs_slicer(wt, [True, False, False, False], (slice(None, None, None), 1, 1)),
      uncertainty[:, 1, 1],
      u.Quantity((2, ), unit=u.pix),
      tuple(['custom:pos.helioprojective.lat']),
      {'time': {'axis': 0, 'value': u.Quantity(range(int(cube.dimensions[0].value)), unit=u.pix)},
       'hello': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'bye': {'axis': None, 'value': u.Quantity(1, unit=u.pix)}}
      ),
     (cube[:, 1, 0:2],
      NDCube,
      mask_cube[:, 1, 0:2],
      _wcs_slicer(wt, [True, False, False, False], (slice(None, None, None), 1, slice(0, 2, None))),
      uncertainty[:, 1, 0:2],
      u.Quantity((2, 2), unit=u.pix),
      ('custom:pos.helioprojective.lat', 'time'),
      {'time': {'axis': 0, 'value': u.Quantity(range(int(cube.dimensions[0].value)), unit=u.pix)},
       'hello': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'bye': {'axis': 1, 'value': u.Quantity(range(2), unit=u.pix)}}
      ),
     (cube[:, 1, :],
      NDCube,
      mask_cube[:, 1, :],
      _wcs_slicer(wt, [True, False, False, False],
                  (slice(None, None, None), 1, slice(None, None, None))),
      uncertainty[:, 1, :],
      u.Quantity((2, 4), unit=u.pix),
      ('custom:pos.helioprojective.lat', 'time'),
      {'time': {'axis': 0, 'value': u.Quantity(range(int(cube.dimensions[0].value)), unit=u.pix)},
       'hello': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'bye': {'axis': 1, 'value': u.Quantity(range(int(cube.dimensions[2].value)), unit=u.pix)}}
      ),
     (cube[1, :, 1],
      NDCube,
      mask_cube[1, :, 1],
      _wcs_slicer(wt, [True, False, False, False], (1, slice(None, None, None), 1)),
      uncertainty[1, :, 1],
      u.Quantity((3, ), unit=u.pix),
      tuple(['em.wl']),
      {'time': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'hello': {'axis': 0, 'value': u.Quantity(range(int(cube.dimensions[1].value)), unit=u.pix)},
       'bye': {'axis': None, 'value': u.Quantity(1, unit=u.pix)}}
      ),
     (cube[1, :, 0:2],
      NDCube,
      mask_cube[1, :, 0:2],
      _wcs_slicer(wt, [True, False, False, False], (1, slice(None, None, None), slice(0, 2, None))),
      uncertainty[1, :, 0:2],
      u.Quantity((3, 2), unit=u.pix),
      ('em.wl', 'time'),
      {'time': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'hello': {'axis': 0, 'value': u.Quantity(range(int(cube.dimensions[1].value)), unit=u.pix)},
       'bye': {'axis': 1, 'value': u.Quantity(range(2), unit=u.pix)}}
      ),
     (cube[1, :, :],
      NDCube,
      mask_cube[1, :, :],
      _wcs_slicer(wt, [True, False, False, False],
                  (1, slice(None, None, None), slice(None, None, None))),
      uncertainty[1, :, :],
      u.Quantity((3, 4), unit=u.pix),
      ('em.wl', 'time'),
      {'time': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'hello': {'axis': 0, 'value': u.Quantity(range(int(cube.dimensions[1].value)), unit=u.pix)},
       'bye': {'axis': 1, 'value': u.Quantity(range(int(cube.dimensions[2].value)), unit=u.pix)}}
      ),
     (cube[1, 1, 1],
      NDCube,
      mask_cube[1, 1, 1],
      _wcs_slicer(wt, [True, False, False, False], (1, 1, 1)),
      uncertainty[1, 1, 1],
      u.Quantity((), unit=u.pix),
      (),
      {'time': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'hello': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'bye': {'axis': None, 'value': u.Quantity(1, unit=u.pix)}}
      ),
     (cube[1, 1, 0:2],
      NDCube,
      mask_cube[1, 1, 0:2],
      _wcs_slicer(wt, [True, False, False, False], (1, 1, slice(0, 2, None))),
      uncertainty[1, 1, 0:2],
      u.Quantity((2, ), unit=u.pix),
      tuple(['time']),
      {'time': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'hello': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'bye': {'axis': 0, 'value': u.Quantity(range(2), unit=u.pix)}}
      ),
     (cube[1, 1, :],
      NDCube,
      mask_cube[1, 1, :],
      _wcs_slicer(wt, [True, False, False, False], (1, 1, slice(0, 2, None))),
      uncertainty[1, 1, :],
      u.Quantity((4, ), unit=u.pix),
      tuple(['time']),
      {'time': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'hello': {'axis': None, 'value': u.Quantity(1, unit=u.pix)},
       'bye': {'axis': 0, 'value': u.Quantity(range(int(cube.dimensions[2].value)), unit=u.pix)}}
      )])
def test_slicing_third_axis(test_input, expected, mask, wcs, uncertainty,
                            dimensions, world_axis_physical_types, extra_coords):
    assert isinstance(test_input, expected)
    assert np.all(test_input.mask == mask)
    helpers.assert_wcs_are_equal(test_input.wcs, wcs[0])
    assert test_input.missing_axis == wcs[1]
    assert test_input.uncertainty.array.shape == uncertainty.shape
    assert np.all(test_input.dimensions.value == dimensions.value)
    assert test_input.dimensions.unit == dimensions.unit
    assert test_input.world_axis_physical_types == world_axis_physical_types
    helpers.assert_extra_coords_equal(test_input.extra_coords, extra_coords)


@pytest.mark.parametrize("test_input", [(cubem)])
def test_slicing_error(test_input):
    with pytest.raises(IndexError):
        test_input[None]
    with pytest.raises(IndexError):
        test_input[0, None]


@pytest.mark.parametrize("test_input,expected", [
    (cubem[1].pixel_to_world(*[
        u.Quantity(np.arange(4), unit=u.pix),
        u.Quantity(np.arange(4), unit=u.pix)
        ])[0],
     wm.all_pix2world(
         u.Quantity(np.arange(4), unit=u.pix),
         u.Quantity(np.arange(4), unit=u.pix), wm.wcs.crpix[2] - 1, 0)[-2]),
    (cubem[1].pixel_to_world(*[
        u.Quantity(np.arange(4), unit=u.pix),
        u.Quantity(np.arange(4), unit=u.pix)
        ])[1],
     wm.all_pix2world(
         u.Quantity(np.arange(4), unit=u.pix),
         u.Quantity(np.arange(4), unit=u.pix), wm.wcs.crpix[2] - 1, 0)[0]),
    (cubem[0:2].pixel_to_world(*[
        u.Quantity(np.arange(4), unit=u.pix),
        u.Quantity(np.arange(4), unit=u.pix),
        u.Quantity(np.arange(4), unit=u.pix)
        ])[0],
     wm.all_pix2world(
         u.Quantity(np.arange(4), unit=u.pix),
         u.Quantity(np.arange(4), unit=u.pix),
         u.Quantity(np.arange(4), unit=u.pix), 0)[-1]),
    (cubem[0:2].pixel_to_world(*[
        u.Quantity(np.arange(4), unit=u.pix),
        u.Quantity(np.arange(4), unit=u.pix),
        u.Quantity(np.arange(4), unit=u.pix)
        ])[1],
     wm.all_pix2world(
         u.Quantity(np.arange(4), unit=u.pix),
         u.Quantity(np.arange(4), unit=u.pix),
         u.Quantity(np.arange(4), unit=u.pix), 0)[1]),
    (cubem[0:2].pixel_to_world(*[
        u.Quantity(np.arange(4), unit=u.pix),
        u.Quantity(np.arange(4), unit=u.pix),
        u.Quantity(np.arange(4), unit=u.pix)
        ])[2],
     wm.all_pix2world(
         u.Quantity(np.arange(4), unit=u.pix),
         u.Quantity(np.arange(4), unit=u.pix),
         u.Quantity(np.arange(4), unit=u.pix), 0)[0]),
    (cube[1].pixel_to_world(*[
        u.Quantity(np.arange(4), unit=u.pix),
        u.Quantity(np.arange(4), unit=u.pix)
        ])[0],
     wt.all_pix2world(
         u.Quantity(np.arange(4), unit=u.pix),
         u.Quantity(np.arange(4), unit=u.pix), wt.wcs.crpix[2] - 1,
         wt.wcs.crpix[3] - 1, 0)[1]),
    (cube[1].pixel_to_world(*[
        u.Quantity(np.arange(4), unit=u.pix),
        u.Quantity(np.arange(4), unit=u.pix)
        ])[1],
     wt.all_pix2world(
         u.Quantity(np.arange(4), unit=u.pix),
         u.Quantity(np.arange(4), unit=u.pix), wt.wcs.crpix[2] - 1,
         wt.wcs.crpix[3] - 1, 0)[0]),
    (cube[0:2].pixel_to_world(*[
        u.Quantity(np.arange(4), unit=u.pix),
        u.Quantity(np.arange(4), unit=u.pix),
        u.Quantity(np.arange(4), unit=u.pix)
        ])[0],
     wt.all_pix2world(
         u.Quantity(np.arange(4), unit=u.pix),
         u.Quantity(np.arange(4), unit=u.pix),
         u.Quantity(np.arange(4), unit=u.pix), wt.wcs.crpix[3] - 1, 0)[2]),
    (cube[0:2].pixel_to_world(*[
        u.Quantity(np.arange(4), unit=u.pix),
        u.Quantity(np.arange(4), unit=u.pix),
        u.Quantity(np.arange(4), unit=u.pix)
        ])[1],
     wt.all_pix2world(
         u.Quantity(np.arange(4), unit=u.pix),
         u.Quantity(np.arange(4), unit=u.pix),
         u.Quantity(np.arange(4), unit=u.pix), wt.wcs.crpix[3] - 1, 0)[1]),
    (cube[0:2].pixel_to_world(*[
        u.Quantity(np.arange(4), unit=u.pix),
        u.Quantity(np.arange(4), unit=u.pix),
        u.Quantity(np.arange(4), unit=u.pix)
        ])[2],
     wt.all_pix2world(
         u.Quantity(np.arange(4), unit=u.pix),
         u.Quantity(np.arange(4), unit=u.pix),
         u.Quantity(np.arange(4), unit=u.pix), wt.wcs.crpix[3] - 1, 0)[0])])
def test_pixel_to_world(test_input, expected):
    assert np.all(test_input.value == expected)


@pytest.mark.parametrize("test_input,expected", [
    (cubem[1].world_to_pixel(*[
        u.Quantity(np.arange(4), unit=u.deg),
        u.Quantity(np.arange(4), unit=u.m)
        ])[1],
     wm.all_world2pix(
         u.Quantity(np.arange(4), unit=u.deg),
         u.Quantity(np.arange(4), unit=u.m), wm.wcs.crpix[2] - 1, 0)[0]),
    (cubem[0:2].world_to_pixel(*[
        u.Quantity(np.arange(4), unit=u.deg),
        u.Quantity(np.arange(4), unit=u.deg),
        u.Quantity(np.arange(4), unit=u.m)
        ])[0],
     wm.all_world2pix(
         u.Quantity(np.arange(4), unit=u.deg),
         u.Quantity(np.arange(4), unit=u.deg),
         u.Quantity(np.arange(4), unit=u.m), 0)[-1]),
    (cubem[0:2].world_to_pixel(*[
        u.Quantity(np.arange(4), unit=u.deg),
        u.Quantity(np.arange(4), unit=u.deg),
        u.Quantity(np.arange(4), unit=u.m)
        ])[1],
     wm.all_world2pix(
         u.Quantity(np.arange(4), unit=u.deg),
         u.Quantity(np.arange(4), unit=u.deg),
         u.Quantity(np.arange(4), unit=u.m), 0)[1]),
    (cubem[0:2].world_to_pixel(*[
        u.Quantity(np.arange(4), unit=u.deg),
        u.Quantity(np.arange(4), unit=u.deg),
        u.Quantity(np.arange(4), unit=u.m)
        ])[2],
     wm.all_world2pix(
         u.Quantity(np.arange(4), unit=u.deg),
         u.Quantity(np.arange(4), unit=u.deg),
         u.Quantity(np.arange(4), unit=u.m), 0)[0]),
    (cube[1].world_to_pixel(*[
        u.Quantity(np.arange(4), unit=u.m),
        u.Quantity(np.arange(4), unit=u.min)
        ])[0],
     wt.all_world2pix(
         u.Quantity(np.arange(4), unit=u.m),
         u.Quantity(np.arange(4), unit=u.min), wt.wcs.crpix[2] - 1,
         wt.wcs.crpix[3] - 1, 0)[1]),
    (cube[1].world_to_pixel(*[
        u.Quantity(np.arange(4), unit=u.m),
        u.Quantity(np.arange(4), unit=u.min)
        ])[1],
     wt.all_world2pix(
         u.Quantity(np.arange(4), unit=u.m),
         u.Quantity(np.arange(4), unit=u.min), wt.wcs.crpix[2] - 1,
         wt.wcs.crpix[3] - 1, 0)[0]),
    (cube[0:2].world_to_pixel(*[
        u.Quantity(np.arange(4), unit=u.deg),
        u.Quantity(np.arange(4), unit=u.m),
        u.Quantity(np.arange(4), unit=u.min)
        ])[0],
     wt.all_world2pix(
         u.Quantity(np.arange(4), unit=u.deg),
         u.Quantity(np.arange(4), unit=u.m),
         u.Quantity(np.arange(4), unit=u.min), wt.wcs.crpix[3] - 1, 0)[2]),
    (cube[0:2].world_to_pixel(*[
        u.Quantity(np.arange(4), unit=u.deg),
        u.Quantity(np.arange(4), unit=u.m),
        u.Quantity(np.arange(4), unit=u.min)
        ])[1],
     wt.all_world2pix(
         u.Quantity(np.arange(4), unit=u.deg),
         u.Quantity(np.arange(4), unit=u.m),
         u.Quantity(np.arange(4), unit=u.min), wt.wcs.crpix[3] - 1, 0)[1]),
    (cube[0:2].world_to_pixel(*[
        u.Quantity(np.arange(4), unit=u.deg),
        u.Quantity(np.arange(4), unit=u.m),
        u.Quantity(np.arange(4), unit=u.min)
        ])[2],
     wt.all_world2pix(
         u.Quantity(np.arange(4), unit=u.deg),
         u.Quantity(np.arange(4), unit=u.m),
         u.Quantity(np.arange(4), unit=u.min), wt.wcs.crpix[3] - 1, 0)[0])])
def test_world_to_pixel(test_input, expected):
    assert np.allclose(test_input.value, expected)


@pytest.mark.parametrize("test_input,expected", [
    ((cubem, [0.7*u.deg, 1.3e-5*u.deg, 1.02e-9*u.m], [1*u.deg, 1*u.deg, 4.e-11*u.m], None),
     cubem[:, :, :3]),
    ((cube_rotated, [0*u.s, 1.5*u.arcsec, 0*u.arcsec], [1*u.s, 1*u.arcsec, 0.5*u.arcsec], None),
     cube_rotated[:, :4, 1:5]),
    ((cubem, [0.7*u.deg, 1.3e-5*u.deg, 1.02e-9*u.m], None, [1.7*u.deg, 1*u.deg, 1.06e-9*u.m]),
     cubem[:, :, :3]),
    ((cube_rotated, [0*u.s, 1.5*u.arcsec, 0*u.arcsec], None, [1*u.s, 2.5*u.arcsec, 0.5*u.arcsec]),
     cube_rotated[:, :4, 1:5]),
    ((cube_rotated, [0, 1.5, 0], None, [1, 2.5, 0.5], ['s', 'arcsec', 'arcsec']),
     cube_rotated[:, :4, 1:5])])
def test_crop_by_coords(test_input, expected):
    helpers.assert_cubes_equal(
        test_input[0].crop_by_coords(*test_input[1:]), expected)


@pytest.mark.parametrize("test_input", [
    (ValueError, cubem, u.Quantity([0], unit=u.deg), u.Quantity([1.5, 2.], unit=u.deg), None),
    (ValueError, cubem, [1*u.s], [1*u.s], [1*u.s]),
    (ValueError, cubem, u.Quantity([0], unit=u.deg), None, u.Quantity([1.5, 2.], unit=u.deg)),
    (ValueError, cubem, [1], None, [1], ['s', 'deg']),
    (TypeError, cubem, [1, 2, 3], None, [2, 3, 4])])
def test_crop_by_coords_error(test_input):
    with pytest.raises(test_input[0]):
        test_input[1].crop_by_coords(*test_input[2:])


@pytest.mark.parametrize(
    "test_input,expected",
    [((cubem, 0*u.pix, 1.5*u.pix, "time"), cubem[0:2]),
     ((cube, 0*u.pix, 1.5*u.pix, "bye"), cube[:, :, 0:2]),
     ((cubet, datetime.datetime(2000, 1, 1), datetime.timedelta(minutes=2), "time"), cubet[:2])])
def test_crop_by_extra_coord(test_input, expected):
    helpers.assert_cubes_equal(
        test_input[0].crop_by_extra_coord(*tuple(test_input[1:])), expected)


@pytest.mark.parametrize("test_input,expected", [
    (cube_disordered_inputs, cube_ordered)])
def test_ndcubeordered(test_input, expected):
    helpers.assert_cubes_equal(
        NDCubeOrdered(test_input[0], test_input[1], mask=test_input[2],
                      uncertainty=test_input[3], extra_coords=test_input[4]),
        expected)


@pytest.mark.parametrize("test_input,expected", [
    ((cubem, [2]), u.Quantity([1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09], unit=u.m)),
    ((cubem, ['em']), u.Quantity([1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09], unit=u.m))
    ])
def test_all_world_coords_with_input(test_input, expected):
    all_coords = test_input[0].axis_world_coords(*test_input[1])
    for i in range(len(all_coords)):
        np.testing.assert_allclose(all_coords[i].value, expected[i].value)
        assert all_coords[i].unit == expected[i].unit


@pytest.mark.parametrize("test_input,expected", [
    (cubem, (u.Quantity([[0.60002173, 0.59999127, 0.5999608],
                                 [1., 1., 1.]], unit=u.deg),
                     u.Quantity([[1.26915033e-05, 4.99987815e-01, 9.99962939e-01],
                                 [1.26918126e-05, 5.00000000e-01, 9.99987308e-01]], unit=u.deg),
                     u.Quantity([1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09], unit=u.m))),
    ((cubem[:, :, 0]), (u.Quantity([[0.60002173, 0.59999127, 0.5999608],
                                    [1., 1., 1.]], unit=u.deg),
                        u.Quantity([[1.26915033e-05, 4.99987815e-01, 9.99962939e-01],
                                    [1.26918126e-05, 5.00000000e-01, 9.99987308e-01]],
                                   unit=u.deg)))
    ])
def test_axis_world_coords_without_input(test_input, expected):
    all_coords = test_input.axis_world_coords()
    for i in range(len(all_coords)):
        np.testing.assert_allclose(all_coords[i].value, expected[i].value)
        assert all_coords[i].unit == expected[i].unit
