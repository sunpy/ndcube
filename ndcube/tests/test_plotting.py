# -*- coding: utf-8 -*-
import pytest
import datetime

import numpy as np
import astropy.units as u
import matplotlib
import sunpy.visualization.imageanimator

from ndcube import NDCube
from ndcube.utils.wcs import WCS
from ndcube.mixins import plotting


# sample data for tests
# TODO: use a fixture reading from a test file. file TBD.
ht = {'CTYPE3': 'HPLT-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.5, 'CRPIX3': 0, 'CRVAL3': 0, 'NAXIS3': 2,
      'CTYPE2': 'WAVE    ', 'CUNIT2': 'Angstrom', 'CDELT2': 0.2, 'CRPIX2': 0, 'CRVAL2': 0,
      'NAXIS2': 3,
      'CTYPE1': 'TIME    ', 'CUNIT1': 'min', 'CDELT1': 0.4, 'CRPIX1': 0, 'CRVAL1': 0, 'NAXIS1': 4}
wt = WCS(header=ht, naxis=3)

hm = {'CTYPE1': 'WAVE    ', 'CUNIT1': 'Angstrom', 'CDELT1': 0.2, 'CRPIX1': 0, 'CRVAL1': 10,
      'NAXIS1': 4,
      'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5,
      'NAXIS2': 3,
      'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 2}
wm = WCS(header=hm, naxis=3)

data = np.array([[[1, 2, 3, 4], [2, 4, 5, 3], [0, -1, 2, 3]],
                 [[2, 4, 5, 1], [10, 5, 2, 2], [10, 3, 3, 0]]])
uncertainty = np.sqrt(data)
mask_cube = data < 0

cube = NDCube(
    data,
    wt,
    mask=mask_cube,
    uncertainty=uncertainty,
    missing_axis=[False, False, False, True],
    extra_coords=[('time', 0, u.Quantity(range(data.shape[0]), unit=u.pix)),
                  ('hello', 1, u.Quantity(range(data.shape[1]), unit=u.pix)),
                  ('bye', 2, u.Quantity(range(data.shape[2]), unit=u.pix))])

cubem = NDCube(
    data,
    wm,
    mask=mask_cube,
    uncertainty=uncertainty,
    extra_coords=[('time', 0, u.Quantity(range(data.shape[0]), unit=u.pix)),
                  ('hello', 1, u.Quantity(range(data.shape[1]), unit=u.pix)),
                  ('bye', 2, u.Quantity(range(data.shape[2]), unit=u.pix))])


@pytest.mark.parametrize("test_input, test_kwargs, expected_values", [
    (cube[0, 0], {},
     (u.Quantity([0.4, 0.8, 1.2, 1.6], unit="min"), np.array([1, 2, 3, 4]),
      "", "", (0.4, 1.6), (1, 4)))
    ])
def test_cube_plot_1D(test_input, test_kwargs, expected_values):
    # Unpack expected properties.
    expected_xdata, expected_ydata, expected_xlabel, expected_ylabel, \
      expected_xlim, expected_ylim = expected_values
    # Run plot method.
    output = test_input.plot(**test_kwargs)
    # Check plot properties are correct.
    assert type(output) is list
    assert len(output) == 1
    output = output[0]
    assert type(output) is matplotlib.lines.Line2D
    output_xdata = (output.axes.lines[0].get_xdata())
    if type(expected_xdata) == u.Quantity:
        assert output_xdata.unit == expected_xdata.unit
        assert np.allclose(output_xdata.value, expected_xdata.value)
    else:
        np.testing.assert_array_equal(output.axes.lines[0].get_xdata(), expected_xdata)
    if type(expected_ydata) == u.Quantity:
        assert output_ydata.unit == expected_ydata.unit
        assert np.allclose(output_ydata.value, expected_ydata.value)
    else:
        np.testing.assert_array_equal(output.axes.lines[0].get_ydata(), expected_ydata)
    assert output.axes.get_xlabel() == expected_xlabel
    assert output.axes.get_ylabel() == expected_ylabel
    output_xlim = output.axes.get_xlim()
    assert output_xlim[0] <= expected_xlim[0]
    assert output_xlim[1] >= expected_xlim[1]
    output_ylim = output.axes.get_ylim()
    assert output_ylim[0] <= expected_ylim[0]
    assert output_ylim[1] >= expected_ylim[1]
    

@pytest.mark.parametrize("test_input, test_kwargs, expected_values", [
    (cube[0], {},
     (cube[0].data, "", "",
      (-0.5, 3.5, 2.5, -0.5)))
    ])
def test_cube_plot_2D(test_input, test_kwargs, expected_values):
    # Unpack expected properties.
    expected_data, expected_xlabel, expected_ylabel, expected_extent = \
      expected_values
    # Run plot method.
    output = test_input.plot(**test_kwargs)
    # Check plot properties are correct.
    assert type(output) is matplotlib.image.AxesImage
    np.testing.assert_array_equal(output.get_array(), expected_data)
    assert output.axes.xaxis.get_label_text() == expected_xlabel
    assert output.axes.yaxis.get_label_text() == expected_ylabel
    assert np.allclose(output.get_extent(), expected_extent)


@pytest.mark.parametrize("test_input, test_kwargs, expected_values", [
    (cubem, {},
     (cubem.data, [np.array([0., 2.]), [0, 3], [0, 4]], "", ""))
    ])
def test_cube_animate_ND(test_input, test_kwargs, expected_values):
    # Unpack expected properties.
    expected_data, expected_axis_ranges, expected_xlabel, expected_ylabel = expected_values
    # Run plot method.
    output = test_input.plot(**test_kwargs)
    # Check plot properties are correct.
    assert type(output) is sunpy.visualization.imageanimator.ImageAnimatorWCS
    np.testing.assert_array_equal(output.data, expected_data)
    assert output.axes.xaxis.get_label_text() == expected_xlabel
    assert output.axes.yaxis.get_label_text() == expected_ylabel


def test_cube_plot_ND_as_2DAnimation():
    pass


@pytest.mark.parametrize("input_values, expected_values", [
    ((None, None, None, None, {"image_axes": [-1, -2],
                               "axis_ranges": [np.arange(3), np.arange(3)],
                               "unit_x_axis": "km",
                               "unit_y_axis": u.s,
                               "unit": u.W}),
     ([-1, -2], [np.arange(3), np.arange(3)], ["km", u.s], u.W, {})),
    (([-1, -2], [np.arange(3), np.arange(3)], ["km", u.s], u.W, {}),
     ([-1, -2], [np.arange(3), np.arange(3)], ["km", u.s], u.W, {})),
    (([-1], None, None, None, {"unit_x_axis": "km"}),
     ([-1], None, "km", None, {})),
    (([-1, -2], None, None, None, {"unit_x_axis": "km"}),
     (([-1, -2], None, ["km", None], None, {}))),
    (([-1, -2], None, None, None, {"unit_y_axis": "km"}),
     (([-1, -2], None, [None, "km"], None, {})))
    ])
def test_support_101_plot_API(input_values, expected_values):
    # Define expected values.
    expected_plot_axis_indices, expected_axes_coordinates, expected_axes_units, \
      expected_data_unit, expected_kwargs = expected_values
    # Run function
    output_plot_axis_indices, output_axes_coordinates, output_axes_units, \
      output_data_unit, output_kwargs = plotting._support_101_plot_API(*input_values)
    # Check values are correct
    assert output_plot_axis_indices == expected_plot_axis_indices
    if expected_axes_coordinates is None:
        assert output_axes_coordinates == expected_axes_coordinates
    elif type(expected_axes_coordinates) is list:
        for i, ac in enumerate(output_axes_coordinates):
            np.testing.assert_array_equal(ac, expected_axes_coordinates[i])
    assert output_axes_units == expected_axes_units
    assert output_data_unit == expected_data_unit
    assert output_kwargs == expected_kwargs


@pytest.mark.parametrize("input_values", [
    ([0, 1], None, None, None, {"image_axes": [-1, -2]}),
    (None, [np.arange(1, 4), np.arange(1, 4)], None, None,
      {"axis_ranges": [np.arange(3), np.arange(3)]}),
    (None, None, [u.s, "km"], None, {"unit_x_axis": u.W}),
    (None, None, [u.s, "km"], None, {"unit_y_axis": u.W}),
    (None, None, None, u.s, {"unit": u.W}),
    ([0, 1, 2], None, None, None, {"unit_x_axis": [u.s, u.km, u.W]}),
    ])
def test_support_101_plot_API_errors(input_values):
    with pytest.raises(ValueError):
        output = plotting._support_101_plot_API(*input_values)
