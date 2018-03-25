# -*- coding: utf-8 -*-
import pytest
import datetime

import numpy as np
import astropy.units as u
import matplotlib

from ndcube import NDCube, NDCubeSequence
from ndcube.utils.wcs import WCS
import ndcube.mixins.sequence_plotting

# sample data for tests
# TODO: use a fixture reading from a test file. file TBD.
data = np.array([[[1, 2, 3, 4], [2, 4, 5, 3], [0, -1, 2, 3]],
                 [[2, 4, 5, 1], [10, 5, 2, 2], [10, 3, 3, 0]]])

data2 = np.array([[[11, 22, 33, 44], [22, 44, 55, 33], [0, -1, 22, 33]],
                  [[22, 44, 55, 11], [10, 55, 22, 22], [10, 33, 33, 0]]])

ht = {'CTYPE3': 'HPLT-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.5, 'CRPIX3': 0, 'CRVAL3': 0, 'NAXIS3': 2,
      'CTYPE2': 'WAVE    ', 'CUNIT2': 'Angstrom', 'CDELT2': 0.2, 'CRPIX2': 0, 'CRVAL2': 0,
      'NAXIS2': 3,
      'CTYPE1': 'TIME    ', 'CUNIT1': 'min', 'CDELT1': 0.4, 'CRPIX1': 0, 'CRVAL1': 0, 'NAXIS1': 4}

wt = WCS(header=ht, naxis=3)

cube1 = NDCube(
    data, wt, missing_axis=[False, False, False, True],
    extra_coords=[
        ('pix', 0, u.Quantity(range(data.shape[0]), unit=u.pix)),
        ('distance', None, u.Quantity(0, unit=u.cm)),
        ('time', None, datetime.datetime(2000, 1, 1, 0, 0))])

cube1_with_unit = NDCube(
    data, wt, missing_axis=[False, False, False, True],
    unit=u.km,
    extra_coords=[
        ('pix', 0, u.Quantity(range(data.shape[0]), unit=u.pix)),
        ('distance', None, u.Quantity(0, unit=u.cm)),
        ('time', None, datetime.datetime(2000, 1, 1, 0, 0))])

cube1_with_mask = NDCube(
    data, wt, missing_axis=[False, False, False, True],
    mask=np.zeros_like(data, dtype=bool),
    extra_coords=[
        ('pix', 0, u.Quantity(range(data.shape[0]), unit=u.pix)),
        ('distance', None, u.Quantity(0, unit=u.cm)),
        ('time', None, datetime.datetime(2000, 1, 1, 0, 0))])

cube3 = NDCube(
    data2, wt, missing_axis=[False, False, False, True],
    extra_coords=[
        ('pix', 0, u.Quantity(np.arange(1, data2.shape[0]+1), unit=u.pix) +
         cube1.extra_coords['pix']['value'][-1]),
        ('distance', None, u.Quantity(2, unit=u.cm)),
        ('time', None, datetime.datetime(2000, 1, 1, 0, 2))])

cube3_with_unit = NDCube(
    data2, wt, missing_axis=[False, False, False, True],
    unit=u.m,
    extra_coords=[
        ('pix', 0, u.Quantity(np.arange(1, data2.shape[0]+1), unit=u.pix) +
         cube1.extra_coords['pix']['value'][-1]),
        ('distance', None, u.Quantity(2, unit=u.cm)),
        ('time', None, datetime.datetime(2000, 1, 1, 0, 2))])

cube3_with_mask = NDCube(
    data2, wt, missing_axis=[False, False, False, True],
    mask=np.zeros_like(data2, dtype=bool),
    extra_coords=[
        ('pix', 0, u.Quantity(np.arange(1, data2.shape[0]+1), unit=u.pix) +
         cube1.extra_coords['pix']['value'][-1]),
        ('distance', None, u.Quantity(2, unit=u.cm)),
        ('time', None, datetime.datetime(2000, 1, 1, 0, 2))])

# Define some test NDCubeSequences.
common_axis = 0
seq = NDCubeSequence(data_list=[cube1, cube3, cube1, cube3], common_axis=common_axis)

seq_with_units = NDCubeSequence(
    data_list=[cube1_with_unit, cube3_with_unit, cube1_with_unit, cube3_with_unit],
    common_axis=common_axis)

seq_with_masks = NDCubeSequence(
    data_list=[cube1_with_mask, cube3_with_mask, cube1_with_mask, cube3_with_mask],
    common_axis=common_axis)

seq_with_unit0 = NDCubeSequence(data_list=[cube1_with_unit, cube3,
                                           cube1_with_unit, cube3], common_axis=common_axis)

seq_with_mask0 = NDCubeSequence(data_list=[cube1_with_mask, cube3,
                                           cube1_with_mask, cube3], common_axis=common_axis)

# Derive some expected data arrays in plot objects.
seq_data_stack = np.stack([cube.data for cube in seq_with_masks.data])
seq_mask_stack = np.stack([cube.mask for cube in seq_with_masks.data])

seq_stack = np.ma.masked_array(seq_data_stack, seq_mask_stack)
seq_stack_km = np.ma.masked_array(
    np.stack([(cube.data * cube.unit).to(u.km).value for cube in seq_with_units.data]),
    seq_mask_stack)

seq_data_concat = np.concatenate([cube.data for cube in seq_with_masks.data], axis=common_axis)
seq_mask_concat = np.concatenate([cube.mask for cube in seq_with_masks.data], axis=common_axis)
seq_concat = np.ma.masked_array(seq_data_concat, seq_mask_concat)

# Derive expected axis_ranges
x_axis_coords = np.array([0.4, 0.8, 1.2, 1.6]).reshape((1, 1, 4))
new_x_axis_coords_shape = u.Quantity(seq.dimensions, unit=u.pix).value.astype(int)
new_x_axis_coords_shape[-1] = 1
none_axis_ranges_axis3 = [np.arange(len(seq.data)), np.array([0., 2.]), np.array([0., 1.5, 3.]),
                          np.tile(np.array(x_axis_coords), new_x_axis_coords_shape)]


@pytest.mark.parametrize("test_input, test_kwargs, expected_values", [
    (seq[:, 0, 0, 0], {},
     (np.arange(len(seq.data)), np.array([ 1, 11,  1, 11]),
      "meta.obs.sequence [None]", "Data [None]",
      (0, len(seq[:, 0, 0, 0].data)-1),
      (min([cube.data.min() for cube in seq[:, 0, 0, 0].data]),
       max([cube.data.min() for cube in seq[:, 0, 0, 0].data])))),
    (seq_with_units[:, 0, 0, 0], {},
     (np.arange(len(seq.data)), np.array([ 1, 11,  1, 11]),
      "meta.obs.sequence [None]", "Data [None]",
      (0, len(seq[:, 0, 0, 0].data)-1),
      (min([cube.data.min() for cube in seq[:, 0, 0, 0].data]),
       max([cube.data.min() for cube in seq[:, 0, 0, 0].data]))))
    ])
def test_sequence_plot_1D_plot(test_input, test_kwargs, expected_values):
    # Unpack expected values
    expected_x_data, expected_y_data, expected_x_label, expected_y_label, \
    expected_xlim, expected_ylim = expected_values
    # Run plot method
    output = test_input.plot(**test_kwargs)
    # Check values are correct
    #assert isinstance(output, matplotlib.axes._subplots.AxesSubplot)
    np.testing.assert_array_equal(output.lines[0].get_xdata(), expected_x_data)
    np.testing.assert_array_equal(output.lines[0].get_ydata(), expected_y_data)
    assert output.axes.get_xlabel() == expected_x_label
    assert output.axes.get_ylabel() == expected_y_label
    output_xlim = output.axes.get_xlim()
    assert output_xlim[0] <= expected_xlim[0]
    assert output_xlim[1] >= expected_xlim[1]
    output_ylim = output.axes.get_ylim()
    assert output_ylim[0] <= expected_ylim[0]
    assert output_ylim[1] >= expected_ylim[1]
    

def test_sequence_plot_as_cube_1D_plot():
    pass

"""
@pytest.mark.parametrize("test_input, test_kwargs, expected_values", [
    (seq, {"plot_axis_indices": 3},
     (ndcube.mixins.sequence_plotting.LineAnimatorNDCubeSequence, seq_stack.data,
      none_axis_ranges_axis3, "time [min]", "Data [None]",
      (none_axis_ranges_axis3[-1].min(), none_axis_ranges_axis3[-1].max()),
      (seq_stack.data.min(), seq_stack.data.max()))),
    (seq, {"plot_axis_indices": -1, "data_unit": u.km},
     (ndcube.mixins.sequence_plotting.LineAnimatorNDCubeSequence, seq_stack_km.data,
      none_axis_ranges_axis3, "time [min]", "Data [None]",
      (none_axis_ranges_axis3[-1].min(), none_axis_ranges_axis3[-1].max()),
    (seq_stack.data.min(), seq_stack.data.max()))),
    (seq, {"plot_axis_indices": -1},
     (ndcube.mixins.sequence_plotting.LineAnimatorNDCubeSequence, seq_stack,
      none_axis_ranges_axis3, "time [min]", "Data [None]",
      (none_axis_ranges_axis3[-1].min(), none_axis_ranges_axis3[-1].max()),
    (seq_stack.data.min(), seq_stack.data.max())))])
def test_sequence_plot_LineAnimator(test_input, test_kwargs, expected_values):
    # Unpack expected values
    expected_type, expected_data, expected_axis_ranges, expected_xlabel, \
      expected_ylabel, expected_xlim, expected_ylim = expected_values
    # Run plot method.
    output = seq.plot(**test_kwargs)
    # Check right type of plot object is produced.
    assert type(output) is expected_type
    # Check data being plotted is correct
    np.testing.assert_array_equal(output.data, expected_data)
    if type(expected_data) is np.ma.core.MaskedArray:
        np.testing.assert_array_equal(output.data.mask, expected_data.mask)
    # Check values of axes and sliders is correct.
    for i in range(len(output.axis_ranges)):
        print(i)
        np.testing.assert_array_equal(output.axis_ranges[i], expected_axis_ranges[i])
    # Check plot axis labels and limits are correct
    assert output.xlabel == expected_xlabel
    assert output.ylabel == expected_ylabel
    assert output.xlim == expected_xlim
    assert output.ylim == expected_ylim
"""

def test_sequence_plot_as_cube_LineAnimator():
    pass


def test_sequence_plot_2D_image():
    #p.images[0].get_extent() # xlim and ylim
    #p.images[0].get_array() # data
    #p.xaxis.get_label_text()
    #p.yaxis.get_label_text()
    pass


def test_sequence_as_cube_plot_2D_image():
    #p.images[0].get_extent() # xlim and ylim
    #p.images[0].get_array() # data
    #p.xaxis.get_label_text()
    #p.yaxis.get_label_text()
    pass


def test_sequence_plot_ImageAnimator():
    #p.data
    #p.axis_ranges
    #p.axes.get_xaxis().get_label_text()
    #p.axes.get_yaxis().get_label_text()
    pass


def test_sequence_plot_as_cube_ImageAnimator():
    pass


@pytest.mark.parametrize("test_input, expected", [
    ((seq_with_unit0.data, None), (None, None)),
    ((seq_with_unit0.data, u.km), (None, None)),
    ((seq_with_units.data, None), ([u.km, u.m, u.km, u.m], u.km)),
    ((seq_with_units.data, u.cm), ([u.km, u.m, u.km, u.m], u.cm))])
def test_determine_sequence_units(test_input, expected):
    output_seq_unit, output_unit = ndcube.mixins.sequence_plotting._determine_sequence_units(
        test_input[0], unit=test_input[1])
    assert output_seq_unit == expected[0]
    assert output_unit == expected[1]


@pytest.mark.parametrize("test_input, expected", [
    ((3, 1, "time", u.s), ([1], [None, 'time', None], [None, u.s, None])),
    ((3, None, None, None), ([-1, -2], None, None))])
def test_prep_axes_kwargs(test_input, expected):
    output = ndcube.mixins.sequence_plotting._prep_axes_kwargs(*test_input)
    for i in range(3):
        assert output[i] == expected[i]


@pytest.mark.parametrize("test_input, expected_error", [
    ((3, [0, 1, 2], ["time", "pix"], u.s), ValueError),
    ((3, 0, ["time", "pix"], u.s), ValueError),
    ((3, 0, "time", [u.s, u.pix]), ValueError),
    ((3, 0, 0, u.s), TypeError),
    ((3, 0, "time", 0), TypeError)])
def test_prep_axes_kwargs_errors(test_input, expected_error):
    with pytest.raises(expected_error):
        output = ndcube.mixins.sequence_plotting._prep_axes_kwargs(*test_input)
