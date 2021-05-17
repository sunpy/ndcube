# import pytest
# import datetime
# import copy

# import numpy as np
# import astropy.units as u
# import matplotlib

# from ndcube import NDCube, NDCubeSequence
# from ndcube.utils.wcs import WCS
# import ndcube.mixins.sequence_plotting

# # sample data for tests
# # TODO: use a fixture reading from a test file. file TBD.
# data = np.array([[[1, 2, 3, 4], [2, 4, 5, 3], [0, -1, 2, 3]],
#                  [[2, 4, 5, 1], [10, 5, 2, 2], [10, 3, 3, 0]]])

# data2 = np.array([[[11, 22, 33, 44], [22, 44, 55, 33], [0, -1, 22, 33]],
#                   [[22, 44, 55, 11], [10, 55, 22, 22], [10, 33, 33, 0]]])

# ht = {'CTYPE3': 'HPLT-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.5, 'CRPIX3': 0, 'CRVAL3': 0, 'NAXIS3': 2,
#       'CTYPE2': 'WAVE    ', 'CUNIT2': 'Angstrom', 'CDELT2': 0.2, 'CRPIX2': 0, 'CRVAL2': 0,
#       'NAXIS2': 3,
#       'CTYPE1': 'TIME    ', 'CUNIT1': 'min', 'CDELT1': 0.4, 'CRPIX1': 0, 'CRVAL1': 0, 'NAXIS1': 4}
# wt = WCS(header=ht, naxis=3)

# hm = {
#     'CTYPE1': 'WAVE    ', 'CUNIT1': 'Angstrom', 'CDELT1': 0.2, 'CRPIX1': 0, 'CRVAL1': 10,
#     'NAXIS1': 4,
#     'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5, 'NAXIS2': 3,
#     'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 2}
# wm = WCS(header=hm, naxis=3)


# cube1 = NDCube(
#     data, wt, missing_axes=[False, False, False, True],
#     extra_coords=[
#         ('pix', 0, u.Quantity(range(data.shape[0]), unit=u.pix)),
#         ('hi', 1, u.Quantity(range(data.shape[1]), unit=u.s)),
#         ('distance', None, u.Quantity(0, unit=u.cm)),
#         ('time', None, datetime.datetime(2000, 1, 1, 0, 0))])

# cube1_with_unit = NDCube(
#     data, wt, missing_axes=[False, False, False, True],
#     unit=u.km,
#     extra_coords=[
#         ('pix', 0, u.Quantity(range(data.shape[0]), unit=u.pix)),
#         ('hi', 1, u.Quantity(range(data.shape[1]), unit=u.s)),
#         ('distance', None, u.Quantity(0, unit=u.cm)),
#         ('time', None, datetime.datetime(2000, 1, 1, 0, 0))])

# cube1_with_mask = NDCube(
#     data, wt, missing_axes=[False, False, False, True],
#     mask=np.zeros_like(data, dtype=bool),
#     extra_coords=[
#         ('pix', 0, u.Quantity(range(data.shape[0]), unit=u.pix)),
#         ('hi', 1, u.Quantity(range(data.shape[1]), unit=u.s)),
#         ('distance', None, u.Quantity(0, unit=u.cm)),
#         ('time', None, datetime.datetime(2000, 1, 1, 0, 0))])

# cube1_with_uncertainty = NDCube(
#     data, wt, missing_axes=[False, False, False, True],
#     uncertainty=np.sqrt(data),
#     extra_coords=[
#         ('pix', 0, u.Quantity(range(data.shape[0]), unit=u.pix)),
#         ('hi', 1, u.Quantity(range(data.shape[1]), unit=u.s)),
#         ('distance', None, u.Quantity(0, unit=u.cm)),
#         ('time', None, datetime.datetime(2000, 1, 1, 0, 0))])

# cube1_with_unit_and_uncertainty = NDCube(
#     data, wt, missing_axes=[False, False, False, True],
#     unit=u.km, uncertainty=np.sqrt(data),
#     extra_coords=[
#         ('pix', 0, u.Quantity(range(data.shape[0]), unit=u.pix)),
#         ('hi', 1, u.Quantity(range(data.shape[1]), unit=u.s)),
#         ('distance', None, u.Quantity(0, unit=u.cm)),
#         ('time', None, datetime.datetime(2000, 1, 1, 0, 0))])

# cube3 = NDCube(
#     data2, wt, missing_axes=[False, False, False, True],
#     extra_coords=[
#         ('pix', 0, u.Quantity(np.arange(1, data2.shape[0]+1), unit=u.pix) +
#          cube1.extra_coords['pix']['value'][-1]),
#         ('hi', 1, u.Quantity(range(data2.shape[1]), unit=u.s)),
#         ('distance', None, u.Quantity(2, unit=u.cm)),
#         ('time', None, datetime.datetime(2000, 1, 1, 0, 2))])

# cube3_with_unit = NDCube(
#     data2, wt, missing_axes=[False, False, False, True],
#     unit=u.m,
#     extra_coords=[
#         ('pix', 0, u.Quantity(np.arange(1, data2.shape[0]+1), unit=u.pix) +
#          cube1.extra_coords['pix']['value'][-1]),
#         ('hi', 1, u.Quantity(range(data2.shape[1]), unit=u.s)),
#         ('distance', None, u.Quantity(2, unit=u.cm)),
#         ('time', None, datetime.datetime(2000, 1, 1, 0, 2))])

# cube3_with_mask = NDCube(
#     data2, wt, missing_axes=[False, False, False, True],
#     mask=np.zeros_like(data2, dtype=bool),
#     extra_coords=[
#         ('pix', 0, u.Quantity(np.arange(1, data2.shape[0]+1), unit=u.pix) +
#          cube1.extra_coords['pix']['value'][-1]),
#         ('hi', 1, u.Quantity(range(data2.shape[1]), unit=u.s)),
#         ('distance', None, u.Quantity(2, unit=u.cm)),
#         ('time', None, datetime.datetime(2000, 1, 1, 0, 2))])

# cube3_with_uncertainty = NDCube(
#     data2, wt, missing_axes=[False, False, False, True],
#     uncertainty=np.sqrt(data2),
#     extra_coords=[
#         ('pix', 0, u.Quantity(np.arange(1, data2.shape[0]+1), unit=u.pix) +
#          cube1.extra_coords['pix']['value'][-1]),
#         ('hi', 1, u.Quantity(range(data2.shape[1]), unit=u.s)),
#         ('distance', None, u.Quantity(2, unit=u.cm)),
#         ('time', None, datetime.datetime(2000, 1, 1, 0, 2))])

# cube3_with_unit_and_uncertainty = NDCube(
#     data2, wt, missing_axes=[False, False, False, True],
#     unit=u.m, uncertainty=np.sqrt(data2),
#     extra_coords=[
#         ('pix', 0, u.Quantity(np.arange(1, data2.shape[0]+1), unit=u.pix) +
#          cube1.extra_coords['pix']['value'][-1]),
#         ('hi', 1, u.Quantity(range(data2.shape[1]), unit=u.s)),
#         ('distance', None, u.Quantity(2, unit=u.cm)),
#         ('time', None, datetime.datetime(2000, 1, 1, 0, 2))])

# cubem1 = NDCube(
#     data, wm,
#     extra_coords=[
#         ('pix', 0, u.Quantity(range(data.shape[0]), unit=u.pix)),
#         ('hi', 1, u.Quantity(range(data.shape[1]), unit=u.s)),
#         ('distance', None, u.Quantity(0, unit=u.cm)),
#         ('time', None, datetime.datetime(2000, 1, 1, 0, 0))])

# cubem3 = NDCube(
#     data2, wm,
#     extra_coords=[
#         ('pix', 0, u.Quantity(range(data.shape[0]), unit=u.pix)),
#         ('hi', 1, u.Quantity(range(data.shape[1]), unit=u.s)),
#         ('distance', None, u.Quantity(0, unit=u.cm)),
#         ('time', None, datetime.datetime(2000, 1, 1, 0, 0))])

# # Define some test NDCubeSequences.
# common_axis = 0
# seq = NDCubeSequence(data_list=[cube1, cube3, cube1, cube3], common_axis=common_axis)

# seq_no_common_axis = NDCubeSequence(data_list=[cube1, cube3, cube1, cube3])

# seq_with_units = NDCubeSequence(
#     data_list=[cube1_with_unit, cube3_with_unit, cube1_with_unit, cube3_with_unit],
#     common_axis=common_axis)

# seq_with_masks = NDCubeSequence(
#     data_list=[cube1_with_mask, cube3_with_mask, cube1_with_mask, cube3_with_mask],
#     common_axis=common_axis)

# seq_with_unit0 = NDCubeSequence(data_list=[cube1_with_unit, cube3,
#                                            cube1_with_unit, cube3], common_axis=common_axis)

# seq_with_mask0 = NDCubeSequence(data_list=[cube1_with_mask, cube3,
#                                            cube1_with_mask, cube3], common_axis=common_axis)

# seq_with_uncertainty = NDCubeSequence(data_list=[cube1_with_uncertainty, cube3_with_uncertainty,
#                                                  cube1_with_uncertainty, cube3_with_uncertainty],
#                                       common_axis=common_axis)

# seq_with_some_uncertainty = NDCubeSequence(
#     data_list=[cube1_with_uncertainty, cube3, cube1, cube3_with_uncertainty],
#     common_axis=common_axis)

# seq_with_units_and_uncertainty = NDCubeSequence(
#     data_list=[cube1_with_unit_and_uncertainty, cube3_with_unit_and_uncertainty,
#                cube1_with_unit_and_uncertainty, cube3_with_unit_and_uncertainty],
#     common_axis=common_axis)

# seq_with_units_and_some_uncertainty = NDCubeSequence(
#     data_list=[cube1_with_unit_and_uncertainty, cube3_with_unit,
#                cube1_with_unit, cube3_with_unit_and_uncertainty],
#     common_axis=common_axis)

# seq_with_some_masks = NDCubeSequence(data_list=[cube1_with_mask, cube3, cube1, cube3_with_mask],
#                                      common_axis=common_axis)

# seqm = NDCubeSequence(data_list=[cubem1, cubem3, cubem1, cubem3], common_axis=common_axis)

# # Derive some expected data arrays in plot objects.
# seq_data_stack = np.stack([cube.data for cube in seq_with_masks.data])
# seq_mask_stack = np.stack([cube.mask for cube in seq_with_masks.data])

# seq_stack = np.ma.masked_array(seq_data_stack, seq_mask_stack)
# seq_stack_km = np.ma.masked_array(
#     np.stack([(cube.data * cube.unit).to(u.km).value for cube in seq_with_units.data]),
#     seq_mask_stack)

# seq_data_concat = np.concatenate([cube.data for cube in seq_with_masks.data], axis=common_axis)
# seq_mask_concat = np.concatenate([cube.mask for cube in seq_with_masks.data], axis=common_axis)

# seq_concat = np.ma.masked_array(seq_data_concat, seq_mask_concat)
# seq_concat_km = np.ma.masked_array(
#     np.concatenate([(cube.data * cube.unit).to(u.km).value
#                     for cube in seq_with_units.data], axis=common_axis),
#     seq_mask_concat)

# # Derive expected axis_ranges for non-cube-like cases.
# x_axis_coords3 = np.array([0.4, 0.8, 1.2, 1.6]).reshape((1, 1, 4))
# new_x_axis_coords3_shape = u.Quantity(seq.dimensions, unit=u.pix).value.astype(int)
# new_x_axis_coords3_shape[-1] = 1
# none_axis_ranges_axis3 = [np.arange(0, len(seq.data)+1),
#                           np.array([0., 1., 2.]), np.arange(0, 4),
#                           np.tile(np.array(x_axis_coords3), new_x_axis_coords3_shape)]
# none_axis_ranges_axis0 = [np.arange(len(seq.data)),
#                           np.array([0., 1., 2.]), np.arange(0, 4),
#                           np.arange(0, int(seq.dimensions[-1].value)+1)]
# distance0_none_axis_ranges_axis0 = [seq.sequence_axis_extra_coords["distance"].value,
#                                     np.array([0., 1., 2.]), np.arange(0, 4),
#                                     np.arange(0, int(seq.dimensions[-1].value)+1)]
# distance0_none_axis_ranges_axis0_mm = [seq.sequence_axis_extra_coords["distance"].to("mm").value,
#                                        np.array([0., 1., 2.]), np.arange(0, 4),
#                                        np.arange(0, int(seq.dimensions[-1].value)+1)]
# userrangequantity_none_axis_ranges_axis0 = [
#     np.arange(int(seq.dimensions[0].value)), np.array([0., 1., 2.]), np.arange(0, 4),
#     np.arange(0, int(seq.dimensions[-1].value)+1)]

# userrangequantity_none_axis_ranges_axis0_1e7 = [
#     (np.arange(int(seq.dimensions[0].value)) * u.J).to(u.erg).value, np.array([0., 1., 2.]),
#     np.arange(0, 4), np.arange(0, int(seq.dimensions[-1].value)+1)]

# hi2_none_axis_ranges_axis2 = [
#     np.arange(0, len(seq.data)+1), np.array([0., 1., 2.]),
#     np.arange(int(seq.dimensions[2].value)), np.arange(0, int(seq.dimensions[-1].value)+1)]

# x_axis_coords1 = np.zeros(tuple([int(s.value) for s in seq.dimensions]))
# x_axis_coords1[0, 1] = 1.
# x_axis_coords1[1, 0] = 2.
# x_axis_coords1[1, 1] = 3.
# x_axis_coords1[2, 1] = 1.
# x_axis_coords1[3, 0] = 2.
# x_axis_coords1[3, 1] = 3.
# pix1_none_axis_ranges_axis1 = [
#     np.arange(0, len(seq.data)+1), x_axis_coords1, np.arange(0, 4),
#     np.arange(0, int(seq.dimensions[-1].value)+1)]

# # Derive expected extents
# seq_axis1_lim_deg = [0.49998731, 0.99989848]
# seq_axis1_lim_arcsec = [(axis1_xlim*u.deg).to(u.arcsec).value for axis1_xlim in seq_axis1_lim_deg]
# seq_axis2_lim_m = [seq[:, :, :, 0].data[0].axis_world_coords()[-1][0].value,
#                    seq[:, :, :, 0].data[0].axis_world_coords()[-1][-1].value]

# # Derive expected axis_ranges for cube-like cases.
# cube_like_new_x_axis_coords2_shape = u.Quantity(
#     seq.cube_like_dimensions, unit=u.pix).value.astype(int)
# cube_like_new_x_axis_coords2_shape[-1] = 1
# cubelike_none_axis_ranges_axis2 = [
#     np.arange(0, int(seq.cube_like_dimensions[0].value)+1), np.arange(0, 4),
#     np.tile(x_axis_coords3, cube_like_new_x_axis_coords2_shape)]

# cubelike_none_axis_ranges_axis2_s = copy.deepcopy(cubelike_none_axis_ranges_axis2)
# cubelike_none_axis_ranges_axis2_s[2] = cubelike_none_axis_ranges_axis2_s[2] * 60.

# cubelike_none_axis_ranges_axis0 = [[0, 8], np.arange(0, 4),
#                                    np.arange(0, int(seq.cube_like_dimensions[-1].value)+1)]


# @pytest.mark.parametrize("test_input, test_kwargs, expected_values", [
#     (seq[:, 0, 0, 0], {},
#      (np.arange(len(seq.data)), np.array([1, 11,  1, 11]),
#       "meta.obs.sequence [None]", "Data [None]", (0, len(seq[:, 0, 0, 0].data)-1),
#       (min([cube.data.min() for cube in seq[:, 0, 0, 0].data]),
#        max([cube.data.max() for cube in seq[:, 0, 0, 0].data])))),

#     (seq_with_units[:, 0, 0, 0], {},
#      (np.arange(len(seq_with_units.data)), np.array([1, 0.011,  1, 0.011]),
#       "meta.obs.sequence [None]", "Data [km]", (0, len(seq_with_units[:, 0, 0, 0].data)-1),
#       (min([(cube.data * cube.unit).to(seq_with_units[:, 0, 0, 0].data[0].unit).value
#             for cube in seq_with_units[:, 0, 0, 0].data]),
#        max([(cube.data * cube.unit).to(seq_with_units[:, 0, 0, 0].data[0].unit).value
#             for cube in seq_with_units[:, 0, 0, 0].data])))),

#     (seq_with_uncertainty[:, 0, 0, 0], {},
#      (np.arange(len(seq_with_uncertainty.data)), np.array([1, 11,  1, 11]),
#       "meta.obs.sequence [None]", "Data [None]", (0, len(seq_with_uncertainty[:, 0, 0, 0].data)-1),
#       (min([cube.data for cube in seq_with_uncertainty[:, 0, 0, 0].data]),
#        max([cube.data for cube in seq_with_uncertainty[:, 0, 0, 0].data])))),

#     (seq_with_units_and_uncertainty[:, 0, 0, 0], {},
#      (np.arange(len(seq_with_units_and_uncertainty.data)), np.array([1, 0.011,  1, 0.011]),
#       "meta.obs.sequence [None]", "Data [km]",
#       (0, len(seq_with_units_and_uncertainty[:, 0, 0, 0].data)-1),
#       (min([(cube.data*cube.unit).to(seq_with_units_and_uncertainty[:, 0, 0, 0].data[0].unit).value
#             for cube in seq_with_units_and_uncertainty[:, 0, 0, 0].data]),
#        max([(cube.data*cube.unit).to(seq_with_units_and_uncertainty[:, 0, 0, 0].data[0].unit).value
#             for cube in seq_with_units_and_uncertainty[:, 0, 0, 0].data])))),

#     (seq_with_units_and_some_uncertainty[:, 0, 0, 0], {},
#      (np.arange(len(seq_with_units_and_some_uncertainty.data)), np.array([1, 0.011,  1, 0.011]),
#       "meta.obs.sequence [None]", "Data [km]",
#       (0, len(seq_with_units_and_some_uncertainty[:, 0, 0, 0].data)-1),
#       (min([(cube.data*cube.unit).to(
#           seq_with_units_and_some_uncertainty[:, 0, 0, 0].data[0].unit).value
#           for cube in seq_with_units_and_some_uncertainty[:, 0, 0, 0].data]),
#        max([(cube.data*cube.unit).to(
#            seq_with_units_and_some_uncertainty[:, 0, 0, 0].data[0].unit).value
#            for cube in seq_with_units_and_some_uncertainty[:, 0, 0, 0].data])))),

#     (seq[:, 0, 0, 0], {"axes_coordinates": "distance"},
#      ((seq.sequence_axis_extra_coords["distance"]), np.array([1, 11,  1, 11]),
#       "distance [{0}]".format(seq.sequence_axis_extra_coords["distance"].unit), "Data [None]",
#      (min(seq.sequence_axis_extra_coords["distance"].value),
#       max(seq.sequence_axis_extra_coords["distance"].value)),
#      (min([cube.data.min() for cube in seq[:, 0, 0, 0].data]),
#       max([cube.data.max() for cube in seq[:, 0, 0, 0].data])))),

#     (seq[:, 0, 0, 0], {"axes_coordinates": u.Quantity(np.arange(len(seq.data)), unit=u.cm),
#                        "axes_units": u.km},
#      (u.Quantity(np.arange(len(seq.data)), unit=u.cm).to(u.km), np.array([1, 11,  1, 11]),
#       "meta.obs.sequence [km]", "Data [None]",
#      (min((u.Quantity(np.arange(len(seq.data)), unit=u.cm).to(u.km).value)),
#       max((u.Quantity(np.arange(len(seq.data)), unit=u.cm).to(u.km).value))),
#      (min([cube.data.min() for cube in seq[:, 0, 0, 0].data]),
#       max([cube.data.max() for cube in seq[:, 0, 0, 0].data]))))
#     ])
# def test_sequence_plot_1D_plot(test_input, test_kwargs, expected_values):
#     # Unpack expected values
#     expected_x_data, expected_y_data, expected_xlabel, expected_ylabel, \
#       expected_xlim, expected_ylim = expected_values
#     # Run plot method
#     output = test_input.plot(**test_kwargs)
#     # Check values are correct
#     assert isinstance(output, matplotlib.axes.Axes)
#     np.testing.assert_array_equal(output.lines[0].get_xdata(), expected_x_data)
#     np.testing.assert_array_equal(output.lines[0].get_ydata(), expected_y_data)
#     assert output.axes.get_xlabel() == expected_xlabel
#     assert output.axes.get_ylabel() == expected_ylabel
#     output_xlim = output.axes.get_xlim()
#     assert output_xlim[0] <= expected_xlim[0]
#     assert output_xlim[1] >= expected_xlim[1]
#     output_ylim = output.axes.get_ylim()
#     assert output_ylim[0] <= expected_ylim[0]
#     assert output_ylim[1] >= expected_ylim[1]


# @pytest.mark.parametrize("test_input, test_kwargs, expected_values", [
#     (seq[:, :, 0, 0], {},
#      (np.array([0.49998731, 0.99989848, 0.49998731, 0.99989848,
#                 0.49998731, 0.99989848, 0.49998731, 0.99989848]),
#       np.array([1, 2, 11, 22,  1, 2, 11, 22]),
#       "{0} [{1}]".format(seq[:, :, 0, 0].cube_like_world_axis_physical_types[common_axis], "deg"),
#       "Data [None]", tuple(seq_axis1_lim_deg),
#       (min([cube.data.min() for cube in seq[:, :, 0, 0].data]),
#        max([cube.data.max() for cube in seq[:, :, 0, 0].data])))),

#     (seq_with_units[:, :, 0, 0], {},
#      (np.array([0.49998731, 0.99989848, 0.49998731, 0.99989848,
#                 0.49998731, 0.99989848, 0.49998731, 0.99989848]),
#       np.array([1, 2, 0.011, 0.022,  1, 2, 0.011, 0.022]),
#       "{0} [{1}]".format(seq[:, :, 0, 0].cube_like_world_axis_physical_types[common_axis], "deg"),
#       "Data [km]", tuple(seq_axis1_lim_deg),
#       (min([min((cube.data * cube.unit).to(u.km).value)
#             for cube in seq_with_units[:, :, 0, 0].data]),
#        max([max((cube.data * cube.unit).to(u.km).value)
#             for cube in seq_with_units[:, :, 0, 0].data])))),

#     (seq_with_uncertainty[:, :, 0, 0], {},
#      (np.array([0.49998731, 0.99989848, 0.49998731, 0.99989848,
#                 0.49998731, 0.99989848, 0.49998731, 0.99989848]),
#       np.array([1, 2, 11, 22,  1, 2, 11, 22]),
#       "{0} [{1}]".format(
#           seq_with_uncertainty[:, :, 0, 0].cube_like_world_axis_physical_types[
#               common_axis], "deg"),
#       "Data [None]", tuple(seq_axis1_lim_deg),
#       (min([cube.data.min() for cube in seq_with_uncertainty[:, :, 0, 0].data]),
#        max([cube.data.max() for cube in seq_with_uncertainty[:, :, 0, 0].data])))),

#     (seq_with_some_uncertainty[:, :, 0, 0], {},
#      (np.array([0.49998731, 0.99989848, 0.49998731, 0.99989848,
#                 0.49998731, 0.99989848, 0.49998731, 0.99989848]),
#       np.array([1, 2, 11, 22,  1, 2, 11, 22]),
#       "{0} [{1}]".format(
#           seq_with_some_uncertainty[:, :, 0, 0].cube_like_world_axis_physical_types[
#               common_axis], "deg"),
#       "Data [None]", tuple(seq_axis1_lim_deg),
#       (min([cube.data.min() for cube in seq_with_some_uncertainty[:, :, 0, 0].data]),
#        max([cube.data.max() for cube in seq_with_some_uncertainty[:, :, 0, 0].data])))),

#     (seq_with_units_and_uncertainty[:, :, 0, 0], {},
#      (np.array([0.49998731, 0.99989848, 0.49998731, 0.99989848,
#                 0.49998731, 0.99989848, 0.49998731, 0.99989848]),
#       np.array([1, 2, 0.011, 0.022,  1, 2, 0.011, 0.022]),
#       "{0} [{1}]".format(
#           seq_with_units_and_uncertainty[:, :, 0, 0].cube_like_world_axis_physical_types[
#               common_axis], "deg"),
#       "Data [km]", tuple(seq_axis1_lim_deg),
#       (min([min((cube.data * cube.unit).to(u.km).value)
#             for cube in seq_with_units[:, :, 0, 0].data]),
#        max([max((cube.data * cube.unit).to(u.km).value)
#             for cube in seq_with_units[:, :, 0, 0].data])))),

#     (seq_with_units_and_some_uncertainty[:, :, 0, 0], {},
#      (np.array([0.49998731, 0.99989848, 0.49998731, 0.99989848,
#                 0.49998731, 0.99989848, 0.49998731, 0.99989848]),
#       np.array([1, 2, 0.011, 0.022,  1, 2, 0.011, 0.022]),
#       "{0} [{1}]".format(
#           seq_with_units_and_some_uncertainty[:, :, 0, 0].cube_like_world_axis_physical_types[
#               common_axis], "deg"),
#       "Data [km]", tuple(seq_axis1_lim_deg),
#       (min([min((cube.data * cube.unit).to(u.km).value)
#             for cube in seq_with_units[:, :, 0, 0].data]),
#        max([max((cube.data * cube.unit).to(u.km).value)
#             for cube in seq_with_units[:, :, 0, 0].data])))),

#     (seq[:, :, 0, 0], {"axes_coordinates": "pix"},
#      (seq[:, :, 0, 0].common_axis_extra_coords["pix"].value,
#       np.array([1, 2, 11, 22,  1, 2, 11, 22]), "pix [pix]", "Data [None]",
#       (min(seq[:, :, 0, 0].common_axis_extra_coords["pix"].value),
#        max(seq[:, :, 0, 0].common_axis_extra_coords["pix"].value)),
#       (min([cube.data.min() for cube in seq[:, :, 0, 0].data]),
#        max([cube.data.max() for cube in seq[:, :, 0, 0].data])))),

#     (seq[:, :, 0, 0],
#      {"axes_coordinates": np.arange(10, 10+seq[:, :, 0, 0].cube_like_dimensions[0].value)},
#      (np.arange(10, 10 + seq[:, :, 0, 0].cube_like_dimensions[0].value),
#       np.array([1, 2, 11, 22,  1, 2, 11, 22]),
#       "{0} [{1}]".format("", None), "Data [None]",
#       (10, 10 + seq[:, :, 0, 0].cube_like_dimensions[0].value - 1),
#       (min([cube.data.min() for cube in seq[:, :, 0, 0].data]),
#        max([cube.data.max() for cube in seq[:, :, 0, 0].data]))))
#     ])
# def test_sequence_plot_as_cube_1D_plot(test_input, test_kwargs, expected_values):
#     # Unpack expected values
#     expected_x_data, expected_y_data, expected_xlabel, expected_ylabel, \
#       expected_xlim, expected_ylim = expected_values
#     # Run plot method
#     output = test_input.plot_as_cube(**test_kwargs)
#     # Check values are correct
#     # Check type of ouput plot object
#     assert isinstance(output, matplotlib.axes.Axes)
#     # Check x and y data are correct.
#     assert np.allclose(output.lines[0].get_xdata(), expected_x_data)
#     assert np.allclose(output.lines[0].get_ydata(), expected_y_data)
#     # Check x and y axis labels are correct.
#     assert output.axes.get_xlabel() == expected_xlabel
#     assert output.axes.get_ylabel() == expected_ylabel
#     # Check all data is contained within x and y axes limits.
#     output_xlim = output.axes.get_xlim()
#     assert output_xlim[0] <= expected_xlim[0]
#     assert output_xlim[1] >= expected_xlim[1]
#     output_ylim = output.axes.get_ylim()
#     assert output_ylim[0] <= expected_ylim[0]
#     assert output_ylim[1] >= expected_ylim[1]


# def test_sequence_plot_as_cube_error():
#     with pytest.raises(TypeError):
#         seq_no_common_axis.plot_as_cube()


# @pytest.mark.parametrize("test_input, test_kwargs, expected_values", [
#     (seq[:, :, 0, 0], {},
#      (seq_stack[:, :, 0, 0],
#       "custom:pos.helioprojective.lat [deg]", "meta.obs.sequence [None]",
#       tuple(seq_axis1_lim_deg + [0, len(seq.data)-1]))),

#     (seq_with_units[:, :, 0, 0], {},
#      (seq_stack_km[:, :, 0, 0],
#       "custom:pos.helioprojective.lat [deg]", "meta.obs.sequence [None]",
#       tuple(seq_axis1_lim_deg + [0, len(seq.data)-1]))),

#     (seq[:, :, 0, 0], {"plot_axis_indices": [0, 1]},
#      (seq_stack[:, :, 0, 0].transpose(),
#       "meta.obs.sequence [None]", "custom:pos.helioprojective.lat [deg]",
#       tuple([0, len(seq.data)-1] + seq_axis1_lim_deg))),

#     (seq[:, :, 0, 0], {"axes_coordinates": ["pix", "distance"]},
#      (seq_stack[:, :, 0, 0],
#       "pix [pix]", "distance [cm]",
#       (min(seq[0, :, 0, 0].extra_coords["pix"]["value"].value),
#        max(seq[0, :, 0, 0].extra_coords["pix"]["value"].value),
#        min(seq[:, :, 0, 0].sequence_axis_extra_coords["distance"].value),
#        max(seq[:, :, 0, 0].sequence_axis_extra_coords["distance"].value)))),
#     # This example shows weakness of current extra coord axis values on 2D plotting!
#     # Only the coordinates from the first cube are shown.

#     (seq[:, :, 0, 0], {"axes_coordinates": [np.arange(
#         10, 10+seq[:, :, 0, 0].dimensions[-1].value), "distance"], "axes_units": [None, u.m]},
#      (seq_stack[:, :, 0, 0],
#       " [None]", "distance [m]",
#       (10, 10+seq[:, :, 0, 0].dimensions[-1].value-1,
#        min(seq[:, :, 0, 0].sequence_axis_extra_coords["distance"].to(u.m).value),
#        max(seq[:, :, 0, 0].sequence_axis_extra_coords["distance"].to(u.m).value)))),

#     (seq[:, :, 0, 0], {"axes_coordinates": [np.arange(
#         10, 10+seq[:, :, 0, 0].dimensions[-1].value)*u.deg, None], "axes_units": [u.arcsec, None]},
#      (seq_stack[:, :, 0, 0],
#       " [arcsec]", "meta.obs.sequence [None]",
#       tuple(list(
#           (np.arange(10, 10+seq[:, :, 0, 0].dimensions[-1].value)*u.deg).to(u.arcsec).value) \
#           + [0, len(seq.data)-1])))
#        ])
# def test_sequence_plot_2D_image(test_input, test_kwargs, expected_values):
#     # Unpack expected values
#     expected_data, expected_xlabel, expected_ylabel, expected_extent = expected_values
#     # Run plot method
#     output = test_input.plot(**test_kwargs)
#     # Check values are correct
#     assert isinstance(output, matplotlib.axes.Axes)
#     np.testing.assert_array_equal(output.images[0].get_array(), expected_data)
#     assert output.xaxis.get_label_text() == expected_xlabel
#     assert output.yaxis.get_label_text() == expected_ylabel
#     assert np.allclose(output.images[0].get_extent(), expected_extent, rtol=1e-3)
#     # Also check x and y values?????


# @pytest.mark.parametrize("test_input, test_kwargs, expected_error", [
#     (seq[:, :, 0, 0], {"axes_coordinates": [
#         np.arange(10, 10+seq[:, :, 0, 0].dimensions[-1].value), None],
#         "axes_units": [u.m, None]}, ValueError),

#     (seq[:, :, 0, 0], {"axes_coordinates": [
#         None, np.arange(10, 10+seq[:, :, 0, 0].dimensions[0].value)],
#         "axes_units": [None, u.m]}, ValueError)
#     ])
# def test_sequence_plot_2D_image_errors(test_input, test_kwargs, expected_error):
#     with pytest.raises(expected_error):
#         output = test_input.plot(**test_kwargs)


# @pytest.mark.parametrize("test_input, test_kwargs, expected_values", [
#     (seq[:, :, :, 0], {},
#      (seq_concat[:, :, 0],
#       "em.wl [m]", "custom:pos.helioprojective.lat [deg]",
#       tuple(seq_axis2_lim_m + seq_axis1_lim_deg))),

#     (seq_with_units[:, :, :, 0], {},
#      (seq_concat_km[:, :, 0],
#       "em.wl [m]", "custom:pos.helioprojective.lat [deg]",
#       tuple(seq_axis2_lim_m + seq_axis1_lim_deg))),

#     (seq[:, :, :, 0], {"plot_axis_indices": [0, 1],
#                        "axes_coordinates": ["pix", "hi"]},
#      (seq_concat[:, :, 0].transpose(), "pix [pix]", "hi [s]",
#       ((seq[:, :, :, 0].common_axis_extra_coords["pix"][0].value,
#         seq[:, :, :, 0].common_axis_extra_coords["pix"][-1].value,
#         seq[:, :, :, 0].data[0].extra_coords["hi"]["value"][0].value,
#         seq[:, :, :, 0].data[0].extra_coords["hi"]["value"][-1].value)))),

#     (seq[:, :, :, 0], {"axes_coordinates": [
#         np.arange(10, 10+seq[:, :, :, 0].cube_like_dimensions[-1].value) * u.m,
#         np.arange(10, 10+seq[:, :, :, 0].cube_like_dimensions[0].value) * u.m]},
#      (seq_concat[:, :, 0], " [m]", " [m]",
#       (10, 10+seq[:, :, :, 0].cube_like_dimensions[-1].value-1,
#        10, 10+seq[:, :, :, 0].cube_like_dimensions[0].value-1))),

#     (seq[:, :, :, 0], {"axes_coordinates": [
#         np.arange(10, 10+seq[:, :, :, 0].cube_like_dimensions[-1].value) * u.m,
#         np.arange(10, 10+seq[:, :, :, 0].cube_like_dimensions[0].value) * u.m],
#         "axes_units": ["cm", u.cm]},
#      (seq_concat[:, :, 0], " [cm]", " [cm]",
#       (10*100, (10+seq[:, :, :, 0].cube_like_dimensions[-1].value-1)*100,
#        10*100, (10+seq[:, :, :, 0].cube_like_dimensions[0].value-1)*100)))
#     ])
# def test_sequence_plot_as_cube_2D_image(test_input, test_kwargs, expected_values):
#     # Unpack expected values
#     expected_data, expected_xlabel, expected_ylabel, expected_extent = expected_values
#     # Run plot method
#     output = test_input.plot_as_cube(**test_kwargs)
#     # Check values are correct
#     assert isinstance(output, matplotlib.axes.Axes)
#     np.testing.assert_array_equal(output.images[0].get_array(), expected_data)
#     assert output.xaxis.get_label_text() == expected_xlabel
#     assert output.yaxis.get_label_text() == expected_ylabel
#     assert np.allclose(output.images[0].get_extent(), expected_extent, rtol=1e-3)
#     # Also check x and y values?????


# @pytest.mark.parametrize("test_input, test_kwargs, expected_error", [
#     (seq[:, :, :, 0], {"axes_coordinates": [
#         np.arange(10, 10+seq[:, :, :, 0].cube_like_dimensions[-1].value), None],
#         "axes_units": [u.m, None]}, ValueError),

#     (seq[:, :, :, 0], {"axes_coordinates": [
#         None, np.arange(10, 10+seq[:, :, :, 0].cube_like_dimensions[0].value)],
#         "axes_units": [None, u.m]}, ValueError)
#     ])
# def test_sequence_plot_as_cube_2D_image_errors(test_input, test_kwargs, expected_error):
#     with pytest.raises(expected_error):
#         output = test_input.plot_as_cube(**test_kwargs)


# @pytest.mark.parametrize("test_input, test_kwargs, expected_data", [
#     (seq, {}, seq_stack.reshape(4, 1, 2, 3, 4)),
#     (seq_with_units, {}, seq_stack_km.reshape(4, 1, 2, 3, 4))
#     ])
# def test_sequence_plot_ImageAnimator(test_input, test_kwargs, expected_data):
#     # Run plot method
#     output = test_input.plot(**test_kwargs)
#     # Check plot object properties are correct.
#     assert isinstance(output, ndcube.mixins.sequence_plotting.ImageAnimatorNDCubeSequence)
#     np.testing.assert_array_equal(output.data, expected_data)


# @pytest.mark.parametrize("test_input, test_kwargs, expected_data", [
#     (seq, {}, seq_concat.reshape(1, 8, 3, 4)),
#     (seq_with_units, {}, seq_concat_km.reshape(1, 8, 3, 4))
#     ])
# def test_sequence_plot_as_cube_ImageAnimator(test_input, test_kwargs, expected_data):
#     # Run plot method
#     output = test_input.plot_as_cube(**test_kwargs)
#     # Check plot object properties are correct.
#     assert isinstance(output, ndcube.mixins.sequence_plotting.ImageAnimatorCubeLikeNDCubeSequence)
#     np.testing.assert_array_equal(output.data, expected_data)


# @pytest.mark.parametrize("test_input, expected", [
#     ((seq_with_unit0.data, None), (None, None)),
#     ((seq_with_unit0.data, u.km), (None, None)),
#     ((seq_with_units.data, None), ([u.km, u.m, u.km, u.m], u.km)),
#     ((seq_with_units.data, u.cm), ([u.km, u.m, u.km, u.m], u.cm))])
# def test_determine_sequence_units(test_input, expected):
#     output_seq_unit, output_unit = ndcube.mixins.sequence_plotting._determine_sequence_units(
#         test_input[0], unit=test_input[1])
#     assert output_seq_unit == expected[0]
#     assert output_unit == expected[1]


# def test_determine_sequence_units():
#     with pytest.raises(ValueError):
#         output_seq_unit, output_unit = ndcube.mixins.sequence_plotting._determine_sequence_units(
#             seq.data, u.m)


# @pytest.mark.parametrize("test_input, expected", [
#     ((3, 1, "time", u.s), ([1], [None, 'time', None], [None, u.s, None])),
#     ((3, None, None, None), ([-1, -2], None, None))])
# def test_prep_axes_kwargs(test_input, expected):
#     output = ndcube.mixins.sequence_plotting._prep_axes_kwargs(*test_input)
#     for i in range(3):
#         assert output[i] == expected[i]


# @pytest.mark.parametrize("test_input, expected_error", [
#     ((3, [0, 1, 2], ["time", "pix"], u.s), ValueError),
#     ((3, 0, ["time", "pix"], u.s), ValueError),
#     ((3, 0, "time", [u.s, u.pix]), ValueError),
#     ((3, 0, 0, u.s), TypeError),
#     ((3, 0, "time", 0), TypeError)])
# def test_prep_axes_kwargs_errors(test_input, expected_error):
#     with pytest.raises(expected_error):
#         output = ndcube.mixins.sequence_plotting._prep_axes_kwargs(*test_input)


# @pytest.mark.parametrize("test_input, test_kwargs, expected_values", [
#     (seq, {"plot_axis_indices": 3},
#      (seq_stack.data, none_axis_ranges_axis3, "time [min]", "Data [None]",
#       (none_axis_ranges_axis3[-1].min(), none_axis_ranges_axis3[-1].max()),
#       (seq_stack.data.min(), seq_stack.data.max()))),

#     (seq_with_units, {"plot_axis_indices": -1, "data_unit": u.km},
#      (seq_stack_km.data, none_axis_ranges_axis3, "time [min]", "Data [km]",
#       (none_axis_ranges_axis3[-1].min(), none_axis_ranges_axis3[-1].max()),
#       (seq_stack_km.data.min(), seq_stack_km.data.max()))),

#     (seq_with_masks, {"plot_axis_indices": 0},
#      (seq_stack, none_axis_ranges_axis0, "meta.obs.sequence [None]", "Data [None]",
#       (none_axis_ranges_axis0[0].min(), none_axis_ranges_axis0[0].max()),
#       (seq_stack.data.min(), seq_stack.data.max()))),

#     (seq_with_some_masks, {"plot_axis_indices": 0},
#      (seq_stack, none_axis_ranges_axis0, "meta.obs.sequence [None]", "Data [None]",
#       (none_axis_ranges_axis0[0].min(), none_axis_ranges_axis0[0].max()),
#       (seq_stack.data.min(), seq_stack.data.max()))),

#     (seq, {"plot_axis_indices": 0, "axes_coordinates": "distance"},
#      (seq_stack.data, distance0_none_axis_ranges_axis0, "distance [cm]", "Data [None]",
#       (seq.sequence_axis_extra_coords["distance"].value.min(),
#        seq.sequence_axis_extra_coords["distance"].value.max()),
#       (seq_stack.data.min(), seq_stack.data.max()))),

#     (seq, {"plot_axis_indices": 0, "axes_coordinates": "distance", "axes_units": "mm"},
#      (seq_stack.data, distance0_none_axis_ranges_axis0_mm, "distance [mm]", "Data [None]",
#       (seq.sequence_axis_extra_coords["distance"].to("mm").value.min(),
#        seq.sequence_axis_extra_coords["distance"].to("mm").value.max()),
#       (seq_stack.data.min(), seq_stack.data.max()))),

#     (seq, {"plot_axis_indices": 0,
#            "axes_coordinates": userrangequantity_none_axis_ranges_axis0[0]*u.J},
#      (seq_stack.data, userrangequantity_none_axis_ranges_axis0, " [J]", "Data [None]",
#       (userrangequantity_none_axis_ranges_axis0[0].min(),
#        userrangequantity_none_axis_ranges_axis0[0].max()),
#       (seq_stack.data.min(), seq_stack.data.max()))),

#     (seq, {"plot_axis_indices": 0, "axes_units": u.erg,
#            "axes_coordinates": userrangequantity_none_axis_ranges_axis0[0]*u.J},
#      (seq_stack.data, userrangequantity_none_axis_ranges_axis0_1e7, " [erg]", "Data [None]",
#       (userrangequantity_none_axis_ranges_axis0_1e7[0].min(),
#        userrangequantity_none_axis_ranges_axis0_1e7[0].max()),
#       (seq_stack.data.min(), seq_stack.data.max()))),

#     (seq, {"plot_axis_indices": 2, "axes_coordinates": "hi"},
#      (seq_stack.data, hi2_none_axis_ranges_axis2, "hi [s]", "Data [None]",
#       (hi2_none_axis_ranges_axis2[2].min(), hi2_none_axis_ranges_axis2[2].max()),
#       (seq_stack.data.min(), seq_stack.data.max()))),

#     (seq, {"plot_axis_indices": 1, "axes_coordinates": "pix"},
#      (seq_stack.data, pix1_none_axis_ranges_axis1, "pix [pix]", "Data [None]",
#       (pix1_none_axis_ranges_axis1[1].min(), pix1_none_axis_ranges_axis1[1].max()),
#       (seq_stack.data.min(), seq_stack.data.max())))
#     ])
# def test_sequence_plot_LineAnimator(test_input, test_kwargs, expected_values):
#     # Unpack expected values
#     expected_data, expected_axis_ranges, expected_xlabel, \
#       expected_ylabel, expected_xlim, expected_ylim = expected_values
#     # Run plot method.
#     output = test_input.plot(**test_kwargs)
#     # Check right type of plot object is produced.
#     assert type(output) is ndcube.mixins.sequence_plotting.LineAnimatorNDCubeSequence
#     # Check data being plotted is correct
#     np.testing.assert_array_equal(output.data, expected_data)
#     if type(expected_data) is np.ma.core.MaskedArray:
#         np.testing.assert_array_equal(output.data.mask, expected_data.mask)
#     # Check values of axes and sliders is correct.
#     for i in range(len(output.axis_ranges)):
#         print(i)
#         assert np.allclose(output.axis_ranges[i], expected_axis_ranges[i])
#     # Check plot axis labels and limits are correct
#     assert output.xlabel == expected_xlabel
#     assert output.ylabel == expected_ylabel
#     assert output.xlim == expected_xlim
#     assert output.ylim == expected_ylim


# @pytest.mark.parametrize("test_input, test_kwargs, expected_values", [
#     (seq, {"plot_axis_indices": 2, "axes_units": u.s},
#      (seq_concat.data, cubelike_none_axis_ranges_axis2_s, "time [s]", "Data [None]",
#       (cubelike_none_axis_ranges_axis2_s[2].min(), cubelike_none_axis_ranges_axis2_s[2].max()),
#       (seq_concat.data.min(), seq_concat.data.max()))),

#     (seq, {"plot_axis_indices": 0},
#      (seq_concat.data, cubelike_none_axis_ranges_axis0,
#       "custom:pos.helioprojective.lat [deg]", "Data [None]",
#       (0, 7), (seq_concat.data.min(), seq_concat.data.max()))),

#     (seq_with_masks, {"plot_axis_indices": 0},
#      (seq_concat.data, cubelike_none_axis_ranges_axis0,
#       "custom:pos.helioprojective.lat [deg]", "Data [None]",
#       (0, 7), (seq_concat.data.min(), seq_concat.data.max()))),

#     (seq_with_some_masks, {"plot_axis_indices": -3},
#      (seq_concat.data, cubelike_none_axis_ranges_axis0,
#       "custom:pos.helioprojective.lat [deg]", "Data [None]",
#       (0, 7), (seq_concat.data.min(), seq_concat.data.max()))),

#     (seqm, {"plot_axis_indices": 0},
#      (seq_concat.data, cubelike_none_axis_ranges_axis0,
#       "custom:pos.helioprojective.lon [deg]", "Data [None]",
#       (0, 7), (seq_concat.data.min(), seq_concat.data.max())))
#     ])
# def test_sequence_plot_as_cube_LineAnimator(test_input, test_kwargs, expected_values):
#     # Unpack expected values
#     expected_data, expected_axis_ranges, expected_xlabel, \
#       expected_ylabel, expected_xlim, expected_ylim = expected_values
#     # Run plot method.
#     output = test_input.plot_as_cube(**test_kwargs)
#     # Check right type of plot object is produced.
#     assert type(output) is ndcube.mixins.sequence_plotting.LineAnimatorCubeLikeNDCubeSequence
#     # Check data being plotted is correct
#     np.testing.assert_array_equal(output.data, expected_data)
#     if type(expected_data) is np.ma.core.MaskedArray:
#         np.testing.assert_array_equal(output.data.mask, expected_data.mask)
#     # Check values of axes and sliders is correct.
#     for i in range(len(output.axis_ranges)):
#         assert np.allclose(output.axis_ranges[i], expected_axis_ranges[i])
#     # Check plot axis labels and limits are correct
#     assert output.xlabel == expected_xlabel
#     assert output.ylabel == expected_ylabel
#     assert output.xlim == expected_xlim
#     assert output.ylim == expected_ylim
