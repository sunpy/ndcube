
import numpy as np
import pytest
from astropy.wcs import WCS

from ndcube import utils

ht_with_celestial = {
    'CTYPE4': 'HPLN-TAN', 'CUNIT4': 'deg', 'CDELT4': 1, 'CRPIX4': 0, 'CRVAL4': 0, 'NAXIS4': 1,
    'CNAME4': 'redundant axis', 'CROTA4': 0,
    'CTYPE3': 'HPLT-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.5, 'CRPIX3': 0, 'CRVAL3': 0, 'NAXIS3': 2,
    'CTYPE2': 'WAVE    ', 'CUNIT2': 'Angstrom', 'CDELT2': 0.2, 'CRPIX2': 0, 'CRVAL2': 0,
    'NAXIS2': 3,
    'CTYPE1': 'TIME    ', 'CUNIT1': 'min', 'CDELT1': 0.4, 'CRPIX1': 0, 'CRVAL1': 0, 'NAXIS1': 4}

hm = {'CTYPE1': 'WAVE    ', 'CUNIT1': 'Angstrom', 'CDELT1': 0.2, 'CRPIX1': 0, 'CRVAL1': 10,
      'NAXIS1': 4,
      'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5,
      'NAXIS2': 3,
      'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 2}
wm = WCS(header=hm)

hm_reindexed_102 = {
    'CTYPE2': 'WAVE    ', 'CUNIT2': 'Angstrom', 'CDELT2': 0.2, 'CRPIX2': 0, 'CRVAL2': 10,
    'NAXIS2': 4,
    'CTYPE1': 'HPLT-TAN', 'CUNIT1': 'deg', 'CDELT1': 0.5, 'CRPIX1': 2, 'CRVAL1': 0.5, 'NAXIS1': 3,
    'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 2}
wm_reindexed_102 = WCS(header=hm_reindexed_102)


@pytest.fixture
def axis_correlation_matrix():
    return _axis_correlation_matrix()


def _axis_correlation_matrix():
    return np.array([[True, True, False, False],
                     [True, True, False, False],
                     [False, False, True, False],
                     [True, False, False, True]], dtype=bool)


@pytest.fixture
def test_wcs():
    return WCSTest()


class WCSTest():
    def __init__(self):
        self.world_axis_physical_types = [
            'custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat', 'em.wl', 'time']
        self.axis_correlation_matrix = _axis_correlation_matrix()


def test_convert_between_array_and_pixel_axes():
    test_input = np.array([1, 4, -2])
    naxes = 5
    expected = np.array([3, 0, 1])
    output = utils.wcs.convert_between_array_and_pixel_axes(test_input, naxes)
    assert all(output == expected)


def test_pixel_axis_to_world_axes(axis_correlation_matrix):
    output = utils.wcs.pixel_axis_to_world_axes(0, axis_correlation_matrix)
    expected = np.array([0, 1, 3])
    assert all(output == expected)


def test_world_axis_to_pixel_axes(axis_correlation_matrix):
    output = utils.wcs.world_axis_to_pixel_axes(1, axis_correlation_matrix)
    expected = np.array([0, 1])
    assert all(output == expected)


def test_pixel_axis_to_physical_types(test_wcs):
    output = utils.wcs.pixel_axis_to_physical_types(0, test_wcs)
    expected = np.array(['custom:pos.helioprojective.lon',
                         'custom:pos.helioprojective.lat', 'time'])
    assert all(output == expected)


def test_physical_type_to_pixel_axes(test_wcs):
    output = utils.wcs.physical_type_to_pixel_axes('lon', test_wcs)
    expected = np.array([0, 1])
    assert all(output == expected)


@pytest.mark.parametrize("test_input,expected", [('wl', 2), ('em.wl', 2)])
def test_physical_type_to_world_axis(test_input, expected):
    world_axis_physical_types = ['custom:pos.helioprojective.lon',
                                 'custom:pos.helioprojective.lat', 'em.wl', 'time']
    output = utils.wcs.physical_type_to_world_axis(test_input, world_axis_physical_types)
    assert output == expected


def test_get_dependent_pixel_axes(axis_correlation_matrix):
    output = utils.wcs.get_dependent_pixel_axes(0, axis_correlation_matrix)
    expected = np.array([0, 1, 3])
    assert all(output == expected)


def test_get_dependent_array_axes(axis_correlation_matrix):
    output = utils.wcs.get_dependent_array_axes(3, axis_correlation_matrix)
    expected = np.array([0, 2, 3])
    assert all(output == expected)


def test_get_dependent_world_axes(axis_correlation_matrix):
    output = utils.wcs.get_dependent_world_axes(3, axis_correlation_matrix)
    expected = np.array([0, 1, 3])
    assert all(output == expected)


def test_get_dependent_physical_types(test_wcs):
    output = utils.wcs.get_dependent_physical_types("time", test_wcs)
    expected = np.array(['custom:pos.helioprojective.lon',
                         'custom:pos.helioprojective.lat', 'time'])
    assert all(output == expected)


def test_array_indices_for_world_objects(wcs_4d_t_l_lt_ln):
    array_indices = utils.wcs.array_indices_for_world_objects(wcs_4d_t_l_lt_ln, None)
    assert len(array_indices) == 3
    assert array_indices == ((3,), (2,), (0, 1))

    array_indices = utils.wcs.array_indices_for_world_objects(wcs_4d_t_l_lt_ln, ('time',))
    assert len(array_indices) == 1
    assert array_indices == ((3,),)

    array_indices = utils.wcs.array_indices_for_world_objects(wcs_4d_t_l_lt_ln, ('time', 'em.wl'))
    assert len(array_indices) == 2
    assert array_indices == ((3,), (2,))

    array_indices = utils.wcs.array_indices_for_world_objects(wcs_4d_t_l_lt_ln, ('lat',))
    assert len(array_indices) == 1
    assert array_indices == ((0, 1),)


def test_array_indices_for_world_objects_2(wcs_4d_lt_t_l_ln):
    array_indices = utils.wcs.array_indices_for_world_objects(wcs_4d_lt_t_l_ln, None)
    assert len(array_indices) == 3
    assert array_indices == ((0, 3), (2,), (1,))

    array_indices = utils.wcs.array_indices_for_world_objects(wcs_4d_lt_t_l_ln, ('lat',))
    assert len(array_indices) == 1
    assert array_indices == ((0, 3),)

    array_indices = utils.wcs.array_indices_for_world_objects(wcs_4d_lt_t_l_ln, ('lat', 'time'))
    assert len(array_indices) == 2
    assert array_indices == ((0, 3), (2,))

    array_indices = utils.wcs.array_indices_for_world_objects(wcs_4d_lt_t_l_ln, ('lon', 'time'))
    assert len(array_indices) == 2
    assert array_indices == ((0, 3), (2,))


def test_compare_wcs_physical_types(wcs_4d_t_l_lt_ln, wcs_3d_l_lt_ln):
    assert utils.wcs.compare_wcs_physical_types(wcs_4d_t_l_lt_ln, wcs_4d_t_l_lt_ln) is True
    assert utils.wcs.compare_wcs_physical_types(wcs_4d_t_l_lt_ln, wcs_3d_l_lt_ln) is False


def test_identify_invariant_axes(wcs_3d_l_lt_ln):
    source_wcs = wcs_3d_l_lt_ln

    target_wcs_header = wcs_3d_l_lt_ln.low_level_wcs.to_header().copy()
    target_wcs_header['CDELT2'] = 10
    target_wcs_header['CDELT3'] = 20
    target_wcs = WCS(header=target_wcs_header)

    invariant_axes = utils.wcs.identify_invariant_axes(source_wcs, target_wcs, (4, 4, 4))
    assert invariant_axes == [True, False, False]

    invariant_axes = utils.wcs.identify_invariant_axes(source_wcs, target_wcs, (4, 4, 4),
                                                       atol=1e-20, rtol=1e-20)
    assert invariant_axes == [False, False, False]
