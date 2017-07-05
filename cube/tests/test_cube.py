# -*- coding: utf-8 -*-
'''
Tests for Cube
'''
from __future__ import absolute_import
from sunpycube.cube.datacube import Cube
from sunpycube.cube import cube_utils as cu
from sunpy.map.mapbase import GenericMap
from sunpycube.spectra.spectrum import Spectrum
from sunpycube.spectra.spectrogram import Spectrogram
from sunpy.lightcurve.lightcurve import LightCurve
import numpy as np
from sunpycube.wcs_util import WCS
import pytest
import astropy.units as u


# sample data for tests
# TODO: use a fixture reading from a test file. file TBD.
ht = {'CTYPE1': 'HPLT-TAN', 'CUNIT1': 'deg', 'CDELT1': 0.5, 'CRPIX1': 0, 'CRVAL1': 0, 'NAXIS1': 2,
      'CTYPE2': 'WAVE    ', 'CUNIT2': 'Angstrom', 'CDELT2': 0.2, 'CRPIX2': 0, 'CRVAL2': 0, 'NAXIS2': 3,
      'CTYPE3': 'TIME    ', 'CUNIT3': 'min', 'CDELT3': 0.4, 'CRPIX3': 0, 'CRVAL3': 0, 'NAXIS3': 4}
wt = WCS(header=ht, naxis=3)
data = np.array([[[1, 2, 3, 4], [2, 4, 5, 3], [0, -1, 2, 3]],
                 [[2, 4, 5, 1], [10, 5, 2, 2], [10, 3, 3, 0]]])
cube = Cube(data, wt)

hm = {
    'CTYPE1': 'WAVE    ', 'CUNIT1': 'Angstrom', 'CDELT1': 0.2, 'CRPIX1': 0, 'CRVAL1': 10, 'NAXIS1': 4,
    'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5, 'NAXIS2': 3,
    'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 2,
}
wm = WCS(header=hm, naxis=3)
cubem = Cube(data, wm)

h4 = {
    'CTYPE1': 'TIME    ', 'CUNIT1': 'min', 'CDELT1': 0.6, 'CRPIX1': 0, 'CRVAL1': 0, 'NAXIS1': 4,
    'CTYPE2': 'WAVE    ', 'CUNIT2': 'nm', 'CDELT2': 0.2, 'CRPIX2': 0, 'CRVAL2': 10, 'NAXIS2': 3,
    'CTYPE3': 'HPLT-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.5, 'CRPIX3': 0, 'CRVAL3': 0, 'NAXIS3': 2,
    'CTYPE4': 'HPLN-TAN', 'CUNIT4': 'deg', 'CDELT4': 0.4, 'CRPIX4': 2, 'CRVAL4': 1, 'NAXIS4': 2
}
w4 = WCS(header=h4, naxis=4)
data4 = np.array([[[[1, 2, 3, 4], [2, 4, 5, 3], [0, -1, 2, 3]],
                   [[2, 4, 5, 1], [10, 5, 2, 2], [10, 3, 3, 0]]],

                  [[[4, 6, 3, 5], [1, 0, 5, 3], [-4, 8, 7, 6]],
                   [[7, 3, 2, 6], [1, 7, 8, 7], [2, 4, 0, 1]]]])
hcube = Cube(data4, w4)


def test_choose_wavelength_slice_with_time():
    ius = cube._choose_wavelength_slice(-1)  # integer, under range slice
    iis = cube._choose_wavelength_slice(1)  # integer, in range slice
    ios = cube._choose_wavelength_slice(11)  # integer, over range slice

    qus = cube._choose_wavelength_slice(-1 * u.Angstrom)  # quantity, under
    qis = cube._choose_wavelength_slice(0.5 * u.Angstrom)  # quantity, in
    qos = cube._choose_wavelength_slice(8 * u.Angstrom)  # quantity, over

    f = cube._choose_wavelength_slice(0.4)  # no units given

    assert ius is None
    assert np.all(iis == [[2, 4, 5, 3], [10, 5, 2, 2]])
    assert ios is None

    assert qus is None
    assert np.all(qis == [[0, -1, 2, 3], [10, 3, 3, 0]])
    assert qos is None

    assert f is None


def test_choose_wavelength_no_time():
    ius = cubem._choose_wavelength_slice(-1)  # integer, under range slice
    iis = cubem._choose_wavelength_slice(1)  # integer, in range slice
    ios = cubem._choose_wavelength_slice(11)  # integer, over range slice

    qus = cubem._choose_wavelength_slice(-1 * u.Angstrom)  # quantity, under
    qis = cubem._choose_wavelength_slice(0.4 * u.Angstrom)  # quantity, in
    qos = cubem._choose_wavelength_slice(8 * u.Angstrom)  # quantity, over

    f = cubem._choose_wavelength_slice(0.4)  # no units given

    assert ius is None
    assert np.all(iis == [[2, 4, -1], [4, 5, 3]])
    assert ios is None

    assert qus is None
    assert np.all(qis == [[3, 5, 2], [5, 2, 3]])
    assert qos is None

    assert f is None


def test_choose_x_slice():
    ius = cube._choose_x_slice(-1)  # integer, under range slice
    iis = cube._choose_x_slice(1)  # integer, in range slice
    ios = cube._choose_x_slice(11)  # integer, over range slice

    qus = cube._choose_x_slice(-1 * u.deg)  # quantity, under
    qis = cube._choose_x_slice(0.5 * u.deg)  # quantity, in
    qos = cube._choose_x_slice(8 * u.deg)  # quantity, over

    f = cube._choose_x_slice(0.4)  # no units given

    assert ius is None
    assert np.all(iis == [[2, 4, 5, 1], [10, 5, 2, 2], [10, 3, 3, 0]])
    assert ios is None

    assert qus is None
    assert np.all(qis == [[1, 2, 3, 4], [2, 4, 5, 3], [0, -1, 2, 3]])
    assert qos is None

    assert f is None


def test_slice_to_map_with_time():
    with pytest.raises(cu.CubeError):
        cube.slice_to_map(0)
    with pytest.raises(cu.CubeError):
        cube.slice_to_map((0, 3))


def test_slice_to_map_no_time():
    m0 = cubem.slice_to_map(0)
    m1 = cubem.slice_to_map((0, 3))
    assert np.all(m0.data == cubem.data[0])
    assert np.all(m1.data == cubem.data.sum(0))


def test_wavelength_axis():
    cube_wavelength_axis = cube.wavelength_axis()
    cubem_wavelength_axis = cubem.wavelength_axis()
    f1, u1 = cube_wavelength_axis.value, cube_wavelength_axis.unit
    f2, u2 = cubem_wavelength_axis.value, cubem_wavelength_axis.unit
    # the e-11 are the conversions from angstrom to meters
    assert np.allclose(f1, [0.00000000e+00,  3.00000000e-11,  6.00000000e-11])
    assert np.allclose(f2, [1.00000000e-09,  1.01333333e-09,  1.02666667e-09,  1.04000000e-09])
    assert u1 == u.m
    assert u2 == u.m


def test_time_axis():
    cube_time_axis = cube.time_axis()
    t1, u1 = cube_time_axis.value, cube_time_axis.unit
    assert np.allclose(t1, [0., 0.26666667, 0.53333333, 0.8])
    assert u1 == u.min
    with pytest.raises(cu.CubeError):
        cubem.time_axis()


@pytest.mark.parametrize("test_input,expected", [
    (cubem[1], GenericMap),
    (cubem[0:2], Cube),
    (cubem[:], Cube),
    (cube[1], Spectrum),
    (cube[0:2], Cube),
    (cube[:], Cube),
])
def test_slicing_first_axis(test_input, expected):
    assert isinstance(test_input, expected)
    with pytest.raises(IndexError):
        cubem[None]
    with pytest.raises(IndexError):
        cube[None]


@pytest.mark.parametrize("test_input,expected", [
    (cubem[0.5 * u.deg], GenericMap),
    (cubem[0.5 * u.deg:10.4], Cube),
    (cubem[::0.4 * u.deg], Cube),
    (cube[0.5 * u.deg], Spectrum),
    (cube[0:0.8 * u.deg], Cube),
    (cube[::1.0 * u.deg], Cube),
])
def test_slicing_first_axis_world_coord(test_input, expected):
    assert isinstance(test_input, expected)
    with pytest.raises(IndexError):
        cubem[None]
    with pytest.raises(IndexError):
        cube[None]


@pytest.mark.parametrize("test_input,expected", [
    (cubem[:, 1], np.ndarray),
    (cubem[:, 0:2], Cube),
    (cubem[:, :], Cube),
    (cubem[1, 1], np.ndarray),
    (cubem[1, 0:2], GenericMap),
    (cubem[1, :], GenericMap),
    (cube[:, 1], LightCurve),
    (cube[:, 0:2], Cube),
    (cube[:, :], Cube),
    (cube[1, 1], np.ndarray),
    (cube[1, 0:2], Spectrum),
    (cube[1, :], Spectrum),
])
def test_slicing_second_axis(test_input, expected):
    assert isinstance(test_input, expected)


@pytest.mark.parametrize("test_input,expected", [
    (cubem[:, -0.5 * u.deg], np.ndarray),
    (cubem[:, -0.5:0.5 * u.deg], Cube),
    (cubem[:, ::1 * u.deg], Cube),
    (cubem[0.5 * u.deg, 0 * u.deg], np.ndarray),
    (cubem[1, -0.5:0.5 * u.deg], GenericMap),
    (cubem[1, ::1 * u.deg], GenericMap),
    (cube[:, 0.2 * u.Angstrom], LightCurve),
    (cube[:, 0:0.4 * u.Angstrom], Cube),
    (cube[:, 0:0.4 * u.Angstrom:0.4], Cube),
    (cube[1, 0.2 * u.Angstrom], np.ndarray),
    (cube[0.5 * u.deg, 0:2], Spectrum),
    (cube[0.5 * u.deg, 0:0.4 * u.Angstrom:0.4], Spectrum),

])
def test_slicing_second_axis_world_coord(test_input, expected):
    assert isinstance(test_input, expected)


@pytest.mark.parametrize("test_input,expected", [
    (cubem[:, :, 1], np.ndarray),
    (cubem[:, :, 0:2], Cube),
    (cubem[:, :, :], Cube),
    (cubem[:, 1, 1], Spectrum),
    (cubem[:, 1, 0:2], np.ndarray),
    (cubem[:, 1, :], np.ndarray),
    (cubem[1, :, 1], np.ndarray),
    (cubem[1, :, 0:2], GenericMap),
    (cubem[1, :, :], GenericMap),
    (cubem[1, 1, 1], int),
    (cubem[1, 1, 0:2], np.ndarray),
    (cubem[1, 1, :], np.ndarray),
    (cube[:, :, 1], Spectrogram),
    (cube[:, :, 0:2], Cube),
    (cube[:, :, :], Cube),
    (cube[:, 1, 1], LightCurve),
    (cube[:, 1, 0:2], LightCurve),
    (cube[:, 1, :], LightCurve),
    (cube[1, :, 1], Spectrum),
    (cube[1, :, 0:2], Spectrum),
    (cube[1, :, :], Spectrum),
    (cube[1, 1, 1], int),
    (cube[1, 1, 0:2], np.ndarray),
    (cube[1, 1, :], np.ndarray),
])
def test_slicing_third_axis(test_input, expected):
    assert isinstance(test_input, expected)


@pytest.mark.parametrize("test_input,expected", [
    (cubem[:, :,  10 * u.Angstrom], np.ndarray),
    (cubem[:, :, 0.2:0.1 * u.Angstrom], Cube),
    (cubem[:, :, 0.2:1 * u.Angstrom:0.8], Cube),
    (cubem[:, -0.5 * u.deg, 10 * u.Angstrom], Spectrum),
    (cubem[:, -0.5 * u.deg, 0:2], np.ndarray),
    (cubem[:, -0.5 * u.deg, 0.2:1 * u.Angstrom:0.8], np.ndarray),
    (cubem[0.5 * u.deg, :, 10 * u.Angstrom], np.ndarray),
    (cubem[1, :, 0.2:1 * u.Angstrom], GenericMap),
    (cubem[0.5 * u.deg, :, 0.2:1 * u.Angstrom:0.8], GenericMap),
    (cubem[0.5 * u.deg, -0.5 * u.deg, 10 * u.Angstrom], int),
    (cubem[1, 1, 0.2:1 * u.Angstrom], np.ndarray),
    (cubem[0.5 * u.deg, -0.5 * u.deg, 0.2:1 * u.Angstrom:0.8], np.ndarray),
    (cube[:, :, 0.5 * u.min], Spectrogram),
    (cube[:, :, 0:1 * u.min], Cube),
    (cube[:, :, ::1 * u.min], Cube),
    (cube[:, 0.2 * u.Angstrom, 0.5 * u.min], LightCurve),
    (cube[:, 1, 0:1 * u.min], LightCurve),
    (cube[:, 0.2 * u.Angstrom, ::1 * u.min], LightCurve),
    (cube[0.5 * u.deg, :, 0.5 * u.min], Spectrum),
    (cube[1, :, 0:1 * u.min], Spectrum),
    (cube[0.5 * u.deg, :, ::1 * u.min], Spectrum),
    (cube[0.5 * u.deg, 0.2 * u.Angstrom, 0.5 * u.min], int),
    (cube[0.5 * u.deg, 1, 0:1 * u.min], np.ndarray),
    (cube[0.5 * u.deg, 1, :], np.ndarray),
])
def test_slicing_third_axis_world_coord(test_input, expected):
    assert isinstance(test_input, expected)


@pytest.mark.parametrize("test_input,expected", [
    (hcube[1, 1, 1, 1], int),
    (hcube[0, 1, 1], np.ndarray),
    (hcube[1, 0, 1, :], np.ndarray),
    (hcube[1, 1, :, 2], np.ndarray),
])
def test_4d_getitem_to_array(test_input, expected):
    assert isinstance(test_input, expected)


@pytest.mark.parametrize("test_input,expected", [
    (hcube[1, 0.5 * u.deg, 1, 0.6 * u.min], int),
    (hcube[0, 1, :, 1 * u.min], np.ndarray),
    (hcube[0.7 * u.deg, 1, 93 * u.Angstrom], np.ndarray),
    (hcube[0.5 * u.deg, 0.0 * u.deg, 1, :], np.ndarray),
])
def test_4d_getitem_to_array_world_coord(test_input, expected):
    assert isinstance(test_input, expected)


@pytest.mark.parametrize("test_input,expected", [
    (hcube[1, 0], GenericMap),
    (hcube[1, 1, :], GenericMap),
    (hcube[1, 1, :, 0:2], GenericMap),
])
def test_4d_getitem_to_map(test_input, expected):
    assert isinstance(test_input, expected)


@pytest.mark.parametrize("test_input,expected", [
    (hcube[0.5 * u.deg, 0], GenericMap),
    (hcube[0.4 * u.deg, 0.0 * u.deg, :], GenericMap),
    (hcube[0.0 * u.deg, 0.5 * u.deg, :, 0:2], GenericMap)
])
def test_4d_getitem_to_map_world_coord(test_input, expected):
    assert isinstance(test_input, expected)


@pytest.mark.parametrize("test_input,expected", [
    (hcube[1, :, 1, 2], Spectrum),
    (hcube[0, :, 0], Spectrum),
    (hcube[1, :, 1, 0:2], Spectrum),
    (hcube[0, :, :, 0], Spectrum),
])
def test_4d_getitem_to_spectrum(test_input, expected):
    assert isinstance(test_input, expected)


@pytest.mark.parametrize("test_input,expected", [
    (hcube[0.4 * u.deg, :, 95.0 * u.Angstrom, 2], Spectrum),
    (hcube[0.5 * u.deg, :, 100 * u.Angstrom], Spectrum),
    (hcube[0.5 * u.deg, :, 1, :2], Spectrum),
    (hcube[0 * u.deg, :, :, 0], Spectrum),
])
def test_4d_getitem_to_spectrum_world_coord(test_input, expected):
    assert isinstance(test_input, expected)


def test_4d_getitem_to_cube():
    slices = [hcube[1],
              hcube[1, 0:1],
              hcube[0, :, 0:2],
              hcube[0, :, :, 0:2],
              hcube[1:3, 1],
              hcube[0:2, 0, :],
              hcube[:, 0, :, 0:2],
              hcube[0:1, 0, :, 1:2],
              hcube[:, 1, :, :]]
    for s in slices:
        assert isinstance(s, Cube) and s.data.ndim == 3


def test_4d_getitem_to_cube_world_coord():
    slices = [hcube[0.3 * u.deg],
              hcube[0.5 * u.deg, 0:1],
              hcube[0.3 * u.deg, :, 0:2],
              hcube[0.5 * u.deg, :, :, 0:2],
              hcube[1:3, 0.6 * u.deg],
              hcube[0:2, 0.2 * u.deg, :],
              hcube[:, 0.2 * u.deg, :, 0:2],
              hcube[0.0 * u.deg, :, :, 1:2],
              hcube[:, 0.1 * u.deg, :]]
    for s in slices:
        assert isinstance(s, Cube) and s.data.ndim == 3


def test_4d_getitem_to_hypercube():
    slices = [hcube[1:3, :, :, 0:1], hcube[1:, :1, :], hcube[2:, :], hcube[1:]]
    for s in slices:
        assert isinstance(s, Cube) and s.data.ndim == 4


def test_4d_getitem_to_hypercube_world_coord():
    slices = [hcube[0.4 * u.deg:1.2 * u.deg:0.8, :, :, 0:1],
              hcube[2:, 0: 0.5 * u.deg:1],
              hcube[0.4 * u.deg:1.2 * u.deg:0.8]]
    for s in slices:
        assert isinstance(s, Cube) and s.data.ndim == 4


def test_4d_getitem_to_spectrogram():
    s = hcube[2:, :, 1, 2]
    assert isinstance(s, Spectrogram)


def test_4d_getitem_to_spectrogram_world_coord():
    s = hcube[0.4 * u.deg:1.2:0.8, 0:0.5 * u.deg:2, 1, 0.6 * u.min]
    assert isinstance(s, Spectrogram)


@pytest.mark.parametrize("test_input,expected", [
    (hcube[:, 0, 0, 0], LightCurve),
    (hcube[:, 1, 1, :], LightCurve),
    (hcube[:, 1, 0], LightCurve),
    (hcube[:, 1, :, 2], LightCurve),
])
def test_4d_getitem_to_lightcurve(test_input, expected):
    assert isinstance(test_input, expected)


@pytest.mark.parametrize("test_input, expected", [
    (hcube[:, 0, 0, 0.6 * u.min], LightCurve),
    (hcube[:, 1, :, 1 * u.min], LightCurve),
    (hcube[0.4 * u.deg:1.2:0.8, 1, 95 * u.Angstrom, :], LightCurve),
    (hcube[0.4 * u.deg:1.2:0.8, 1, 100 * u.Angstrom], LightCurve)
])
def test_4d_getitem_to_lightcurve_world_coord(test_input, expected):
    assert isinstance(test_input, expected)


def test_reduce_dim():
    slices = [slice(s, e, t) for s, e, t in [(None, None, None), (0, 2, None),
                                             (None, None, 2)]]
    assert np.all(cube.data == cu.reduce_dim(cube, 0, slices[0]))
    assert np.all(cube.data == cu.reduce_dim(cube, 0, slices[1]))
    assert cu.reduce_dim(cubem, 0, slices[1]).data.shape == (2, 3, 4)
    assert cu.reduce_dim(cubem, 2, slices[2]).data.shape == (2, 3, 2)
    assert cu.reduce_dim(cube, 2, slices[2]).axes_wcs.wcs.cdelt[-2] == 0.5
