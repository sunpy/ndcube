from __future__ import absolute_import
from sunpycube.cube.NDCube import NDCube, NDCubeSequence
from sunpycube.cube import cube_utils as cu
from sunpycube.wcs_util import WCS
from collections import namedtuple
import numpy as np
import pytest
import astropy.units as u

SequenceDimensionPair = namedtuple('SequenceDimensionPair', 'shape axis_types')

# sample data for tests
# TODO: use a fixture reading from a test file. file TBD.
data = np.array([[[1, 2, 3, 4], [2, 4, 5, 3], [0, -1, 2, 3]],
                 [[2, 4, 5, 1], [10, 5, 2, 2], [10, 3, 3, 0]]])

data2 = np.array([[[11, 22, 33, 44], [22, 44, 55, 33], [0, -1, 22, 33]],
                  [[22, 44, 55, 11], [10, 55, 22, 22], [10, 33, 33, 0]]])

ht = {'CTYPE3': 'HPLT-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.5, 'CRPIX3': 0, 'CRVAL3': 0, 'NAXIS3': 2,
      'CTYPE2': 'WAVE    ', 'CUNIT2': 'Angstrom', 'CDELT2': 0.2, 'CRPIX2': 0, 'CRVAL2': 0, 'NAXIS2': 3,
      'CTYPE1': 'TIME    ', 'CUNIT1': 'min', 'CDELT1': 0.4, 'CRPIX1': 0, 'CRVAL1': 0, 'NAXIS1': 4}

hm = {
    'CTYPE1': 'WAVE    ', 'CUNIT1': 'Angstrom', 'CDELT1': 0.2, 'CRPIX1': 0, 'CRVAL1': 10, 'NAXIS1': 4,
    'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5, 'NAXIS2': 3,
    'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 2,
}

wt = WCS(header=ht, naxis=3)
wm = WCS(header=hm, naxis=3)

cube1 = NDCube(data, wcs=wt, missing_axis=[False, False, False, True])
cube2 = NDCube(data, wcs=wm)
cube3 = NDCube(data2, wcs=wt, missing_axis=[False, False, False, True])
cube4 = NDCube(data2, wcs=wm)

seq = NDCubeSequence([cube1, cube2, cube3, cube4], common_axis=0)
seq1 = NDCubeSequence([cube1, cube2, cube3, cube4])


@pytest.mark.parametrize("test_input,expected", [
    (seq[0], NDCube),
    (seq[1], NDCube),
    (seq[2], NDCube),
    (seq[3], NDCube),
    (seq[0:1], NDCubeSequence),
    (seq[1:3], NDCubeSequence),
    (seq[0:2], NDCubeSequence),
    (seq[slice(0, 2)], NDCubeSequence),
    (seq[slice(0, 3)], NDCubeSequence),
])
def test_slice_first_index_sequence(test_input, expected):
    assert isinstance(test_input, expected)


@pytest.mark.parametrize("test_input,expected", [
    (seq[0:1].dimensions.shape[0], 1),
    (seq[1:3].dimensions.shape[0], 2),
    (seq[0:2].dimensions.shape[0], 2),
    (seq[0::].dimensions.shape[0], 4),
    (seq[slice(0, 2)].dimensions.shape[0], 2),
    (seq[slice(0, 3)].dimensions.shape[0], 3),
])
def test_slice_first_index_sequence(test_input, expected):
    assert test_input == expected


@pytest.mark.parametrize("test_input,expected", [
    (seq.index_as_cube[0:5].dimensions, SequenceDimensionPair(shape=(
        [3] + list(u.Quantity((2, 3, 4), unit=u.pix))), axis_types=('Sequence Axis', 'HPLT-TAN', 'WAVE', 'TIME'))),
    (seq.index_as_cube[1:3].dimensions, SequenceDimensionPair(shape=(
        [2] + list(u.Quantity((1, 3, 4), unit=u.pix))), axis_types=('Sequence Axis', 'HPLT-TAN', 'WAVE', 'TIME'))),
    (seq.index_as_cube[0:6].dimensions, SequenceDimensionPair(shape=(
        [3] + list(u.Quantity((2, 3, 4), unit=u.pix))), axis_types=('Sequence Axis', 'HPLT-TAN', 'WAVE', 'TIME'))),
    (seq.index_as_cube[0::].dimensions, SequenceDimensionPair(shape=(
        [4] + list(u.Quantity((2, 3, 4), unit=u.pix))), axis_types=('Sequence Axis', 'HPLT-TAN', 'WAVE', 'TIME'))),
    (seq.index_as_cube[0:5, 0].dimensions, SequenceDimensionPair(
        shape=([3] + list(u.Quantity((3, 4), unit=u.pix))), axis_types=('Sequence Axis', 'WAVE', 'TIME'))),
    (seq.index_as_cube[1:3, 0:2].dimensions, SequenceDimensionPair(shape=(
        [2] + list(u.Quantity((1, 3, 4), unit=u.pix))), axis_types=('Sequence Axis', 'HPLT-TAN', 'WAVE', 'TIME'))),
    (seq.index_as_cube[0:6, 0, 0:1].dimensions, SequenceDimensionPair(
        shape=([3] + list(u.Quantity((1, 4), unit=u.pix))), axis_types=('Sequence Axis', 'WAVE', 'TIME'))),
    (seq.index_as_cube[0::, 0, 0].dimensions, SequenceDimensionPair(
        shape=([4] + list(u.Quantity((4,), unit=u.pix))), axis_types=('Sequence Axis', 'TIME'))),
])
def test_index_as_cube(test_input, expected):
    assert test_input.shape[0] == expected.shape[0]
    for seq_indexed, expected_dim in zip(test_input.shape[1::], expected.shape[1::]):
        assert seq_indexed.value == expected_dim.value
    assert test_input.axis_types == expected.axis_types


@pytest.mark.parametrize("test_input,expected", [
    (seq1.explode_along_axis(axis=0), SequenceDimensionPair(
        shape=([8] + list(u.Quantity((3, 4), unit=u.pix))), axis_types=('Sequence Axis', 'WAVE', 'TIME'))),
    (seq1.explode_along_axis(axis=1), SequenceDimensionPair(shape=(
        [12] + list(u.Quantity((2, 4), unit=u.pix))), axis_types=('Sequence Axis', 'HPLT-TAN', 'TIME'))),
    (seq1.explode_along_axis(axis=2), SequenceDimensionPair(shape=(
        [16] + list(u.Quantity((2, 3), unit=u.pix))), axis_types=('Sequence Axis', 'HPLT-TAN', 'WAVE')))
])
def test_explode_along_axis(test_input, expected):
    assert test_input.dimensions.shape[0] == expected.shape[0]
    for seq_indexed, expected_dim in zip(test_input.dimensions.shape[1::], expected.shape[1::]):
        assert seq_indexed.value == expected_dim.value
    assert test_input.dimensions.axis_types == expected.axis_types


def test_explode_along_axis():
    with pytest.raises(ValueError):
        seq.explode_along_axis(axis=1)
