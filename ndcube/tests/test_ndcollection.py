
import astropy.units as u
import astropy.wcs
import numpy as np
import pytest

from ndcube import NDCollection, NDCube, NDCubeSequence
from ndcube.tests import helpers

# Define some mock data
data0 = np.ones((3, 4, 5))
data1 = np.zeros((5, 3, 4))
data2 = data0 * 2

# Define WCS object for all cubes.
wcs_input_dict = {
    'CTYPE1': 'WAVE    ', 'CUNIT1': 'Angstrom', 'CDELT1': 0.2, 'CRPIX1': 0, 'CRVAL1': 10, 'NAXIS1': 5,
    'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5, 'NAXIS2': 4,
    'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 3}
input_wcs = astropy.wcs.WCS(wcs_input_dict)

wcs_input_dict1 = {
    'CTYPE3': 'WAVE    ', 'CUNIT3': 'Angstrom', 'CDELT3': 0.2, 'CRPIX3': 0, 'CRVAL3': 10, 'NAXIS3': 5,
    'CTYPE1': 'HPLT-TAN', 'CUNIT1': 'deg', 'CDELT1': 0.5, 'CRPIX1': 2, 'CRVAL1': 0.5, 'NAXIS1': 4,
    'CTYPE2': 'HPLN-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.4, 'CRPIX2': 2, 'CRVAL2': 1, 'NAXIS2': 3}
input_wcs1 = astropy.wcs.WCS(wcs_input_dict1)

# Define cubes.
cube0 = NDCube(data0, input_wcs)
cube1 = NDCube(data1, input_wcs1)
cube2 = NDCube(data2, input_wcs)

# Define sequences.
sequence02 = NDCubeSequence([cube0, cube2])
sequence20 = NDCubeSequence([cube2, cube0])

# Define collections
aligned_axes = ((1, 2), (2, 0), (1, 2))
keys = ("cube0", "cube1", "cube2")
cube_collection = NDCollection([("cube0", cube0), ("cube1", cube1), ("cube2", cube2)], aligned_axes)
seq_collection = NDCollection([("seq0", sequence02), ("seq1", sequence20)], aligned_axes="all")


@pytest.mark.parametrize("item,collection,expected", [
    (0, cube_collection,
        NDCollection([("cube0", cube0[:, 0]), ("cube1", cube1[:, :, 0]), ("cube2", cube2[:, 0])],
                     aligned_axes=((1,), (0,), (1,)))),

    (slice(1, 3), cube_collection, NDCollection(
        [("cube0", cube0[:, 1:3]), ("cube1", cube1[:, :, 1:3]), ("cube2", cube2[:, 1:3])],
        aligned_axes=aligned_axes)),

    (slice(-3, -1), cube_collection, NDCollection(
        [("cube0", cube0[:, -3:-1]), ("cube1", cube1[:, :, -3:-1]), ("cube2", cube2[:, -3:-1])],
        aligned_axes=aligned_axes)),

    ((slice(None), slice(1, 2)), cube_collection, NDCollection(
        [("cube0", cube0[:, :, 1:2]), ("cube1", cube1[1:2]), ("cube2", cube2[:, :, 1:2])],
        aligned_axes=aligned_axes)),

    ((slice(2, 4), slice(-3, -1)), cube_collection, NDCollection(
        [("cube0", cube0[:, 2:4, -3:-1]), ("cube1", cube1[-3:-1, :, 2:4]),
         ("cube2", cube2[:, 2:4, -3:-1])], aligned_axes=aligned_axes)),

    ((0, 0), cube_collection, NDCollection(
        [("cube0", cube0[:, 0, 0]), ("cube1", cube1[0, :, 0]), ("cube2", cube2[:, 0, 0])],
        aligned_axes=None)),

    (("cube0", "cube2"), cube_collection, NDCollection(
        [("cube0", cube0), ("cube2", cube2)], aligned_axes=(aligned_axes[0], aligned_axes[2]))),

    (0, seq_collection, NDCollection([("seq0", sequence02[0]), ("seq1", sequence20[0])],
                                     aligned_axes=((0, 1, 2), (0, 1, 2)))),

    ((slice(None), 1, slice(1, 3)), seq_collection,
        NDCollection([("seq0", sequence02[:, 1, 1:3]), ("seq1", sequence20[:, 1, 1:3])],
                     aligned_axes=((0, 1, 2), (0, 1, 2))))
])
def test_collection_slicing(item, collection, expected):
    helpers.assert_collections_equal(collection[item], expected)


@pytest.mark.parametrize("item,collection,expected", [("cube1", cube_collection, cube1)])
def test_slice_cube_from_collection(item, collection, expected):
    helpers.assert_cubes_equal(collection[item], expected)


def test_collection_copy():
    helpers.assert_collections_equal(cube_collection.copy(), cube_collection)


@pytest.mark.parametrize("collection,popped_key,expected_popped,expected_collection", [
    (cube_collection, "cube0", cube0, NDCollection([("cube1", cube1), ("cube2", cube2)],
                                                   aligned_axes=aligned_axes[1:]))])
def test_collection_pop(collection, popped_key, expected_popped, expected_collection):
    popped_collection = collection.copy()
    output = popped_collection.pop(popped_key)
    helpers.assert_cubes_equal(output, expected_popped)
    helpers.assert_collections_equal(popped_collection, expected_collection)


@pytest.mark.parametrize("collection,key,expected", [
    (cube_collection, "cube0", NDCollection([("cube1", cube1), ("cube2", cube2)],
                                            aligned_axes=aligned_axes[1:]))])
def test_del_collection(collection, key, expected):
    del_collection = collection.copy()
    del del_collection[key]
    helpers.assert_collections_equal(del_collection, expected)


@pytest.mark.parametrize("collection,key,data,aligned_axes,expected", [
    (cube_collection, "cube1", cube2, aligned_axes[2], NDCollection(
        [("cube0", cube0), ("cube1", cube2), ("cube2", cube2)],
        aligned_axes=((1, 2), (1, 2), (1, 2)))),

    (cube_collection, "cube3", cube2, aligned_axes[2], NDCollection(
        [("cube0", cube0), ("cube1", cube1), ("cube2", cube2), ("cube3", cube2)],
        aligned_axes=((1, 2), (2, 0), (1, 2), (1, 2))))])
def test_collection_update_key_data_pair_input(collection, key, data, aligned_axes, expected):
    updated_collection = collection.copy()
    updated_collection.update([(key, data)], aligned_axes)
    helpers.assert_collections_equal(updated_collection, expected)


def test_collection_update_collecton_input():
    orig_collection = NDCollection([("cube0", cube0), ("cube1", cube1)], aligned_axes[:2])
    cube1_alt = NDCube(data1*2, input_wcs1)
    new_collection = NDCollection([("cube1", cube1_alt), ("cube2", cube2)], aligned_axes[1:])
    orig_collection.update(new_collection)
    expected = NDCollection([("cube0", cube0), ("cube1", cube1_alt), ("cube2", cube2)],
                            aligned_axes)
    helpers.assert_collections_equal(orig_collection, expected)


@pytest.mark.parametrize("collection, expected_aligned_dimensions", [
    (cube_collection, [4, 5]*u.pix),
    (seq_collection, np.array([2*u.pix, 3*u.pix, 4*u.pix, 5*u.pix], dtype=object))])
def test_aligned_dimensions(collection, expected_aligned_dimensions):
    assert all(collection.aligned_dimensions == expected_aligned_dimensions)


@pytest.mark.parametrize("collection, expected", [
    (cube_collection, [('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
                       ('em.wl',)]),
    (seq_collection, [('meta.obs.sequence',),
                      ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
                      ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
                      ('em.wl',)])])
def test_aligned_axis_physical_types(collection, expected):
    output = collection.aligned_axis_physical_types
    print(output)
    assert len(output) == len(expected)
    for output_axis_types, expect_axis_types in zip(output, expected):
        assert set(output_axis_types) == set(expect_axis_types)
