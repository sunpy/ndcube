import astropy.units as u
import gwcs
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS

from ndcube import NDCube
from ndcube.extra_coords import ExtraCoords


@pytest.fixture
def time_lut():
    return Time(["2011-01-01T00:00:00",
                 "2011-01-01T00:00:10",
                 "2011-01-01T00:00:20",
                 "2011-01-01T00:00:30"], format="isot")


@pytest.fixture
def wave_lut():
    return range(10, 20) * u.nm


@pytest.fixture
def skycoord_1d_lut():
    return SkyCoord(range(10), range(10), unit=u.deg)


@pytest.fixture
def skycoord_2d_lut():
    data = np.arange(9).reshape(3,3), np.arange(9, 18).reshape(3,3)
    return SkyCoord(*data, unit=u.deg)


# Extra Coords from lookup tables

# A single lookup along a dimension, i.e. Time along the second dim.
def test_single_from_lut(wave_lut):
    ec = ExtraCoords.from_lookup_tables((10,), ("wave",), (0,), (wave_lut,))
    assert len(ec._lookup_tables) == 1
    assert ec.mapping == (0,)
    assert isinstance(ec.wcs, gwcs.WCS)
    assert ec.wcs.pixel_n_dim == 1
    assert ec.wcs.world_n_dim == 1
    assert ec.wcs.world_axis_names == ("wave",)


def test_two_1d_from_lut(time_lut):
    exposure_lut = range(10) * u.s
    ec = ExtraCoords.from_lookup_tables((10,), ("time", "exposure_time"),
                                        (0, 0), (time_lut, exposure_lut))
    assert len(ec._lookup_tables) == 2
    assert ec.mapping == (0, 0)
    assert isinstance(ec.wcs, gwcs.WCS)
    assert ec.wcs.pixel_n_dim == 2
    assert ec.wcs.world_n_dim == 2
    assert ec.wcs.world_axis_names == ("time", "exposure_time")


def test_skycoord(skycoord_1d_lut):
    ec = ExtraCoords.from_lookup_tables((10, 10), (("lat", "lon"),), ((0, 1),), (skycoord_1d_lut,))
    assert len(ec._lookup_tables) == 1
    assert ec.mapping == (1, 0)
    assert isinstance(ec.wcs, gwcs.WCS)
    assert ec.wcs.pixel_n_dim == 2
    assert ec.wcs.world_n_dim == 2
    assert ec.wcs.world_axis_names == ("lat", "lon")


def test_skycoord_mesh_false(skycoord_2d_lut):
    ec = ExtraCoords(array_shape=(10, 10))
    ec.add_coordinate(("lat", "lon"), (0, 1), skycoord_2d_lut, mesh=False)
    assert len(ec._lookup_tables) == 1
    assert ec.mapping == (1, 0)
    assert isinstance(ec.wcs, gwcs.WCS)
    assert ec.wcs.pixel_n_dim == 2
    assert ec.wcs.world_n_dim == 2
    assert ec.wcs.world_axis_names == ("lat", "lon")


def test_extra_coords_index(skycoord_2d_lut, time_lut):
    ec = ExtraCoords(array_shape=(10, 10))
    ec.add_coordinate(("lat", "lon"), (0, 1), skycoord_2d_lut, mesh=False)
    ec.add_coordinate("exposure_time", (0,), time_lut)
    assert len(ec._lookup_tables) == 2
    assert ec.mapping == (1, 0, 1)
    assert isinstance(ec.wcs, gwcs.WCS)
    assert ec.wcs.pixel_n_dim == 3
    assert ec.wcs.world_n_dim == 3
    assert ec.wcs.world_axis_names == ("lat", "lon", "exposure_time")

    sub_ec = ec["lon"]
    assert len(sub_ec._lookup_tables) == 1
    assert sub_ec.mapping == (1, 0)
    assert isinstance(ec.wcs, gwcs.WCS)
    assert sub_ec.wcs.pixel_n_dim == 2
    assert sub_ec.wcs.world_n_dim == 2
    assert sub_ec.wcs.world_axis_names == ("lat", "lon")

    sub_ec = ec["exposure_time"]
    assert len(sub_ec._lookup_tables) == 1
    assert sub_ec.mapping == (1,)
    assert isinstance(ec.wcs, gwcs.WCS)
    assert sub_ec.wcs.pixel_n_dim == 1
    assert sub_ec.wcs.world_n_dim == 1
    assert sub_ec.wcs.world_axis_names == ("exposure_time",)

# Extra coords from a WCS.

# Inspecting an extra coords
# Should be able to see what tables exists, what axes they account to, and what
# axes have missing dimensions.

# An additional spatial set (i.e. ICRS on top of HPC)


# Extra Coords with NDCube
def test_add_coord_after_create(time_lut):
    ndc = NDCube(np.random.random((10,10)), wcs=WCS(naxis=2))
    assert isinstance(ndc.extra_coords, ExtraCoords)
    ndc.extra_coords.add_coordinate("time", 0, time_lut)

    assert len(ndc.extra_coords._lookup_tables) == 1

    assert ndc.extra_coords["time"]._lookup_tables == ndc.extra_coords._lookup_tables

def test_combined_wcs(time_lut):
    ndc = NDCube(np.random.random((10, 10)), wcs=WCS(naxis=2))
    assert isinstance(ndc.extra_coords, ExtraCoords)
    ndc.extra_coords.add_coordinate("time", 0, time_lut)

    cwcs = ndc.combined_wcs
    assert cwcs.world_n_dim == 3
    assert cwcs.pixel_n_dim == 2
    world = cwcs.pixel_to_world(0, 0)
    assert u.allclose(world[:2], (1,1) * u.one)
    assert world[2] == Time("2011-01-01T00:00:00")

def test_slice_extra(time_lut):
    ndc = NDCube(np.random.random((10, 10)), wcs=WCS(naxis=2))
    assert isinstance(ndc.extra_coords, ExtraCoords)
    ndc.extra_coords.add_coordinate("time", 0, time_lut)

    print(ndc[:, 3].extra_coords._lookup_tables)
    # breakpoint()
