from unittest.mock import MagicMock

import astropy.units as u
import gwcs
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS

from ndcube import NDCube
from ndcube.extra_coords import ExtraCoords

# Fixtures


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
    data = np.arange(9).reshape(3, 3), np.arange(9, 18).reshape(3, 3)
    return SkyCoord(*data, unit=u.deg)


@pytest.fixture
def quantity_2d_lut():
    ec_shape = (3, 3)
    return np.arange(np.product(ec_shape)).reshape(ec_shape) * u.m / u.s


# ExtraCoords from WCS


def test_empty_ec(wcs_1d_l):
    ec = ExtraCoords()
    # Test slice of an empty EC
    assert ec[0].wcs is None

    assert ec.mapping == tuple()
    assert ec.wcs is None
    assert ec.keys() == tuple()

    ec.wcs = wcs_1d_l
    assert ec.wcs is wcs_1d_l
    ec.mapping = (0,)
    assert ec.mapping == (0,)


def test_exceptions(wcs_1d_l):
    # Test unable to specify inconsistent dimensions and tables
    with pytest.raises(ValueError):
        ExtraCoords.from_lookup_tables(None, (0,), (0, 0))

    # Test unable to add to WCS EC
    ec = ExtraCoords()
    ec.wcs = wcs_1d_l
    ec.mapping = (0,)

    with pytest.raises(ValueError):
        ec.add(None, 0, None)

    with pytest.raises(KeyError):
        ExtraCoords()['empty']


def test_mapping_setter(wcs_1d_l, wave_lut):
    ec = ExtraCoords()
    ec.wcs = wcs_1d_l
    ec.mapping = (0,)

    with pytest.raises(AttributeError):
        ec.mapping = None

    ec = ExtraCoords()
    ec.wcs = wcs_1d_l
    with pytest.raises(ValueError):
        ec.mapping = (1,)

    ec = ExtraCoords()
    ec.add("wave", (0,), wave_lut)
    with pytest.raises(AttributeError):
        ec.mapping = None


def test_wcs_setter(wcs_1d_l, wave_lut):
    ec = ExtraCoords()
    ec.wcs = wcs_1d_l
    ec.mapping = (0,)

    with pytest.raises(AttributeError):
        ec.wcs = None

    ec = ExtraCoords()
    ec.mapping = (1,)
    with pytest.raises(ValueError):
        ec.wcs = wcs_1d_l

    ec = ExtraCoords()
    ec.add("wave", (0,), wave_lut)
    with pytest.raises(AttributeError):
        ec.wcs = None


def test_wcs_1d(wcs_1d_l):
    ec = ExtraCoords()
    ec.wcs = wcs_1d_l
    ec.mapping = (0,)

    assert ec.keys() == ('spectral',)
    assert ec.mapping == (0,)
    assert ec.wcs is wcs_1d_l

    subec = ec[1:]
    assert ec.keys() == ('spectral',)
    assert ec.mapping == (0,)
    assert np.allclose(ec.wcs.pixel_to_world_values(1), subec.wcs.pixel_to_world_values(1))

    subec = ec[0]
    assert subec.wcs is None


@pytest.fixture
def extra_coords_wave(wave_lut):
    cube = MagicMock()
    cube.dimensions = [10] * u.pix
    ec = ExtraCoords(cube)
    ec.add("wave", 0, wave_lut)

    return ec


# Extra Coords from lookup tables

# A single lookup along a dimension, i.e. Time along the second dim.
def test_single_from_lut(extra_coords_wave):
    ec = extra_coords_wave
    assert len(ec._lookup_tables) == 1
    assert ec.mapping == (0,)
    assert isinstance(ec.wcs, gwcs.WCS)
    assert ec.wcs.pixel_n_dim == 1
    assert ec.wcs.world_n_dim == 1
    assert ec.wcs.world_axis_names == ("wave",)


def test_two_1d_from_lut(time_lut):
    cube = MagicMock()
    cube.dimensions = [10] * u.pix
    ec = ExtraCoords(cube)

    exposure_lut = range(10) * u.s
    ec.add("time", 0, time_lut)
    ec.add("exposure_time", 0, exposure_lut)
    assert len(ec._lookup_tables) == 2
    assert ec.mapping == (0, 0)
    assert isinstance(ec.wcs, gwcs.WCS)
    assert ec.wcs.pixel_n_dim == 2
    assert ec.wcs.world_n_dim == 2
    assert ec.wcs.world_axis_names == ("time", "exposure_time")


def test_two_1d_from_lookup_tables(time_lut):
    """
    Create ExtraCoords from both tables at once using `from_lookup_tables` with `physical_types`.
    """

    exposure_lut = range(10) * u.s

    pt = ["custom:time:creation"]
    with pytest.raises(ValueError, match=r"The number of physical types and lookup_tables"):
        ec = ExtraCoords.from_lookup_tables(["time", "exposure_time"], (0, 0),
                                            [time_lut, exposure_lut], pt)

    pt.append("custom:time:duration")
    ec = ExtraCoords.from_lookup_tables(["time", "exposure_time"], (0, 0),
                                        [time_lut, exposure_lut], pt)

    # This has created an "orphan" extra_coords with no NDCube connected.
    with pytest.raises(AttributeError, match=r"'NoneType' object has no attribute 'dimensions'"):
        assert ec.mapping == (0, 0)

    assert len(ec._lookup_tables) == 2
    assert isinstance(ec.wcs, gwcs.WCS)
    assert ec.wcs.pixel_n_dim == 2
    assert ec.wcs.world_n_dim == 2
    assert ec.wcs.world_axis_names == ("time", "exposure_time")
    for i, physical_types in enumerate(pt):
        assert ec._lookup_tables[i][1].physical_types == [physical_types]


def test_skycoord(skycoord_1d_lut):
    cube = MagicMock()
    cube.dimensions = [10, 10] * u.pix

    ec = ExtraCoords(cube)
    ec.add(("lat", "lon"), (0, 1), skycoord_1d_lut, mesh=True)
    assert len(ec._lookup_tables) == 1
    assert ec.mapping == (1, 0)
    assert isinstance(ec.wcs, gwcs.WCS)
    assert ec.wcs.pixel_n_dim == 2
    assert ec.wcs.world_n_dim == 2
    assert ec.wcs.world_axis_names == ("lat", "lon")


def test_skycoord_1_pixel(skycoord_1d_lut):
    cube = MagicMock()
    cube.dimensions = [10] * u.pix

    ec = ExtraCoords(cube)
    ec.add(("lon", "lat"), 0, skycoord_1d_lut, mesh=False)
    assert len(ec._lookup_tables) == 1
    assert ec.mapping == (0,)
    assert isinstance(ec.wcs, gwcs.WCS)
    assert ec.wcs.pixel_n_dim == 1
    assert ec.wcs.world_n_dim == 2
    assert ec.wcs.world_axis_names == ("lon", "lat")

    sec = ec[1:4]
    assert sec.wcs.pixel_n_dim == 1
    assert sec.wcs.world_n_dim == 2
    assert sec.wcs.world_axis_names == ("lon", "lat")

    assert isinstance(sec.wcs.pixel_to_world(0), SkyCoord)


def test_skycoord_mesh_false(skycoord_2d_lut):
    cube = MagicMock()
    cube.dimensions = [10, 10] * u.pix

    ec = ExtraCoords(cube)
    ec.add(("lat", "lon"), (0, 1), skycoord_2d_lut, mesh=False)
    assert len(ec._lookup_tables) == 1
    assert ec.mapping == (1, 0)
    assert isinstance(ec.wcs, gwcs.WCS)
    assert ec.wcs.pixel_n_dim == 2
    assert ec.wcs.world_n_dim == 2
    assert ec.wcs.world_axis_names == ("lat", "lon")


def test_extra_coords_index(skycoord_2d_lut, time_lut):
    cube = MagicMock()
    cube.dimensions = [10, 10] * u.pix

    ec = ExtraCoords(cube)
    ec.add(("lat", "lon"), (0, 1), skycoord_2d_lut, mesh=False)
    ec.add("exposure_time", (0,), time_lut)
    assert len(ec._lookup_tables) == 2
    assert ec.mapping == (1, 0, 1)
    assert isinstance(ec.wcs, gwcs.WCS)
    assert ec.wcs.pixel_n_dim == 3
    assert ec.wcs.world_n_dim == 3
    assert ec.wcs.world_axis_names == ("lat", "lon", "exposure_time")

    sub_ec = ec["lon"]
    sub_ec._ndcube = cube
    assert len(sub_ec._lookup_tables) == 1
    assert sub_ec.mapping == (1, 0)
    assert isinstance(ec.wcs, gwcs.WCS)
    assert sub_ec.wcs.pixel_n_dim == 2
    assert sub_ec.wcs.world_n_dim == 2
    assert sub_ec.wcs.world_axis_names == ("lat", "lon")

    sub_ec = ec["exposure_time"]
    sub_ec._ndcube = cube
    assert len(sub_ec._lookup_tables) == 1
    assert sub_ec.mapping == (1,)
    assert isinstance(ec.wcs, gwcs.WCS)
    assert sub_ec.wcs.pixel_n_dim == 1
    assert sub_ec.wcs.world_n_dim == 1
    assert sub_ec.wcs.world_axis_names == ("exposure_time",)


@pytest.mark.xfail(reason=">1D Tables not supported")
def test_extra_coords_2d_quantity(quantity_2d_lut):
    ec = ExtraCoords()
    ec.add("velocity", (0, 1), quantity_2d_lut)

    ec.wcs.pixel_to_world(0, 0)

# Inspecting an extra coords
# Should be able to see what tables exists, what axes they account to, and what
# axes have missing dimensions.

# An additional spatial set (i.e. ICRS on top of HPC)


# Extra Coords with NDCube
def test_add_coord_after_create(time_lut):
    ndc = NDCube(np.random.random((10, 10)), wcs=WCS(naxis=2))
    assert isinstance(ndc.extra_coords, ExtraCoords)
    ndc.extra_coords.add("time", 0, time_lut)

    assert len(ndc.extra_coords._lookup_tables) == 1

    assert ndc.extra_coords["time"]._lookup_tables == ndc.extra_coords._lookup_tables


def test_combined_wcs(time_lut):
    ndc = NDCube(np.random.random((10, 10)), wcs=WCS(naxis=2))
    assert isinstance(ndc.extra_coords, ExtraCoords)
    ndc.extra_coords.add("time", 0, time_lut)

    cwcs = ndc.combined_wcs
    assert cwcs.world_n_dim == 3
    assert cwcs.pixel_n_dim == 2
    world = cwcs.pixel_to_world(0, 0)
    assert u.allclose(world[:2], (1, 1) * u.one)
    assert world[2] == Time("2011-01-01T00:00:00")


def test_slice_extra_1d(time_lut, wave_lut):
    ec = ExtraCoords()
    ec.add("time", 0, time_lut)
    ec.add("wavey", 1, wave_lut)

    sec = ec[:, 3:7]
    assert len(sec._lookup_tables) == 2

    assert u.allclose(sec['wavey'].wcs.pixel_to_world_values(list(range(4))),
                      ec['wavey'].wcs.pixel_to_world_values(list(range(3, 7))))
    assert u.allclose(sec['time'].wcs.pixel_to_world_values(list(range(4))),
                      ec['time'].wcs.pixel_to_world_values(list(range(4))))


def test_slice_extra_2d(time_lut, skycoord_2d_lut):
    ec = ExtraCoords()
    ec.add(("lat", "lon"), (0, 1), skycoord_2d_lut, mesh=False)
    ec.add("exposure_time", (0,), time_lut)

    sec = ec[1:5, 1:5]
    assert len(sec._lookup_tables) == 2

    assert u.allclose(sec['lat'].wcs.pixel_to_world_values(list(range(2)), list(range(2))),
                      ec['lat'].wcs.pixel_to_world_values(list(range(1, 3)), list(range(1, 3))))
    assert u.allclose(sec['lon'].wcs.pixel_to_world_values(list(range(2)), list(range(2))),
                      ec['lon'].wcs.pixel_to_world_values(list(range(1, 3)), list(range(1, 3))))

    assert u.allclose(sec['exposure_time'].wcs.pixel_to_world_values(list(range(0, 3))),
                      ec['exposure_time'].wcs.pixel_to_world_values(list(range(1, 4))))


def test_slice_drop_dimensions(time_lut, skycoord_2d_lut):
    ec = ExtraCoords()
    ec.add(("lat", "lon"), (0, 1), skycoord_2d_lut, mesh=False)
    ec.add("exposure_time", (0,), time_lut)

    sec = ec[0, :]
    assert len(sec._lookup_tables) == 1

    assert u.allclose(sec['lat'].wcs.pixel_to_world_values(list(range(2))),
                      ec['lat'].wcs.pixel_to_world_values([0, 0], list(range(2))))
    assert u.allclose(sec['lon'].wcs.pixel_to_world_values(list(range(2))),
                      ec['lon'].wcs.pixel_to_world_values([0, 0], list(range(2))))

    sec = ec[:, 0]
    assert len(sec._lookup_tables) == 2

    assert u.allclose(sec['lat'].wcs.pixel_to_world_values(list(range(2))),
                      ec['lat'].wcs.pixel_to_world_values(list(range(2)), [0, 0]))
    assert u.allclose(sec['lon'].wcs.pixel_to_world_values(list(range(2))),
                      ec['lon'].wcs.pixel_to_world_values(list(range(2)), [0, 0]))

    assert u.allclose(sec['exposure_time'].wcs.pixel_to_world_values(list(range(2)), list(range(2))),
                      ec['exposure_time'].wcs.pixel_to_world_values(list(range(2)), list(range(2))))


def test_slice_extra_twice(time_lut, wave_lut):
    ec = ExtraCoords()
    ec.add("time", 0, time_lut)
    ec.add("wavey", 1, wave_lut)

    sec = ec[1:, 0]
    assert len(sec._lookup_tables) == 1

    assert u.allclose(sec['time'].wcs.pixel_to_world_values(list(range(0, 2))),
                      ec['time'].wcs.pixel_to_world_values(list(range(1, 3))))

    sec = sec[1:, 0]
    assert len(sec._lookup_tables) == 1

    assert u.allclose(sec['time'].wcs.pixel_to_world_values(list(range(0, 2))),
                      ec['time'].wcs.pixel_to_world_values(list(range(2, 4))))


def test_slice_extra_1d_drop(time_lut, wave_lut):
    ec = ExtraCoords()
    ec.add("time", 0, time_lut)
    ec.add("wavey", 1, wave_lut)

    sec = ec[:, 3]
    assert len(sec._lookup_tables) == 1

    assert u.allclose(sec['time'].wcs.pixel_to_world_values(list(range(4))),
                      ec['time'].wcs.pixel_to_world_values(list(range(4))))

    dwd = sec.dropped_world_dimensions
    dwd.pop("world_axis_object_classes")
    assert dwd
    assert dwd["world_axis_units"] == ["nm"]


def test_dropped_dimension_reordering():
    data = np.ones((3, 4, 5))
    wcs_input_dict = {
        'CTYPE1': 'WAVE    ', 'CUNIT1': 'Angstrom', 'CDELT1': 0.2, 'CRPIX1': 0, 'CRVAL1': 10, 'NAXIS1': 5,
        'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5, 'NAXIS2': 4,
        'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 3}
    input_wcs = WCS(wcs_input_dict)

    base_time = Time('2000-01-01', format='fits', scale='utc')
    timestamps = Time([base_time + TimeDelta(60 * i, format='sec') for i in range(data.shape[0])])

    my_cube = NDCube(data, input_wcs)
    my_cube.extra_coords.add('time', (0,), timestamps)

    # If the argument to extra_coords.add is array index then it should end up
    # in the first element of array_axis_physical_types
    assert "time" in my_cube.array_axis_physical_types[0]

    # When we slice out the dimension with the extra coord in it should go away.
    assert "time" not in my_cube[0].array_axis_physical_types[0]
