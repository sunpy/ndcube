import astropy.units as u
import gwcs.coordinate_frames as cf
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from ndcube.extra_coords import LookupTableCoord
from ndcube.extra_coords.lookup_table_coord import (QuantityTableCoordinate,
                                                    SkyCoordTableCoordinate, TimeTableCoordinate)


@pytest.fixture
def lut_1d_distance():
    lookup_table = u.Quantity(np.arange(10) * u.km)
    return LookupTableCoord(lookup_table)


@pytest.fixture
def lut_3d_distance_mesh():
    lookup_table = (u.Quantity(np.arange(10) * u.km),
                    u.Quantity(np.arange(10, 20) * u.km),
                    u.Quantity(np.arange(20, 30) * u.km))

    return LookupTableCoord(*lookup_table, mesh=True)


@pytest.fixture
def lut_2d_distance_no_mesh():
    lookup_table = np.arange(9).reshape(3, 3) * u.km, np.arange(9, 18).reshape(3, 3) * u.km
    return LookupTableCoord(*lookup_table, mesh=False)


@pytest.fixture
def lut_1d_skycoord_no_mesh():
    sc = SkyCoord(range(10), range(10), unit=u.deg)
    return LookupTableCoord(sc, mesh=False)


@pytest.fixture
def lut_2d_skycoord_no_mesh():
    data = np.arange(9).reshape(3, 3), np.arange(9, 18).reshape(3, 3)
    sc = SkyCoord(*data, unit=u.deg)
    return LookupTableCoord(sc, mesh=False)


@pytest.fixture
def lut_2d_skycoord_mesh():
    sc = SkyCoord(range(10), range(10), unit=u.deg)
    return LookupTableCoord(sc, mesh=True)


@pytest.fixture
def lut_3d_skycoord_mesh():
    sc = SkyCoord(range(10), range(10), range(10), unit=(u.deg, u.deg, u.AU))
    return LookupTableCoord(sc, mesh=True)


@pytest.fixture
def lut_1d_time():
    data = Time(["2011-01-01T00:00:00",
                 "2011-01-01T00:00:10",
                 "2011-01-01T00:00:20",
                 "2011-01-01T00:00:30"], format="isot")
    return LookupTableCoord(data)


@pytest.fixture
def lut_1d_wave():
    # TODO: Make this into a SpectralCoord object
    return LookupTableCoord(range(10) * u.nm)


def test_exceptions(lut_1d_distance):
    with pytest.raises(TypeError):
        LookupTableCoord(u.Quantity([1, 2, 3], u.nm), [1, 2, 3])

    with pytest.raises(TypeError):
        lut_1d_distance & list()

    # Test two Time
    with pytest.raises(TypeError):
        LookupTableCoord(Time("2011-01-01"), Time("2011-01-01"))

    # Test two SkyCoord
    with pytest.raises(TypeError):
        LookupTableCoord(SkyCoord(10, 10, unit=u.deg), SkyCoord(10, 10, unit=u.deg))

    # Test not matching units
    with pytest.raises(u.UnitsError):
        LookupTableCoord(u.Quantity([1, 2, 3], u.nm), u.Quantity([1, 2, 3], u.s))


def test_1d_distance(lut_1d_distance):
    assert lut_1d_distance.model.n_inputs == 1
    assert lut_1d_distance.model.n_outputs == 1
    assert lut_1d_distance.model.lookup_table.shape == (10,)
    assert u.allclose(u.Quantity(range(10), u.pix), lut_1d_distance.model.points)

    assert u.allclose(lut_1d_distance.wcs.pixel_to_world(0), 0 * u.km)
    assert u.allclose(lut_1d_distance.wcs.pixel_to_world(9), 9 * u.km)
    assert lut_1d_distance.wcs.world_to_pixel(0 * u.km) == 0


def test_3d_distance(lut_3d_distance_mesh):
    ltc = lut_3d_distance_mesh

    assert ltc.model.n_inputs == 3
    assert ltc.model.n_outputs == 3

    assert ltc.wcs.world_n_dim == 3
    assert ltc.wcs.pixel_n_dim == 3

    assert u.allclose(ltc.wcs.pixel_to_world(0*u.pix, 0*u.pix, 0*u.pix),
                      (0, 10, 20)*u.km)
    assert u.allclose(ltc.wcs.world_to_pixel(0*u.km, 10*u.km, 20*u.km), (0, 0, 0))


def test_2d_nout_1_no_mesh(lut_2d_distance_no_mesh):
    ltc = lut_2d_distance_no_mesh
    assert ltc.wcs.world_n_dim == 2
    assert ltc.wcs.pixel_n_dim == 2

    assert ltc.model.n_inputs == 2
    assert ltc.model.n_outputs == 2

    assert u.allclose(ltc.wcs.pixel_to_world(0*u.pix, 0*u.pix),
                      (0, 9)*u.km)

    # TODO: this model is not invertable
    # assert u.allclose(ltc.wcs.world_to_pixel(0*u.km, 9*u.km), (0, 0))


def test_1d_skycoord_no_mesh(lut_1d_skycoord_no_mesh):
    ltc = lut_1d_skycoord_no_mesh

    assert ltc.model.n_inputs == 1
    assert ltc.model.n_outputs == 2


def test_2d_skycoord_mesh(lut_2d_skycoord_mesh):
    ltc = lut_2d_skycoord_mesh
    assert ltc.model.n_inputs == 2
    assert ltc.model.n_outputs == 2


def test_3d_skycoord_mesh(lut_3d_skycoord_mesh):
    ltc = lut_3d_skycoord_mesh

    assert ltc.model.n_inputs == 3
    assert ltc.model.n_outputs == 3

    # Known failure due to gwcs#120

    # assert isinstance(ltc.wcs, gwcs.WCS)
    #
    # sub_ltc = ltc[0:4, 0:5, 0:6]
    # assert sub_ltc.delayed_models[0].lookup_table[0].shape == (4, )
    # assert sub_ltc.delayed_models[0].lookup_table[1].shape == (5, )
    # assert sub_ltc.delayed_models[0].lookup_table[2].shape == (6, )


def test_2d_skycoord_no_mesh(lut_2d_skycoord_no_mesh):
    ltc = lut_2d_skycoord_no_mesh

    assert ltc.model.n_inputs == 2
    assert ltc.model.n_outputs == 2


def test_1d_time(lut_1d_time):
    assert lut_1d_time.model.n_inputs == 1
    assert lut_1d_time.model.n_outputs == 1
    assert u.allclose(lut_1d_time.model.lookup_table, u.Quantity((0, 10, 20, 30), u.s))

    assert lut_1d_time.wcs.pixel_to_world(0) == Time("2011-01-01T00:00:00")
    assert lut_1d_time.wcs.world_to_pixel(Time("2011-01-01T00:00:00")) == 0


def test_join(lut_1d_time, lut_1d_wave):
    ltc = lut_1d_time & lut_1d_wave

    assert ltc.model.n_inputs == 2
    assert ltc.model.n_outputs == 2

    assert isinstance(ltc.frame, cf.CompositeFrame)
    world = ltc.wcs.pixel_to_world(0, 0)
    assert world[0] == Time("2011-01-01T00:00:00")
    assert u.allclose(world[1], 0 * u.nm)

    assert u.allclose(ltc.wcs.world_to_pixel(*world), (0, 0))


def test_join_3d(lut_2d_skycoord_mesh, lut_1d_wave):
    ltc = lut_2d_skycoord_mesh & lut_1d_wave

    assert ltc.model.n_inputs == 3
    assert ltc.model.n_outputs == 3

    assert isinstance(ltc.frame, cf.CompositeFrame)
    world = ltc.wcs.pixel_to_world(0, 0, 0)
    assert isinstance(world[0], SkyCoord)
    assert u.allclose(world[1], 0 * u.nm)

    # TODO: Investigate this, something about inverse model
    # assert u.allclose(ltc.wcs.world_to_pixel(*world), (0, 0, 0))


def test_2d_quantity():
    shape = (3, 3)
    data = np.arange(np.product(shape)).reshape(shape) * u.m / u.s

    ltc = LookupTableCoord(data)
    assert u.allclose(ltc.wcs.pixel_to_world(0, 0), 0 * u.m / u.s)


################################################################################
# Slicing Tests
################################################################################


def test_slicing_quantity_table_coordinate():
    qtc = QuantityTableCoordinate(range(10)*u.m, mesh=False)

    assert u.allclose(qtc[2:8].table[0], range(2, 8)*u.m)
    assert u.allclose(qtc[2].table[0], 2*u.m)

    qtc = QuantityTableCoordinate(range(10)*u.m, mesh=True)

    assert u.allclose(qtc[2:8].table[0], range(2, 8)*u.m)
    assert u.allclose(qtc[2].table[0], 2*u.m)

    qtc = QuantityTableCoordinate(*np.mgrid[0:10, 0:10]*u.m, mesh=False)

    assert u.allclose(qtc[2:8, 2:8].table[0], (np.mgrid[2:8, 2:8]*u.m)[0])
    assert u.allclose(qtc[2:8, 2:8].table[1], (np.mgrid[2:8, 2:8]*u.m)[1])

    assert u.allclose(qtc[2, 2:8].table[0], 2*u.m)
    assert u.allclose(qtc[2, 2:8].table[1], (np.mgrid[2:8, 2:8]*u.m)[1])

    qtc = QuantityTableCoordinate(range(10)*u.m, range(10)*u.m, mesh=True)
    assert u.allclose(qtc[2:8, 2:8].table[0], range(2, 8)*u.m)
    assert u.allclose(qtc[2:8, 2:8].table[1], range(2, 8)*u.m)

    assert u.allclose(qtc[2, 2:8].table[0], 2*u.m)


def _assert_skycoord_equal(sc1, sc2):
    sc2 = sc2.transform_to(sc1.frame)

    assert sc1.shape == sc2.shape

    components1 = tuple(getattr(sc1.data, comp) for comp in sc1.data.components)
    components2 = tuple(getattr(sc2.data, comp) for comp in sc2.data.components)

    for c1, c2 in zip(components1, components2):
        assert u.allclose(c1, c2)


def test_slicing_skycoord_table_coordinate():
    # 1D, no mesh
    sc = SkyCoord(range(10)*u.deg, range(10)*u.deg)
    stc = SkyCoordTableCoordinate(sc, mesh=False)

    _assert_skycoord_equal(stc[2:8].table, sc[2:8])
    _assert_skycoord_equal(stc[2].table, sc[2])

    # 2D, no mesh
    sc = SkyCoord(*np.mgrid[0:10, 0:10]*u.deg)
    stc = SkyCoordTableCoordinate(sc, mesh=False)
    _assert_skycoord_equal(stc[2:8, 2:8].table, sc[2:8, 2:8])
    _assert_skycoord_equal(stc[2, 2:8].table, sc[2, 2:8])

    # 2D with mesh
    # When mesh is True the constructor will run meshgrid
    sc = SkyCoord(*u.Quantity(np.meshgrid(range(10), range(10)), u.deg))
    stc = SkyCoordTableCoordinate(SkyCoord(range(10), range(10), unit=u.deg), mesh=True)

    _assert_skycoord_equal(stc.table, sc)

    _assert_skycoord_equal(stc[2:8, 2:8].table, sc[2:8, 2:8])
    _assert_skycoord_equal(stc[2, 2:8].table, sc[2, 2:8])


def test_slicing_time_table_coordinate():
    data = Time(["2011-01-01T00:00:00",
                 "2011-01-01T00:00:10",
                 "2011-01-01T00:00:20",
                 "2011-01-01T00:00:30"], format="isot")

    ttc = TimeTableCoordinate(data)
    assert (ttc.table == data).all()

    assert (ttc[2:8].table == data[2:8]).all()
    assert ttc[2].table == data[2]


def test_1d_distance_slice(lut_1d_distance):
    sub_ltc = lut_1d_distance[0:5]
    assert len(sub_ltc._lookup_tables[0].table[0]) == 5


def test_3d_distance_slice(lut_3d_distance_mesh):
    sub_ltc = lut_3d_distance_mesh[0:5, 0:6, 0:7]
    assert len(sub_ltc._lookup_tables[0].table[0]) == 5
    assert len(sub_ltc._lookup_tables[0].table[1]) == 6
    assert len(sub_ltc._lookup_tables[0].table[2]) == 7

    # sub_ltc = ltc[0]

    # assert ltc.wcs.world_n_dim == 2
    # assert ltc.wcs.pixel_n_dim == 2


def test_2d_nout_1_no_mesh_slice(lut_2d_distance_no_mesh):
    ltc = lut_2d_distance_no_mesh
    sub_ltc = ltc[0:2, 0:2]
    assert sub_ltc._lookup_tables[0].table[0].shape == (2, 2)
    assert sub_ltc._lookup_tables[0].table[1].shape == (2, 2)

    sub_ltc = ltc[0]

    assert ltc.wcs.world_n_dim == 2
    assert ltc.wcs.pixel_n_dim == 2


def test_1d_skycoord_no_mesh_slice(lut_1d_skycoord_no_mesh):
    sub_ltc = lut_1d_skycoord_no_mesh[0:4]
    assert sub_ltc._lookup_tables[0].table.shape == (4, )
    assert sub_ltc._lookup_tables[0].table.shape == (4, )


def test_2d_skycoord_mesh_slice(lut_2d_skycoord_mesh):
    sub_ltc = lut_2d_skycoord_mesh[0:4, 0:5]
    assert sub_ltc._lookup_tables[0].table.shape == (4, 5)


def test_2d_skycoord_no_mesh_slice(lut_2d_skycoord_no_mesh):
    sub_ltc = lut_2d_skycoord_no_mesh[1:3, 1:2]
    assert sub_ltc._lookup_tables[0].table.shape == (2, 1)


def test_1d_time_slice(lut_1d_time):
    sub_ltc = lut_1d_time[1:3]
    assert sub_ltc._lookup_tables[0].table.shape == (2,)


def test_join_slice(lut_1d_time, lut_1d_wave):
    ltc = lut_1d_time & lut_1d_wave

    sub_ltc = ltc[1:3, 1:3]
    assert len(sub_ltc._lookup_tables) == 2
    assert sub_ltc._lookup_tables[0].table.shape == (2,)
    assert sub_ltc._lookup_tables[1].table[0].shape == (2,)

    sub_ltc = ltc[1:3, 2]
    assert len(sub_ltc._lookup_tables) == 1
    assert sub_ltc._lookup_tables[0].table.shape == (2,)


# def test_dropped_world_1(lut_1d_time, lut_1d_wave):
#     ltc = lut_1d_time & lut_1d_wave

#     assert isinstance(ltc.dropped_word_dimensions, dict)
#     assert len(ltc.dropped_word_dimensions) == 0

#     sub_ltc = ltc[0]
#     assert isinstance(sub_ltc.dropped_word_dimensions, dict)
#     assert len(sub_ltc.dropped_word_dimensions) == 1
