import astropy.units as u
import gwcs.coordinate_frames as cf
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from ndcube.extra_coords import LookupTableCoord


@pytest.fixture
def lut_1d_distance():
    lookup_table = u.Quantity(np.arange(10) * u.km)
    return LookupTableCoord(lookup_table)


def test_repr_str(lut_1d_distance):
    assert str(lut_1d_distance.delayed_models) in str(lut_1d_distance)
    assert str(lut_1d_distance.frames) in str(lut_1d_distance)
    assert str(lut_1d_distance) in repr(lut_1d_distance)

    assert str(lut_1d_distance.delayed_models[0]) in repr(lut_1d_distance.delayed_models[0])


def test_exceptions(lut_1d_distance):
    with pytest.raises(TypeError):
        LookupTableCoord(u.Quantity([1, 2, 3], u.nm), [1, 2, 3])

    with pytest.raises(TypeError):
        lut_1d_distance & list()

    # Test two Time
    with pytest.raises(ValueError):
        LookupTableCoord(Time("2011-01-01"), Time("2011-01-01"))

    # Test two SkyCoord
    with pytest.raises(ValueError):
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

    sub_ltc = lut_1d_distance[0:5]
    assert len(sub_ltc.delayed_models[0].lookup_table[0]) == 5


def test_3d_distance():
    lookup_table = (u.Quantity(np.arange(10) * u.km),
                    u.Quantity(np.arange(10, 20) * u.km),
                    u.Quantity(np.arange(20, 30) * u.km))

    ltc = LookupTableCoord(*lookup_table, mesh=True)
    assert ltc.model.n_inputs == 3
    assert ltc.model.n_outputs == 3

    assert ltc.wcs.world_n_dim == 3
    assert ltc.wcs.pixel_n_dim == 3

    assert u.allclose(ltc.wcs.pixel_to_world(0*u.pix, 0*u.pix, 0*u.pix),
                      (0, 10, 20)*u.km)
    assert u.allclose(ltc.wcs.world_to_pixel(0*u.km, 10*u.km, 20*u.km), (0, 0, 0))

    sub_ltc = ltc[0:5, 0:6, 0:7]
    assert len(sub_ltc.delayed_models[0].lookup_table[0]) == 5
    assert len(sub_ltc.delayed_models[0].lookup_table[1]) == 6
    assert len(sub_ltc.delayed_models[0].lookup_table[2]) == 7


def test_2d_nout_1_no_mesh():
    lookup_table = np.arange(9).reshape(3, 3) * u.km, np.arange(9, 18).reshape(3, 3) * u.km

    ltc = LookupTableCoord(*lookup_table, mesh=False)
    assert ltc.wcs.world_n_dim == 2
    assert ltc.wcs.pixel_n_dim == 2

    assert ltc.model.n_inputs == 2
    assert ltc.model.n_outputs == 2

    assert u.allclose(ltc.wcs.pixel_to_world(0*u.pix, 0*u.pix),
                      (0, 9)*u.km)

    # TODO: this model is not invertable
    # assert u.allclose(ltc.wcs.world_to_pixel(0*u.km, 9*u.km), (0, 0))

    sub_ltc = ltc[0:2, 0:2]
    assert sub_ltc.delayed_models[0].lookup_table[0].shape == (2, 2)
    assert sub_ltc.delayed_models[0].lookup_table[1].shape == (2, 2)


def test_1d_skycoord_no_mesh():
    sc = SkyCoord(range(10), range(10), unit=u.deg)
    ltc = LookupTableCoord(sc, mesh=False)
    assert ltc.model.n_inputs == 1
    assert ltc.model.n_outputs == 2

    sub_ltc = ltc[0:4]
    assert sub_ltc.delayed_models[0].lookup_table[0].shape == (4, )
    assert sub_ltc.delayed_models[0].lookup_table[1].shape == (4, )


def test_2d_skycoord_mesh():
    sc = SkyCoord(range(10), range(10), unit=u.deg)
    ltc = LookupTableCoord(sc, mesh=True)
    assert ltc.model.n_inputs == 2
    assert ltc.model.n_outputs == 2

    sub_ltc = ltc[0:4, 0:5]
    assert sub_ltc.delayed_models[0].lookup_table[0].shape == (4, )
    assert sub_ltc.delayed_models[0].lookup_table[1].shape == (5, )


@pytest.mark.xfail
def test_3d_skycoord_mesh():
    """Known failure due to gwcs#120."""
    sc = SkyCoord(range(10), range(10), range(10), unit=(u.deg, u.deg, u.AU))
    ltc = LookupTableCoord(sc, mesh=True)
    assert ltc.model.n_inputs == 3
    assert ltc.model.n_outputs == 3

    sub_ltc = ltc[0:4, 0:5, 0:6]
    assert sub_ltc.delayed_models[0].lookup_table[0].shape == (4, )
    assert sub_ltc.delayed_models[0].lookup_table[1].shape == (5, )
    assert sub_ltc.delayed_models[0].lookup_table[2].shape == (6, )


def test_2d_skycoord_no_mesh():
    data = np.arange(9).reshape(3, 3), np.arange(9, 18).reshape(3, 3)
    sc = SkyCoord(*data, unit=u.deg)
    ltc = LookupTableCoord(sc, mesh=False)
    assert ltc.model.n_inputs == 2
    assert ltc.model.n_outputs == 2

    sub_ltc = ltc[1:3, 1:2]
    assert sub_ltc.delayed_models[0].lookup_table[0].shape == (2, 1)
    assert sub_ltc.delayed_models[0].lookup_table[1].shape == (2, 1)


def test_1d_time():
    data = Time(["2011-01-01T00:00:00",
                 "2011-01-01T00:00:10",
                 "2011-01-01T00:00:20",
                 "2011-01-01T00:00:30"], format="isot")
    ltc = LookupTableCoord(data)
    assert ltc.model.n_inputs == 1
    assert ltc.model.n_outputs == 1
    assert u.allclose(ltc.model.lookup_table, u.Quantity((0, 10, 20, 30), u.s))

    assert ltc.wcs.pixel_to_world(0) == Time("2011-01-01T00:00:00")
    assert ltc.wcs.world_to_pixel(Time("2011-01-01T00:00:00")) == 0

    sub_ltc = ltc[1:3]
    assert sub_ltc.delayed_models[0].lookup_table.shape == (2,)


def test_join():
    time_ltc = LookupTableCoord(Time(["2011-01-01T00:00:00",
                                      "2011-01-01T00:00:10",
                                      "2011-01-01T00:00:20",
                                      "2011-01-01T00:00:30"], format="isot"))

    wave_ltc = LookupTableCoord(range(10) * u.nm)

    ltc = time_ltc & wave_ltc

    assert ltc.model.n_inputs == 2
    assert ltc.model.n_outputs == 2

    assert isinstance(ltc.frame, cf.CompositeFrame)
    world = ltc.wcs.pixel_to_world(0, 0)
    assert world[0] == Time("2011-01-01T00:00:00")
    assert u.allclose(world[1], 0 * u.nm)

    assert u.allclose(ltc.wcs.world_to_pixel(*world), (0, 0))

    sub_ltc = ltc[1:3, 1:3]
    assert len(sub_ltc.delayed_models) == 2
    assert sub_ltc.delayed_models[0].lookup_table.shape == (2,)
    assert sub_ltc.delayed_models[1].lookup_table[0].shape == (2,)

    sub_ltc = ltc[1:3, 2]
    assert len(sub_ltc.delayed_models) == 1
    assert sub_ltc.delayed_models[0].lookup_table.shape == (2,)


def test_join_3d():
    sc = SkyCoord(range(10), range(10), unit=u.deg)
    space_ltc = LookupTableCoord(sc, mesh=True)
    wave_ltc = LookupTableCoord(range(10) * u.nm)

    ltc = space_ltc & wave_ltc

    assert ltc.model.n_inputs == 3
    assert ltc.model.n_outputs == 3

    assert isinstance(ltc.frame, cf.CompositeFrame)
    world = ltc.wcs.pixel_to_world(0, 0, 0)
    assert isinstance(world[0], SkyCoord)
    assert u.allclose(world[1], 0 * u.nm)

    assert u.allclose(ltc.wcs.world_to_pixel(*world), (0, 0, 0))


def test_2d_quantity():
    shape = (3, 3)
    data = np.arange(np.product(shape)).reshape(shape) * u.m / u.s

    ltc = LookupTableCoord(data)
    assert u.allclose(ltc.wcs.pixel_to_world(0, 0), 0 * u.m / u.s)
