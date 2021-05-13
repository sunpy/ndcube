import astropy.units as u
import gwcs.coordinate_frames as cf
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from ndcube.extra_coords.table_coord import (MultipleTableCoordinate, QuantityTableCoordinate,
                                             SkyCoordTableCoordinate, TimeTableCoordinate)


@pytest.fixture
def lut_1d_distance():
    lookup_table = u.Quantity(np.arange(10) * u.km)
    return QuantityTableCoordinate(lookup_table, names='x')


@pytest.fixture
def lut_3d_distance_mesh():
    lookup_table = (u.Quantity(np.arange(10) * u.km),
                    u.Quantity(np.arange(10, 20) * u.km),
                    u.Quantity(np.arange(20, 30) * u.km))

    return QuantityTableCoordinate(*lookup_table, mesh=True, names=['x', 'y', 'z'])


@pytest.fixture
def lut_2d_distance_no_mesh():
    lookup_table = np.arange(9).reshape(3, 3) * u.km, np.arange(9, 18).reshape(3, 3) * u.km
    return QuantityTableCoordinate(*lookup_table, mesh=False)


@pytest.fixture
def lut_1d_skycoord_no_mesh():
    sc = SkyCoord(range(10), range(10), unit=u.deg)
    return SkyCoordTableCoordinate(sc, mesh=False, names=['lon', 'lat'])


@pytest.fixture
def lut_2d_skycoord_no_mesh():
    data = np.arange(9).reshape(3, 3), np.arange(9, 18).reshape(3, 3)
    sc = SkyCoord(*data, unit=u.deg)
    return SkyCoordTableCoordinate(sc, mesh=False)


@pytest.fixture
def lut_2d_skycoord_mesh():
    sc = SkyCoord(range(10), range(10), unit=u.deg)
    return SkyCoordTableCoordinate(sc, mesh=True)


@pytest.fixture
def lut_3d_skycoord_mesh():
    sc = SkyCoord(range(10), range(10), range(10), unit=(u.deg, u.deg, u.AU))
    return SkyCoordTableCoordinate(sc, mesh=True)


@pytest.fixture
def lut_1d_time():
    data = Time(["2011-01-01T00:00:00",
                 "2011-01-01T00:00:10",
                 "2011-01-01T00:00:20",
                 "2011-01-01T00:00:30"], format="isot")
    return TimeTableCoordinate(data, names='time', physical_types='time')


@pytest.fixture
def lut_1d_wave():
    # TODO: Make this into a SpectralCoord object
    return QuantityTableCoordinate(range(10) * u.nm)


def test_exceptions():
    with pytest.raises(TypeError) as ei:
        QuantityTableCoordinate(u.Quantity([1, 2, 3], u.nm), [1, 2, 3])
    assert "All tables must be astropy Quantity objects" in str(ei)

    with pytest.raises(u.UnitsError) as ei:
        QuantityTableCoordinate(u.Quantity([1, 2, 3], u.nm), [1, 2, 3] * u.deg)
    assert "All tables must have equivalent units." in str(ei)

    with pytest.raises(ValueError) as ei:
        QuantityTableCoordinate(u.Quantity([1, 2, 3], u.nm), [1, 2, 3] * u.m, names='x')
    assert "The number of names should match the number of world dimensions" in str(ei)

    with pytest.raises(ValueError) as ei:
        QuantityTableCoordinate(u.Quantity([1, 2, 3], u.nm), [1, 2, 3] * u.m, physical_types='x')
    assert "The number of physical types should match the number of world dimensions" in str(ei)

    # Test two Time
    with pytest.raises(ValueError) as ei:
        TimeTableCoordinate(Time("2011-01-01"), Time("2011-01-01"))
    assert "single Time object" in str(ei)

    with pytest.raises(ValueError) as ei:
        TimeTableCoordinate(Time("2011-01-01"), names=['a', 'b'])
    assert "only have one name." in str(ei)

    with pytest.raises(ValueError) as ei:
        TimeTableCoordinate(Time("2011-01-01"), physical_types=['a', 'b'])
    assert "only have one physical type." in str(ei)

    # Test two SkyCoord
    with pytest.raises(ValueError) as ei:
        SkyCoordTableCoordinate(SkyCoord(10, 10, unit=u.deg), SkyCoord(10, 10, unit=u.deg))
    assert "single SkyCoord object" in str(ei)

    with pytest.raises(ValueError) as ei:
        SkyCoordTableCoordinate(SkyCoord(10, 10, unit=u.deg), names='x')
    assert "names must equal two" in str(ei)

    with pytest.raises(ValueError) as ei:
        SkyCoordTableCoordinate(SkyCoord(10, 10, unit=u.deg), physical_types='x')
    assert "physical types must equal two" in str(ei)

    with pytest.raises(TypeError) as ei:
        MultipleTableCoordinate(10, SkyCoordTableCoordinate(SkyCoord(10, 10, unit=u.deg)))
    assert "All arguments must be BaseTableCoordinate" in str(ei)

    with pytest.raises(TypeError) as ei:
        MultipleTableCoordinate(MultipleTableCoordinate(SkyCoordTableCoordinate(SkyCoord(10, 10, unit=u.deg))))
    assert "All arguments must be BaseTableCoordinate" in str(ei)


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


@pytest.mark.xfail(reason=">1D Tables not supported")
def test_2d_nout_1_no_mesh(lut_2d_distance_no_mesh):
    ltc = lut_2d_distance_no_mesh
    assert ltc.wcs.world_n_dim == 2
    assert ltc.wcs.pixel_n_dim == 2

    assert ltc.model.n_inputs == 2
    assert ltc.model.n_outputs == 2

    assert u.allclose(ltc.wcs.pixel_to_world(0*u.pix, 0*u.pix),
                      (0, 9)*u.km)

    # TODO: this model is not invertable
    assert u.allclose(ltc.wcs.world_to_pixel(0*u.km, 9*u.km), (0, 0))


def test_1d_skycoord_no_mesh(lut_1d_skycoord_no_mesh):
    ltc = lut_1d_skycoord_no_mesh

    assert ltc.model.n_inputs == 1
    assert ltc.model.n_outputs == 2

    pixel_coords = (0,)*u.pix
    sc = ltc.wcs.pixel_to_world(*pixel_coords)
    pix = ltc.wcs.world_to_pixel(sc)
    assert u.allclose(pix, pixel_coords.value)


def test_2d_skycoord_mesh(lut_2d_skycoord_mesh):
    ltc = lut_2d_skycoord_mesh
    assert ltc.model.n_inputs == 2
    assert ltc.model.n_outputs == 2

    pixel_coords = (0, 0)*u.pix
    sc = ltc.wcs.pixel_to_world(*pixel_coords)
    pix = ltc.wcs.world_to_pixel(sc)
    assert u.allclose(pix, pixel_coords.value)


def test_3d_skycoord_mesh(lut_3d_skycoord_mesh):
    ltc = lut_3d_skycoord_mesh

    assert ltc.model.n_inputs == 3
    assert ltc.model.n_outputs == 3

    # Known failure due to gwcs#120

    # pixel_coords = (0, 0, 0)*u.pix
    # sc = ltc.wcs.pixel_to_world(*pixel_coords)
    # pix = ltc.wcs.world_to_pixel(sc)
    # assert u.allclose(pix, pixel_coords.value)

    # assert isinstance(ltc.wcs, gwcs.WCS)
    #
    # sub_ltc = ltc[0:4, 0:5, 0:6]
    # assert sub_ltc.delayed_models[0].lookup_table[0].shape == (4, )
    # assert sub_ltc.delayed_models[0].lookup_table[1].shape == (5, )
    # assert sub_ltc.delayed_models[0].lookup_table[2].shape == (6, )


@pytest.mark.xfail(reason=">1D Tables not supported")
def test_2d_skycoord_no_mesh(lut_2d_skycoord_no_mesh):
    ltc = lut_2d_skycoord_no_mesh

    assert ltc.model.n_inputs == 2
    assert ltc.model.n_outputs == 2

    pixel_coords = (0, 0)*u.pix
    sc = ltc.wcs.pixel_to_world(*pixel_coords)
    pix = ltc.wcs.world_to_pixel(sc)
    assert u.allclose(pix, pixel_coords.value)


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

    pixel_coords = (0, 0, 0)*u.pix
    sc = ltc.wcs.pixel_to_world(*pixel_coords)
    pix = ltc.wcs.world_to_pixel(*sc)
    assert u.allclose(pix, pixel_coords.value)

    assert u.allclose(ltc.wcs.world_to_pixel(*world), (0, 0, 0))


@pytest.mark.xfail(reason=">1D Tables not supported")
def test_2d_quantity():
    shape = (3, 3)
    data = np.arange(np.product(shape)).reshape(shape) * u.m / u.s

    ltc = QuantityTableCoordinate(data)
    assert u.allclose(ltc.wcs.pixel_to_world(0, 0), 0 * u.m / u.s)


def test_repr_str(lut_1d_time, lut_1d_wave):
    assert str(lut_1d_time.table) in str(lut_1d_time)
    assert "TimeTableCoordinate" in repr(lut_1d_time)

    join = lut_1d_time & lut_1d_wave
    assert str(lut_1d_time.table) in str(join)
    assert str(lut_1d_wave.table) in str(join)
    assert "TimeTableCoordinate" not in repr(join)
    assert "MultipleTableCoordinate" in repr(join)


################################################################################
# Slicing Tests
################################################################################


def test_slicing_quantity_table_coordinate():
    qtc = QuantityTableCoordinate(range(10)*u.m, mesh=False, names='x', physical_types='pos:x')

    assert u.allclose(qtc[2:8].table[0], range(2, 8)*u.m)
    assert u.allclose(qtc[2].table[0], 2*u.m)
    assert qtc.names == ['x']
    assert qtc.physical_types == ['pos:x']

    qtc = QuantityTableCoordinate(range(10)*u.m, mesh=True)

    assert u.allclose(qtc[2:8].table[0], range(2, 8)*u.m)
    assert u.allclose(qtc[2].table[0], 2*u.m)

    qtc = QuantityTableCoordinate(range(10)*u.m, range(10)*u.m, mesh=True,
                                  names=['x', 'y'], physical_types=['pos:x', 'pos:y'])
    assert u.allclose(qtc[2:8, 2:8].table[0], range(2, 8)*u.m)
    assert u.allclose(qtc[2:8, 2:8].table[1], range(2, 8)*u.m)

    # we have dropped one dimension
    assert len(qtc[2, 2:8].table) == 1
    assert u.allclose(qtc[2, 2:8].table[0], range(2, 8)*u.m)

    assert qtc.names == ['x', 'y']
    assert qtc.physical_types == ['pos:x', 'pos:y']

    assert qtc.frame.axes_names == ('x', 'y')
    assert qtc.frame.axis_physical_types == ('custom:pos:x', 'custom:pos:y')


@pytest.mark.xfail(reason=">1D Tables not supported")
def test_slicing_quantity_table_coordinate_2d():
    qtc = QuantityTableCoordinate(*np.mgrid[0:10, 0:10]*u.m, mesh=False,
                                  names=['x', 'y'], physical_types=['pos:x', 'pos:y'])

    assert u.allclose(qtc[2:8, 2:8].table[0], (np.mgrid[2:8, 2:8]*u.m)[0])
    assert u.allclose(qtc[2:8, 2:8].table[1], (np.mgrid[2:8, 2:8]*u.m)[1])
    assert qtc.names == ['x', 'y']
    assert qtc.physical_types == ['pos:x', 'pos:y']

    assert qtc.frame.axes_names == ('x', 'y')
    assert qtc.frame.axis_physical_types == ('custom:pos:x', 'custom:pos:y')

    assert u.allclose(qtc[2, 2:8].table[0], 2*u.m)
    assert u.allclose(qtc[2, 2:8].table[1], (np.mgrid[2:8, 2:8]*u.m)[1])


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
    stc = SkyCoordTableCoordinate(sc, mesh=False, names=['lon', 'lat'], physical_types=['pos:x', 'pos:y'])

    _assert_skycoord_equal(stc[2:8].table, sc[2:8])
    _assert_skycoord_equal(stc[2].table, sc[2])
    assert stc.names == ['lon', 'lat']
    assert stc.physical_types == ['pos:x', 'pos:y']

    assert stc.frame.axes_names == ('lon', 'lat')
    assert stc.frame.axis_physical_types == ('custom:pos:x', 'custom:pos:y')

    # 2D, no mesh
    sc = SkyCoord(*np.mgrid[0:10, 0:10]*u.deg)
    stc = SkyCoordTableCoordinate(sc, mesh=False)
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
    assert len(sub_ltc.table[0]) == 5


def test_3d_distance_slice(lut_3d_distance_mesh):
    sub_ltc = lut_3d_distance_mesh[0:5, 0:6, 0:7]
    assert len(sub_ltc.table[0]) == 5
    assert len(sub_ltc.table[1]) == 6
    assert len(sub_ltc.table[2]) == 7


@pytest.mark.xfail(reason=">1D Tables not supported")
def test_2d_nout_1_no_mesh_slice(lut_2d_distance_no_mesh):
    ltc = lut_2d_distance_no_mesh
    sub_ltc = ltc[0:2, 0:2]
    assert sub_ltc.table[0].shape == (2, 2)
    assert sub_ltc.table[1].shape == (2, 2)


def test_1d_skycoord_no_mesh_slice(lut_1d_skycoord_no_mesh):
    sub_ltc = lut_1d_skycoord_no_mesh[0:4]
    assert sub_ltc.table.shape == (4, )
    assert sub_ltc.table.shape == (4, )


@pytest.mark.xfail(reason=">1D Tables not supported")
def test_2d_skycoord_mesh_slice(lut_2d_skycoord_mesh):
    sub_ltc = lut_2d_skycoord_mesh[4:10, 5:10]
    assert sub_ltc.table.shape == (10,)
    assert sub_ltc._slice == [slice(4, 10, None), slice(5, 10, None)]

    assert sub_ltc.wcs.world_to_pixel(4*u.deg, 5*u.deg) == [0.0, 0.0]
    assert sub_ltc[1:, 1:].wcs.world_to_pixel(5*u.deg, 6*u.deg) == [0.0, 0.0]


@pytest.mark.xfail(reason=">1D Tables not supported")
def test_2d_skycoord_no_mesh_slice(lut_2d_skycoord_no_mesh):
    sub_ltc = lut_2d_skycoord_no_mesh[1:3, 1:2]
    assert sub_ltc.table.shape == (2, 1)


def test_1d_time_slice(lut_1d_time):
    sub_ltc = lut_1d_time[1:3]
    assert sub_ltc.table.shape == (2,)


def test_join_slice(lut_1d_time, lut_1d_wave):
    ltc = lut_1d_time & lut_1d_wave

    sub_ltc = ltc[2:8, 2:8]
    assert len(sub_ltc._table_coords) == 2
    assert (sub_ltc._table_coords[0].table == lut_1d_time.table[2:8]).all()
    assert u.allclose(sub_ltc._table_coords[1].table[0], lut_1d_wave.table[0][2:8])


def test_slicing_errors(lut_1d_time, lut_1d_wave, lut_1d_distance, lut_2d_skycoord_mesh):
    with pytest.raises(ValueError) as ei:
        lut_1d_time[1, 2]
    assert "slice with incorrect length" in str(ei)

    with pytest.raises(ValueError) as ei:
        lut_1d_wave[1, 2]
    assert "slice with incorrect length" in str(ei)

    with pytest.raises(ValueError) as ei:
        lut_1d_distance[1, 2]
    assert "slice with incorrect length" in str(ei)

    with pytest.raises(ValueError) as ei:
        lut_2d_skycoord_mesh[1, 2, 3]
    assert "slice with incorrect length" in str(ei)

    join = lut_1d_time & lut_1d_distance

    with pytest.raises(ValueError) as ei:
        join[1]
    assert "length of the slice" in str(ei)


def test_mtc_dropped_table(lut_1d_time):
    mtc = MultipleTableCoordinate(lut_1d_time)
    sub = mtc[0]

    assert len(sub._table_coords) == 0
    assert len(sub._dropped_coords) == 1

    dwd = sub.dropped_world_dimensions
    assert isinstance(dwd, dict)
    wao_classes = dwd.pop("world_axis_object_classes")
    assert all(isinstance(value, list) for value in dwd.values())
    assert all(len(value) == 1 for value in dwd.values())

    assert dwd["world_axis_names"] == ["time"]
    assert dwd["world_axis_units"] == ["s"]
    assert dwd["world_axis_physical_types"] == ["time"]
    assert dwd["world_axis_object_components"][0][0:2] == ("temporal", 0)
    assert wao_classes["temporal"][0] is Time
    assert dwd["value"] == [0*u.s]


def test_mtc_dropped_table_join(lut_1d_time, lut_2d_skycoord_mesh):
    mtc = MultipleTableCoordinate(lut_1d_time, lut_2d_skycoord_mesh)
    sub = mtc[0, :, :]

    assert len(sub._table_coords) == 1
    assert len(sub._dropped_coords) == 1

    dwd = sub.dropped_world_dimensions
    assert isinstance(dwd, dict)
    wao_classes = dwd.pop("world_axis_object_classes")
    assert all(isinstance(value, list) for value in dwd.values())
    assert all(len(value) == 1 for value in dwd.values())

    assert dwd["world_axis_names"] == ["time"]
    assert all(isinstance(u, str) for u in dwd["world_axis_units"])
    assert dwd["world_axis_units"] == ["s"]
    assert dwd["world_axis_physical_types"] == ["time"]
    assert dwd["world_axis_object_components"][0][0:2] == ("temporal", 0)
    assert wao_classes["temporal"][0] is Time
    assert dwd["value"] == [0*u.s]


def test_mtc_dropped_table_skycoord_join(lut_1d_time, lut_2d_skycoord_mesh):
    mtc = MultipleTableCoordinate(lut_1d_time, lut_2d_skycoord_mesh)
    sub = mtc[:, 0, 0]

    assert len(sub._table_coords) == 1
    assert len(sub._dropped_coords) == 1

    dwd = sub.dropped_world_dimensions
    assert isinstance(dwd, dict)
    wao_classes = dwd.pop("world_axis_object_classes")
    assert all(isinstance(value, list) for value in dwd.values())
    assert all(len(value) == 2 for value in dwd.values())

    assert dwd["world_axis_names"] == ["lon", "lat"]
    assert all(isinstance(u, str) for u in dwd["world_axis_units"])
    assert dwd["world_axis_units"] == ["deg", "deg"]
    assert dwd["world_axis_physical_types"] == ["pos.eq.ra", "pos.eq.dec"]
    assert dwd["world_axis_object_components"] == [("celestial", 0, "spherical.lon"), ("celestial", 1, "spherical.lat")]
    assert wao_classes["celestial"][0] is SkyCoord
    assert dwd["value"] == [0*u.deg, 0*u.deg]


@pytest.mark.xfail(reason=">1D Tables not supported")
def test_mtc_dropped_quantity_table(lut_1d_time, lut_2d_distance_no_mesh):
    mtc = MultipleTableCoordinate(lut_1d_time, lut_2d_distance_no_mesh)
    sub = mtc[:, 0, 0]

    assert len(sub._table_coords) == 1
    assert len(sub._dropped_coords) == 1

    pytest.importorskip("gwcs", minversion="0.16.2a1.dev17")

    dwd = sub.dropped_world_dimensions
    assert isinstance(dwd, dict)
    wao_classes = dwd.pop("world_axis_object_classes")
    assert all(isinstance(value, list) for value in dwd.values())
    assert dwd
    assert all(len(value) == 2 for value in dwd.values())

    assert dwd["world_axis_names"] == ["", ""]
    assert all(isinstance(u, str) for u in dwd["world_axis_units"])
    assert dwd["world_axis_units"] == ["km", "km"]
    assert dwd["world_axis_physical_types"] == ["custom:SPATIAL", "custom:SPATIAL"]
    assert dwd["world_axis_object_components"] == [("SPATIAL", 0, "value"), ("SPATIAL1", 0, "value")]
    assert wao_classes["SPATIAL"][0] is u.Quantity
    assert wao_classes["SPATIAL1"][0] is u.Quantity
    assert dwd["value"] == [0*u.km, 9*u.km]


def test_mtc_dropped_quantity_inside_table(lut_3d_distance_mesh):
    sub = lut_3d_distance_mesh[:, 0, :]

    assert len(sub.table) == 2

    pytest.importorskip("gwcs", minversion="0.16.2a1.dev17")

    dwd = sub.dropped_world_dimensions
    assert isinstance(dwd, dict)
    dwd.pop("world_axis_object_classes")
    assert all(isinstance(value, list) for value in dwd.values())
    assert dwd
    assert all(len(value) == 1 for value in dwd.values())

    sub = lut_3d_distance_mesh[:, 0, 0]

    assert len(sub.table) == 1

    dwd = sub.dropped_world_dimensions
    assert isinstance(dwd, dict)
    dwd.pop("world_axis_object_classes")
    assert all(isinstance(value, list) for value in dwd.values())
    assert dwd
    assert all(len(value) == 2 for value in dwd.values())


@pytest.mark.xfail(reason=">1D Tables not supported")
def test_mtc_dropped_quantity_inside_table_no_mesh(lut_2d_distance_no_mesh):
    """
    When not meshing, we don't drop a coord, as the coordinate for the sliced
    out axis can still vary along the remaining coordinate.
    """
    sub = lut_2d_distance_no_mesh[:, 0]

    assert len(sub.table) == 2

    pytest.importorskip("gwcs", minversion="0.16.2a1.dev17")

    dwd = sub.dropped_world_dimensions
    assert isinstance(dwd, dict)
    dwd.pop("world_axis_object_classes")
    assert not dwd


def test_mtc_dropped_quantity_join_drop_table(lut_1d_time, lut_3d_distance_mesh):
    mtc = MultipleTableCoordinate(lut_1d_time, lut_3d_distance_mesh)
    sub = mtc[:, 0, :, :]

    assert len(sub._table_coords) == 2
    assert len(sub._dropped_coords) == 0

    pytest.importorskip("gwcs", minversion="0.16.2a1.dev17")

    dwd = sub.dropped_world_dimensions
    assert isinstance(dwd, dict)
    dwd.pop("world_axis_object_classes")
    assert all(isinstance(value, list) for value in dwd.values())
    assert all(len(value) == 1 for value in dwd.values())

    sub = mtc[0, 0, :, :]

    assert len(sub._table_coords) == 1
    assert len(sub._dropped_coords) == 1

    pytest.importorskip("gwcs", minversion="0.16.2a1.dev17")

    dwd = sub.dropped_world_dimensions
    assert isinstance(dwd, dict)
    dwd.pop("world_axis_object_classes")
    assert all(isinstance(value, list) for value in dwd.values())
    assert all(len(value) == 2 for value in dwd.values())


################################################################################
# Tests of & operator
################################################################################


def test_and_base_table_coordinate():
    data = Time(["2011-01-01T00:00:00",
                 "2011-01-01T00:00:10",
                 "2011-01-01T00:00:20",
                 "2011-01-01T00:00:30"], format="isot")

    ttc = TimeTableCoordinate(data)

    qtc = QuantityTableCoordinate(range(10)*u.m, mesh=False)

    join = ttc & ttc
    assert isinstance(join, MultipleTableCoordinate)

    join2 = join & qtc

    assert isinstance(join2, MultipleTableCoordinate)
    assert len(join2._table_coords) == 3
    assert join2._table_coords[2] is qtc

    join3 = qtc & join

    assert isinstance(join3, MultipleTableCoordinate)
    assert len(join3._table_coords) == 3
    assert join3._table_coords[0] is qtc

    join4 = ttc & qtc
    assert isinstance(join4, MultipleTableCoordinate)
    assert len(join4._table_coords) == 2
    assert join4._table_coords[0] is ttc
    assert join4._table_coords[1] is qtc

    join5 = join & join
    assert isinstance(join5, MultipleTableCoordinate)
    assert len(join5._table_coords) == 4


def test_and_errors():
    data = Time(["2011-01-01T00:00:00",
                 "2011-01-01T00:00:10",
                 "2011-01-01T00:00:20",
                 "2011-01-01T00:00:30"], format="isot")

    ttc = TimeTableCoordinate(data)

    qtc = QuantityTableCoordinate(range(10)*u.m, mesh=False)

    with pytest.raises(TypeError) as ei:
        ttc & 5
    assert "unsupported operand type(s) for &: 'TimeTableCoordinate' and 'int'" in str(ei)

    join = ttc & qtc

    with pytest.raises(TypeError) as ei:
        join & 5
    assert "unsupported operand type(s) for &: 'MultipleTableCoordinate' and 'int'" in str(ei)

    with pytest.raises(TypeError) as ei:
        5 & join
    assert "unsupported operand type(s) for &: 'int' and 'MultipleTableCoordinate'" in str(ei)
