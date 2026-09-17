import numpy as np
import pytest

import asdf
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from ndcube.extra_coords import MultipleTableCoordinate, QuantityTableCoordinate, TimeTableCoordinate


@pytest.fixture
def lut(request):
    return request.getfixturevalue(request.param)


def assert_table_coord_equal(test_table, expected_table):
    test_table = test_table.table
    expected_table = expected_table.table
    if not isinstance(expected_table, tuple):
        test_table = (test_table,)
        expected_table = (expected_table,)
    for test_tab, ex_tab in zip(test_table, expected_table):
        if ex_tab.isscalar:
            assert test_tab == ex_tab
        elif isinstance(ex_tab, SkyCoord):
            assert u.allclose(ex_tab.spherical.lat, test_tab.spherical.lat)
            assert u.allclose(ex_tab.spherical.lon, test_tab.spherical.lon)
        else:
            assert all(test_tab == ex_tab)


@pytest.mark.parametrize("lut",
                         [
                             "lut_1d_distance",
                             "lut_3d_distance_mesh",
                             "lut_1d_skycoord_no_mesh",
                             "lut_2d_skycoord_no_mesh",
                             "lut_2d_skycoord_mesh",
                             "lut_3d_skycoord_mesh",
                             "lut_1d_time",
                             "lut_1d_wave",
                         ], indirect=True)
def test_serialize(lut, tmp_path):
    file_path = tmp_path / "test.asdf"
    with asdf.AsdfFile() as af:
        af["lut"] = lut
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        assert_table_coord_equal(af["lut"], lut)


def assert_mtc_equal(test_mtc, expected_mtc):
    assert len(test_mtc._table_coords) == len(expected_mtc._table_coords)
    assert len(test_mtc._dropped_coords) == len(expected_mtc._dropped_coords)

    for (test_tc, expected_tc) in zip(test_mtc._table_coords, expected_mtc._table_coords):
        assert_table_coord_equal(test_tc, expected_tc)

    for (test_tc, expected_tc) in zip(test_mtc._dropped_coords, expected_mtc._dropped_coords):
        assert_table_coord_equal(test_tc, expected_tc)


def test_serialize_multiple_coord(lut_1d_distance, lut_1d_time, tmp_path):
    mtc = MultipleTableCoordinate(lut_1d_distance, lut_1d_time)
    file_path = tmp_path / "test.asdf"
    with asdf.AsdfFile() as af:
        af["lut"] = mtc
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        new_mtc = af["lut"]
        assert_mtc_equal(new_mtc, mtc)


def test_serialize_sliced_multiple_coord(lut_1d_distance, lut_1d_time, tmp_path):
    mtc = MultipleTableCoordinate(lut_1d_distance, lut_1d_time)[0, :]
    file_path = tmp_path / "test.asdf"
    with asdf.AsdfFile() as af:
        af["lut"] = mtc
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        new_mtc = af["lut"]
        assert_mtc_equal(new_mtc, mtc)


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
@pytest.mark.parametrize("kind", ["quantity", "time"])
def test_serialize_nd_table(shape, kind, tmp_path):
    values = np.arange(np.prod(shape)).reshape(shape)
    if kind == "quantity":
        coord = QuantityTableCoordinate(values * u.m, names="distance", physical_types="pos.distance")
    else:
        origin = Time("2020-01-01", scale="tai")
        coord = TimeTableCoordinate(origin + values * u.s, names="time", physical_types="time",
                                    reference_time=origin - 1 * u.day)
    path = tmp_path / "nd-table.asdf"
    with asdf.AsdfFile({"coord": coord}) as af:
        af.write_to(path)
    with asdf.open(path) as af:
        restored = af["coord"]
        assert restored.n_inputs == len(shape)
        assert restored.names == coord.names
        assert restored.physical_types == coord.physical_types
        if kind == "time":
            assert restored.reference_time == coord.reference_time
            assert restored.table.scale == coord.table.scale
        pixels = np.indices(shape)[::-1]
        assert (restored.wcs.pixel_to_world(*pixels) == coord.wcs.pixel_to_world(*pixels)).all()
        sliced = restored[(0,) + (slice(None),) * (len(shape) - 1)]
        assert sliced.n_inputs == len(shape) - 1
