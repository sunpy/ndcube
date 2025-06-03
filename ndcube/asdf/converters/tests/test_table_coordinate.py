import pytest

import asdf
import astropy.units as u
from astropy.coordinates import SkyCoord

from ndcube.extra_coords import MultipleTableCoordinate


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
