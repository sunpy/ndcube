import astropy.units as u
import numpy as np
import pytest
from astropy.wcs.wcsapi.utils import wcs_info_str
from astropy.coordinates import SkyCoord
from astropy.time import Time

from ndcube.extra_coords import LookupTableCoord


def test_1d_distance():
    lookup_table = u.Quantity(np.arange(10) * u.km)

    ltc = LookupTableCoord(lookup_table)
    assert ltc.model.n_inputs == 1
    assert ltc.model.n_outputs == 1
    assert ltc.model.lookup_table is lookup_table
    assert u.allclose(u.Quantity(range(10), u.pix), ltc.model.points)

    assert u.allclose(ltc.wcs.pixel_to_world(0), 0*u.km)
    assert u.allclose(ltc.wcs.pixel_to_world(9), 9*u.km)


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

def test_2d_nout_1_no_mesh():
    lookup_table = np.arange(9).reshape(3,3) * u.km, np.arange(9, 18).reshape(3,3) * u.km

    ltc = LookupTableCoord(*lookup_table, mesh=False)
    assert ltc.wcs.world_n_dim == 2
    assert ltc.wcs.pixel_n_dim == 2

    assert ltc.model.n_inputs == 2
    assert ltc.model.n_outputs == 2

    assert u.allclose(ltc.wcs.pixel_to_world(0*u.pix, 0*u.pix),
                      (0, 9)*u.km)


def test_2d_skycoord_mesh():
    sc = SkyCoord(range(10), range(10), unit=u.deg)
    ltc = LookupTableCoord(sc, mesh=True)
    assert ltc.model.n_inputs == 2
    assert ltc.model.n_outputs == 2


@pytest.mark.xfail
def test_3d_skycoord_mesh():
    """Known failure due to bug in gwcs."""
    sc = SkyCoord(range(10), range(10), range(10), unit=(u.deg, u.deg, u.AU))
    ltc = LookupTableCoord(sc, mesh=True)
    assert ltc.model.n_inputs == 3
    assert ltc.model.n_outputs == 3


def test_2d_skycoord_no_mesh():
    data = np.arange(9).reshape(3,3), np.arange(9, 18).reshape(3,3)
    sc = SkyCoord(*data, unit=u.deg)
    ltc = LookupTableCoord(sc, mesh=False)
    assert ltc.model.n_inputs == 2
    assert ltc.model.n_outputs == 2


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
