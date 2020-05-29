import astropy.units as u
import numpy as np
import pytest
from astropy.wcs.wcsapi.utils import wcs_info_str

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
