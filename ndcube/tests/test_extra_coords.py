import astropy.units as u
import numpy as np
import pytest
from astropy.wcs.wcsapi.utils import wcs_info_str

from ndcube.extra_coords import LookupTableCoord


def test_1d_distance():
    lookup_table = u.Quantity(np.arange(10) * u.km)

    ltc = LookupTableCoord(lookup_table)
    print(ltc.model)
    print(ltc.frame)
    print(wcs_info_str(ltc.wcs))


def test_1d_spectral():
    lookup_table = u.Quantity(np.arange(10) * u.nm)

    ltc = LookupTableCoord(lookup_table, frame_type="spectral")
    print(ltc.model)
    print(ltc.frame)
    print(ltc.wcs.pixel_to_world(0*u.pix))
    print(wcs_info_str(ltc.wcs))


def test_3d_distance():
    lookup_table = u.Quantity(np.arange(10) * u.km), u.Quantity(np.arange(10, 20) * u.km), u.Quantity(np.arange(20, 30) * u.km)

    ltc = LookupTableCoord(lookup_table, mesh=True)
    print(ltc.model)
    print(ltc.frame)
    print(ltc.wcs.pixel_to_world(0*u.pix, 0*u.pix, 0*u.pix))
    print(wcs_info_str(ltc.wcs))


def test_3d_nout_1_no_mesh():
    lookup_table = np.arange(9).reshape(3,3) * u.km

    ltc = LookupTableCoord(lookup_table, mesh=False)
    print(wcs_info_str(ltc.wcs))


def test_3d_nout_1_no_mesh():
    lookup_table = np.arange(9).reshape(3,3) * u.km, np.arange(9, 18).reshape(3,3) * u.km

    ltc = LookupTableCoord(lookup_table)
    print(wcs_info_str(ltc.wcs))
    breakpoint()
