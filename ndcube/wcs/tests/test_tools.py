import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal

import astropy
from astropy.time import Time
from astropy.wcs import WCS, Sip
from astropy.wcs.wcsapi import SlicedLowLevelWCS

from ndcube.wcs.tools import unwrap_wcs_to_fitswcs
from ndcube.wcs.wrappers import ResampledLowLevelWCS


def test_unwrap_wcs_to_fitswcs():
    # Build FITS-WCS and wrap it in different operations.
    time_ref = Time("2000-01-01T00:00:00", scale="utc", format="fits")
    header = {
        "CTYPE1": "TIME", "CTYPE2": "WAVE", "CTYPE3": "HPLT-TAN", "CTYPE4": "HPLN-TAN",
        "CUNIT1": "s", "CUNIT2": "Angstrom", "CUNIT3": "deg", "CUNIT4": "deg",
        "CDELT1": 600, "CDELT2": 0.2, "CDELT3": 0.5, "CDELT4": 0.4,
        "CRPIX1": 0, "CRPIX2": 0, "CRPIX3": 2, "CRPIX4": 2,
        "CRVAL1": 0, "CRVAL2": 10, "CRVAL3": 0.5, "CRVAL4": 1,
        "CNAME1": "time", "CNAME2": "wavelength", "CNAME3": "HPC lat", "CNAME4": "HPC lon",
        "NAXIS1": 5, "NAXIS2": 9, "NAXIS3": 4, "NAXIS4": 4,
        "DATEREF": time_ref.fits}
    orig_wcs = WCS(header)
    # Slice WCS
    wcs1 = SlicedLowLevelWCS(orig_wcs, (0, 0, slice(None), slice(1, None)))  # numpy order
    # Resample WCS
    wcs2 = ResampledLowLevelWCS(wcs1, [2, 3], offset=[0.5, 1])  # WCS order
    # Slice WCS again
    wcs3 = SlicedLowLevelWCS(wcs2, (slice(0, 2), slice(1, 2)))  # numpy order
    # Reconstruct fitswcs
    output_wcs, dropped_data_dimensions = unwrap_wcs_to_fitswcs(wcs3)
    # Assert output_wcs is correct
    assert_array_equal(dropped_data_dimensions, np.array([True, True, False, False]))
    assert isinstance(output_wcs, WCS)
    assert output_wcs._naxis == [1, 2, 1, 1]
    assert list(output_wcs.wcs.ctype) == ['TIME', 'WAVE', 'HPLT-TAN', 'HPLN-TAN']
    world_values = output_wcs.array_index_to_world_values([0], [0], [0, 1], [0])
    expected_time, expected_wave = wcs3.array_index_to_world_values([0, 1], [0])
    assert_array_almost_equal(world_values[0], expected_time)
    assert_array_almost_equal(world_values[1], expected_wave)
    # Lat/lon were sliced away in wcs1 so compare to its dropped world values
    expected_lat, expected_lon = wcs1.dropped_world_dimensions["value"]
    assert_array_almost_equal(world_values[2], expected_lat)
    assert_array_almost_equal(world_values[3], expected_lon)


def test_unwrap_wcs_to_fitswcs_does_not_modify_input():
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cdelt = [1.0, 1.0]
    wcs.wcs.crpix = [1.0, 1.0]
    wcs._naxis = [4, 4]

    resampled_wcs = ResampledLowLevelWCS(wcs, [2, 2])
    output_wcs, _ = unwrap_wcs_to_fitswcs(resampled_wcs)

    # The resample was applied to the returned WCS
    assert output_wcs is not wcs
    assert_array_equal(output_wcs.wcs.cdelt, [2.0, 2.0])
    # assert_array_equal(output_wcs.wcs.crpix, [0.75, 0.75])
    pixels = [-0.5, 0, 0.5, 1, 1.5]
    assert_allclose(output_wcs.pixel_to_world_values(pixels, pixels),
                    resampled_wcs.pixel_to_world_values(pixels, pixels))

    assert list(output_wcs._naxis) == [2, 2]
    # and not to the WCS that was wrapped
    assert_array_equal(wcs.wcs.cdelt, [1.0, 1.0])
    assert_array_equal(wcs.wcs.crpix, [1.0, 1.0])
    assert list(wcs._naxis) == [4, 4]

@pytest.mark.skipif(astropy.__version__ < "7.2.0", reason="preserve_units was added in astropy 7.2")
def test_unwrap_wcs_to_fitswcs_preserve_units():
    # With preserve_units=True and non-degree celestial units, astropy returns
    # cdelt as a read-only copy, so the resample must not modify it in place.
    header = {"CTYPE1": "HPLN-TAN", "CTYPE2": "HPLT-TAN",
              "CUNIT1": "arcsec", "CUNIT2": "arcsec",
              "CDELT1": 1.0, "CDELT2": 1.0, "CRPIX1": 1.0, "CRPIX2": 1.0,
              "NAXIS1": 4, "NAXIS2": 4}
    wcs = WCS(header, preserve_units=True)

    resampled = ResampledLowLevelWCS(wcs, [2, 2])
    output_wcs, _ = unwrap_wcs_to_fitswcs(resampled)

    assert list(output_wcs.wcs.cunit) == ["arcsec", "arcsec"]
    assert_array_equal(output_wcs.wcs.cdelt, [2.0, 2.0])
    assert_array_equal(output_wcs.wcs.crpix, [0.75, 0.75])
    assert list(output_wcs._naxis) == [2, 2]
    assert_array_equal(wcs.wcs.cdelt, [1.0, 1.0])
    pixels = ([-0.25, -0.25], [1.75, 1.75])
    assert_allclose(resampled.pixel_to_world_values(*pixels),
                    output_wcs.pixel_to_world_values(*pixels))

@pytest.mark.parametrize("cdelt", [[1, 1], [2, 1]])
def test_unwrap_wcs_to_fitswcs_resampled_pc(cdelt):
    wcs = WCS(naxis=2)
    wcs.wcs.pc = [[0, -1], [1, 0]]  # 90-degree rotation
    wcs.wcs.crpix = [1, 1]
    wcs.wcs.cdelt = cdelt
    wcs.pixel_shape = (4, 4)

    resampled = ResampledLowLevelWCS(wcs, [2, 1])
    unwrapped, _ = unwrap_wcs_to_fitswcs(resampled)

    pixels = ([0, 1, 0], [0, 0, 1])
    # in diff between two world values crpix and crval cancel so isolates PC
    expected = np.diff(resampled.pixel_to_world_values(*pixels), axis=1)
    actual = np.diff(unwrapped.pixel_to_world_values(*pixels), axis=1)
    assert_allclose(actual, expected)

@pytest.mark.parametrize("factor", [[2, 2], [2, 1]])
def test_unwrap_wcs_to_fitswcs_resampled_cd(factor):
    # Same as PC above but using combined CD
    wcs = WCS(naxis=2)
    wcs.wcs.cd = [[0, -1], [1, 0]]  # 90 degree rotation
    wcs.wcs.crpix = [1, 1]
    wcs.pixel_shape = (4, 4)

    resampled = ResampledLowLevelWCS(wcs, factor)
    unwrapped, _ = unwrap_wcs_to_fitswcs(resampled)

    pixels = ([0, 1, 0], [0, 0, 1])
    assert_allclose(unwrapped.pixel_to_world_values(*pixels),
                    resampled.pixel_to_world_values(*pixels))

def test_unwrap_wcs_to_fitswcs_resampled_sip():
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    wcs.wcs.crval = [10, 20]
    wcs.wcs.cdelt = [0.001, 0.001]
    wcs.wcs.crpix = [6, 7]
    wcs.pixel_shape = (12, 12)
    a = np.zeros((3, 3))
    a[2, 0] = 2e-3
    a[1, 1] = -1e-3
    b = np.zeros((3, 3))
    b[0, 2] = 1e-3
    b[1, 1] = 2e-3
    wcs.sip = Sip(a, b, -a, -b, wcs.wcs.crpix)

    resampled = ResampledLowLevelWCS(wcs, [2, 3], offset=[1, 2])
    unwrapped, _ = unwrap_wcs_to_fitswcs(resampled)

    pixels = ([0, 1, 2.5, -0.5], [0, 3, 1.5, -0.5])
    world = resampled.pixel_to_world_values(*pixels)
    assert_allclose(unwrapped.pixel_to_world_values(*pixels), world)
    assert_allclose(unwrapped.world_to_pixel_values(*world),
                    resampled.world_to_pixel_values(*world))
