import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from astropy.time import Time
from astropy.wcs import WCS
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
    assert_array_almost_equal(world_values[0][0], np.array([2700]))
    assert_array_almost_equal(world_values[1], np.array([1.04e-09, 1.10e-09]))
    assert_array_almost_equal(world_values[2][0], np.array([1.26915033e-05]))
    assert_array_almost_equal(world_values[3][0], np.array([0.60002173]))
