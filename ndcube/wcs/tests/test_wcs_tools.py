import astropy.units as u
import numpy as np
from astropy.wcs import WCS
from astropy.wcs.wcsapi import SlicedLowLevelWCS
from numpy.testing import assert_array_almost_equal, assert_array_equal

from ndcube.wcs.wcs_tools import unwrap_wcs_to_fitswcs
from ndcube.wcs.wrappers import ResampledLowLevelWCS


def test_unwrap_wcs_to_fitswcs():
    # Build FITS-WCS and wrap it in different operations.
    input_wcs = WCS(naxis=3)
    input_wcs.wcs.ctype = 'WAVE', 'HPLT-TAN', 'HPLN-TAN'
    input_wcs.wcs.cunit = 'Angstrom', 'deg', 'deg'
    input_wcs.wcs.cdelt = 0.2, 0.5, 0.4
    input_wcs.wcs.crpix = 0, 2, 2
    input_wcs.wcs.crval = 10, 0.5, 1
    input_wcs.wcs.cname = 'wavelength', 'HPC lat', 'HPC lon'
    input_wcs._naxis = [5, 4, 4]
    # Slice WCS
    input_wcs = SlicedLowLevelWCS(input_wcs, (slice(None), slice(1, 3), 0))
    # Resample WCS
    input_wcs = ResampledLowLevelWCS(input_wcs, [2, 2], offset=[0.5, 0.5])
    # Slice WCS again
    input_wcs = SlicedLowLevelWCS(input_wcs, (slice(0, 1), 0))
    # Reconstruct fitswcs
    output_wcs, dropped_data_dimensions = unwrap_wcs_to_fitswcs(input_wcs)
    # Assert output_wcs is correct
    assert_array_equal(dropped_data_dimensions, np.array([False, True, True]))
    assert isinstance(output_wcs, WCS)
    assert output_wcs._naxis == [1, 1, 1]
    assert list(output_wcs.wcs.ctype) == ['WAVE', 'HPLT-TAN', 'HPLN-TAN']
    world_values = output_wcs.array_index_to_world([0], [0], [0])
    assert_array_almost_equal(world_values[0].to_value(u.m), np.array([1.02e-09]))
    assert_array_almost_equal(world_values[1].Ty.to_value(u.deg), np.array([0.5]), decimal=5)
    assert_array_almost_equal(world_values[1].Tx.to_value(u.deg), np.array([0.6]), decimal=5)
