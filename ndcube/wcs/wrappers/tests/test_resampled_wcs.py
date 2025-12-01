import numbers

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.tests.helper import assert_quantity_allclose
from astropy.wcs import WCS
from astropy.wcs.wcsapi import HighLevelWCSWrapper

from ndcube.wcs.wrappers import ResampledLowLevelWCS


@pytest.fixture
def celestial_wcs(request):
    return request.getfixturevalue(request.param)


EXPECTED_2D_REPR_NUMPY2 = """
ResampledLowLevelWCS Transformation

This transformation has 2 pixel and 2 world dimensions

Array shape (Numpy order): (2, 15)

Pixel Dim  Axis Name  Data size  Bounds
        0  None              15  (-1.75, 13.25)
        1  None         2.33333  (0.0, 2.0)

World Dim  Axis Name        Physical Type  Units
        0  Right Ascension  pos.eq.ra      deg
        1  Declination      pos.eq.dec     deg

Correlation between pixel and world axes:

           Pixel Dim
World Dim    0    1
        0  yes  yes
        1  yes  yes
""".strip()
EXPECTED_2D_REPR_NUMPY1 = """
ResampledLowLevelWCS Transformation

This transformation has 2 pixel and 2 world dimensions

Array shape (Numpy order): (2.3333333333333335, 15.0)

Pixel Dim  Axis Name  Data size  Bounds
        0  None              15  (-1.75, 13.25)
        1  None         2.33333  (0.0, 2.0)

World Dim  Axis Name        Physical Type  Units
        0  Right Ascension  pos.eq.ra      deg
        1  Declination      pos.eq.dec     deg

Correlation between pixel and world axes:

           Pixel Dim
World Dim    0    1
        0  yes  yes
        1  yes  yes
""".strip()

@pytest.mark.parametrize('celestial_wcs',
                         ['celestial_2d_ape14_wcs', 'celestial_2d_fitswcs'],
                         indirect=True)
def test_2d(celestial_wcs):

    # Upsample along the first pixel dimension and downsample along the second
    # pixel dimension.
    factors = [0.4, 3]
    wcs = ResampledLowLevelWCS(celestial_wcs, factors)

    # The following shouldn't change compared to the original WCS
    assert wcs.pixel_n_dim == 2
    assert wcs.world_n_dim == 2
    assert tuple(wcs.world_axis_physical_types) == ('pos.eq.ra', 'pos.eq.dec')
    assert tuple(wcs.world_axis_units) == ('deg', 'deg')
    assert tuple(wcs.pixel_axis_names) == ('', '')
    assert tuple(wcs.world_axis_names) == ('Right Ascension', 'Declination')
    assert_equal(wcs.axis_correlation_matrix, np.ones((2, 2)))

    # Shapes and bounds should be floating-point if needed
    assert_allclose(wcs.pixel_shape, (15, 7/3))
    assert_allclose(wcs.array_shape, (7/3, 15))
    assert_allclose(wcs.pixel_bounds, ((-1.75, 13.25), (0, 2)))

    under_pixel, over_pixel = (2.3, 1.2), (5.75+0.6*1.25, 0.2/1.5*0.5)
    world = celestial_wcs.pixel_to_world_values(*under_pixel)

    assert_allclose(wcs.pixel_to_world_values(*over_pixel), world)
    assert_allclose(wcs.array_index_to_world_values(*over_pixel[::-1]), world)
    assert_allclose(wcs.world_to_pixel_values(*world), over_pixel)
    assert_allclose(wcs.world_to_array_index_values(*world),
                    np.around(over_pixel[::-1]).astype(int))

    EXPECTED_2D_REPR = EXPECTED_2D_REPR_NUMPY2 if np.__version__ >= '2.0.0' else EXPECTED_2D_REPR_NUMPY1
    assert str(wcs) == EXPECTED_2D_REPR
    assert EXPECTED_2D_REPR in repr(wcs)

    celestial_wcs.pixel_bounds = None
    under_pixel_array = (np.array([2.3, 2.4]),
                         np.array([4.3, 4.4]))
    over_pixel_array = (np.array([over_pixel[0], 5.75+0.8*1.25]),
                        1 + np.array([0.3, 0.4]) / 1.5 * 0.5)
    world_array = celestial_wcs.pixel_to_world_values(*under_pixel_array)

    assert_allclose(wcs.pixel_to_world_values(*over_pixel_array), world_array)
    assert_allclose(wcs.array_index_to_world_values(*over_pixel_array[::-1]), world_array)
    assert_allclose(wcs.world_to_pixel_values(*world_array), over_pixel_array)
    assert_allclose(wcs.world_to_array_index_values(*world_array),
                    np.around(np.asarray(over_pixel_array)[::-1]).astype(int))

    wcs_hl = HighLevelWCSWrapper(wcs)

    celestial = wcs_hl.pixel_to_world(*over_pixel)
    assert isinstance(celestial, SkyCoord)
    assert_quantity_allclose(celestial.ra, world[0] * u.deg)
    assert_quantity_allclose(celestial.dec, world[1] * u.deg)

    celestial = wcs_hl.pixel_to_world(*over_pixel_array)
    assert isinstance(celestial, SkyCoord)
    assert_quantity_allclose(celestial.ra, world_array[0] * u.deg)
    assert_quantity_allclose(celestial.dec, world_array[1] * u.deg)


@pytest.mark.parametrize('celestial_wcs',
                         ['celestial_2d_ape14_wcs',
                          'celestial_2d_fitswcs'],
                         indirect=True)
@pytest.mark.parametrize(('factor', 'offset', 'over_pixel'),
                         [(2, 0, (0.9, 1.9)),
                          (2, 1, (0.4, 1.4)),
                         ])
def test_scalar_factor_and_offset(celestial_wcs, factor, offset, over_pixel):
    celestial_wcs.pixel_bounds = None
    wcs = ResampledLowLevelWCS(celestial_wcs, factor, offset=offset)
    # Define the pixel coord pre-resampled pixel grid corresponding to
    # the same location in the resampled grid, as defined in the parameterization.
    under_pixel = (2.3, 4.3)
    # Get the corresponding world location using original WCS.
    world = celestial_wcs.pixel_to_world_values(*under_pixel)
    # Confirm resampled WCS maps to and from the same world coordinate to the
    # pixel location in the resample pixel coords.
    assert_allclose(wcs.pixel_to_world_values(*over_pixel), world)
    assert_allclose(wcs.array_index_to_world_values(*over_pixel[::-1]), world)
    assert_allclose(wcs.world_to_pixel_values(*world), over_pixel)
    assert_allclose(wcs.world_to_array_index_values(*world),
                    np.around(over_pixel[::-1]).astype(int))


@pytest.mark.parametrize('celestial_wcs',
                         ['celestial_2d_ape14_wcs'],
                         indirect=True)
def test_factor_wrong_length_error(celestial_wcs):
    with pytest.raises(ValueError):
        ResampledLowLevelWCS(celestial_wcs, [2] * 3)


@pytest.mark.parametrize('celestial_wcs',
                         ['celestial_2d_ape14_wcs'],
                         indirect=True)
def test_scalar_wrong_length_error(celestial_wcs):
    with pytest.raises(ValueError):
        ResampledLowLevelWCS(celestial_wcs, 2, offset=[1] * 3)


@pytest.mark.parametrize('celestial_wcs',
                         ['celestial_2d_ape14_wcs', 'celestial_2d_fitswcs'],
                         indirect=True)
def test_int_fraction_pixel_shape(celestial_wcs):
    # Some fractional factors are not representable by exact floats, e.g. 1/3.
    # However, it is still desirable for the pixel shape to return ints in these cases.
    # This test checks that this is the case.
    wcs = ResampledLowLevelWCS(celestial_wcs, 1/3)
    assert wcs.pixel_shape == (18, 21)
    for dim in wcs.pixel_shape:
        assert isinstance(dim, numbers.Integral)

@pytest.fixture
def four_five_wcs():
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cdelt = [2, 2]
    wcs.wcs.crval = [0, 0]
    wcs.wcs.crpix = [1, 1]

    wcs.wcs.set()
    wcs.pixel_shape = [4, 5]
    return wcs

@pytest.mark.parametrize(('factor', 'offset', 'expected_over_pixels'),
                         [([2, 3], [0, 0], np.meshgrid(np.linspace(-0.5, 1.5, 4*2+1), np.linspace(-0.5, 1+1/6, 5*2+1))),
                          ([2, 3], [1, 2], np.meshgrid(np.linspace(-1, 1, 4*2+1), np.linspace(-1-1/6, 0.5, 5*2+1))),
                         ])
def test_resampled_pixel_to_world_values(four_five_wcs, factor, offset, expected_over_pixels):
    """
    Notes
    -----
    Below shows schematics of the two test cases tested in this test of how ResampledLowLevelWCS.
    The asterisks show the corners of pixels in a grid before resampling, while the dashes and
    pipes show the edges the resampled pixels. In the first case, the resampled pixels are
    obtained by applied resampling factors of 2 along the x-axis and 3 along the y-axis. No
    offset is applied to either axis. In the second case, the same resampling factors have
    been applied, but an offsets of 1 and 2 pixels have been applied to the x- and y-axes,
    respectively. The right column/upper row of numbers along the side of/below the grids denote
    the edges and centres of the original pixel grid in the original pixel coordinates.
    The left column and lower row gives the same locations in the pixel coordinates of the
    resampled grid.

    Test Case 1

    ::

        resampled  original
        factor=3
        offset=0

        1+1/6       4.5*         *         *         *         *
                       |                   |                   |
          1          4 |                   |                   |
                       |                   |                   |
         5/6        3.5*         *         *         *         *
                       |                   |                   |
         4/6         3 |                   |                   |
                       |                   |                   |
                       |                   |                   |
         0.5        2.5*---------*---------*---------*---------*
                       |                   |                   |
         2/6         2 |                   |                   |
                       |                   |                   |
         1/6        1.5*         *         *         *         *
                       |                   |                   |
          0          1 |                   |                   |
                       |                   |                   |
        -1/6        0.5*         *         *         *         *
                       |                   |                   |
        -2/6         0 |                   |                   |
                       |                   |                   |
        -0.5       -0.5*---------*---------*---------*---------*
                     -0.5   0   0.5   1   1.5   2   2.5   3   3.5  original pixel indices
                     -0.5 -0.25  0  0.25  0.5  0.75  1   1.25 1.5 resampled pixel indices: factor=2, offset=0

    Test Case 2

    ::

        resampled  original
        factor=3
        offset=2

          0.5      4.5 *-----------*-----------*-----------*-----------*
                                   |                       |
          2/6       4              |                       |
                                   |                       |
          1/6      3.5 *           *           *           *           *
                                   |                       |
           0        3              |                       |
                                   |                       |
         -1/3      2.5 *           *           *           *           *
                                   |                       |
         -2/6       2              |                       |
                                   |                       |
         -0.5      1.5 *-----------*-----------*-----------*-----------*
                                   |                       |
         -4/6       1              |                       |
                                   |                       |
         -5/6      0.5 *           *           *           *           *
                                   |                       |
          -1        0              |                       |
                                   |                       |
         -1-1/6   -0.5 *           *           *           *           *
                     -0.5    0    0.5    1    1.5    2    2.5    3    3.5  original pixel indices
                      -1   -0.75 -0.5  -0.25   0    0.25  0.5   0.75   1   resampled pixel indices: factor=2, offset=1
    """
    wcs = four_five_wcs
    # Get world values of original pixel grid.
    under_pixels = np.meshgrid(np.arange(-0.5, 4, 0.5), np.arange(-0.5, 5, 0.5))
    expected_world = wcs.pixel_to_world_values(*under_pixels)

    # Resample WCS
    new_wcs = ResampledLowLevelWCS(wcs, factor, offset)
    # Get expected pixel coords in resampled WCS of same pixel locations as above.
    output_world = new_wcs.pixel_to_world_values(*expected_over_pixels)
    assert_allclose(np.asarray(output_world), np.asarray(expected_world), atol=1e-15)
