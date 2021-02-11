import astropy.units as u
import pytest

import ndcube.visualization.plotting_utils as utils


@pytest.mark.parametrize("ndim, plist, output", (
    (2, ['x', 'y'], ['x', 'y']),
    (2, [..., 'x', 'y'], ['x', 'y']),
    (2, ['x', 'y', ...], ['x', 'y']),
    (3, ['x', ...], ['x', None, None]),
    (4, ['x', ..., 'y'], ['x', None, None, 'y']),
    (5, [..., 'x'], [None, None, None, None, 'x']),
    (5, [..., 'x', None], [None, None, None, 'x', None]),
    (5, [None, ..., 'x', None, 'y'], [None, None, 'x', None, 'y']),
))
def test_expand_ellipsis(ndim, plist, output):
    result = utils._expand_ellipsis(ndim, plist)
    assert result == output


def test_expand_ellipsis_error():
    with pytest.raises(IndexError):
        utils._expand_ellipsis(1, (..., 'x', ...))


def test_prep_plot_kwargs_errors(ndcube_4d_ln_lt_l_t):
    """
    Check a whole bunch of different error conditions.
    """
    # plot_axes has incorrect length
    with pytest.raises(ValueError):
        utils.prep_plot_kwargs(4, ndcube_4d_ln_lt_l_t.wcs, ['wibble'], None, None)

    # axes_coordinates is not in world_axis_physical_types
    with pytest.raises(ValueError):
        utils.prep_plot_kwargs(4, ndcube_4d_ln_lt_l_t.wcs, None, [..., "wibble"], None)

    # axes_coordinates has incorrect type
    with pytest.raises(TypeError):
        utils.prep_plot_kwargs(4, ndcube_4d_ln_lt_l_t.wcs, None, [..., 10], None)

    # axes_units has incorrect length
    with pytest.raises(ValueError):
        utils.prep_plot_kwargs(4, ndcube_4d_ln_lt_l_t.wcs, None, None, ['m'])

    # axes_units has incorrect type
    with pytest.raises(TypeError):
        utils.prep_plot_kwargs(4, ndcube_4d_ln_lt_l_t.wcs, None, None, [[], ...])

    with pytest.raises(u.UnitsError):
        utils.prep_plot_kwargs(4, ndcube_4d_ln_lt_l_t.wcs, None, None, [u.eV, u.m, u.m, u.m])


@pytest.mark.parametrize("ndcube_2d, args, output", (
    ("ln_lt",
     (None, None, None),
     (['x', 'y'], None, None)),
    ("ln_lt",
     (None, [..., 'custom:pos.helioprojective.lon'], None),
     (['x', 'y'], ['custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'], None)),
    ("ln_lt",
     (None, None, [u.deg, 'arcsec']),
     (['x', 'y'], None, [u.arcsec, u.deg])),
), indirect=['ndcube_2d'])
def test_prep_plot_kwargs(ndcube_2d, args, output):
    result = utils.prep_plot_kwargs(2, ndcube_2d.wcs, *args)
    assert result == output
