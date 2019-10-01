import pytest

import astropy.units as u

import ndcube.mixins.plotting_utils as utils


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


def test_prep_plot_kwargs_errors(ndcube_4d_simple):
    """
    Check a whole bunch of different error conditions.
    """
    # plot_axes has incorrect length
    with pytest.raises(ValueError):
        utils.prep_plot_kwargs(ndcube_4d_simple, ['wibble'], None, None)

    # axes_cooordinates has incorrect length
    with pytest.raises(ValueError):
        utils.prep_plot_kwargs(ndcube_4d_simple, None, [1], None)

    # axes_coordinates is not in world_axis_physical_types
    with pytest.raises(ValueError):
        utils.prep_plot_kwargs(ndcube_4d_simple, None, [..., "wibble"], None)

    # axes_coordinates has incorrect type
    with pytest.raises(TypeError):
        utils.prep_plot_kwargs(ndcube_4d_simple, None, [..., 10], None)

    # axes_units has incorrect length
    with pytest.raises(ValueError):
        utils.prep_plot_kwargs(ndcube_4d_simple, None, None, ['m'])

    # axes_units has incorrect type
    with pytest.raises(TypeError):
        utils.prep_plot_kwargs(ndcube_4d_simple, None, None, [[], ...])

    with pytest.raises(u.UnitsError):
        utils.prep_plot_kwargs(ndcube_4d_simple, None, None, [u.eV, u.m, u.m, u.m])

@pytest.mark.parametrize("ndcube_2d, args, output", (
    ("simple",
     (None, None, None),
     (['x', 'y'], None, None)),
    ("simple",
     (None, [..., 'custom:pos.helioprojective.lat'], None),
     (['x', 'y'], [None, 'custom:pos.helioprojective.lat'], None)),
    ("simple",
     (None, None, ['arcsec', u.deg]),
     (['x', 'y'], None, [u.arcsec, u.deg])),
    ),
                         indirect=['ndcube_2d'])
def test_prep_plot_kwargs(ndcube_2d, args, output):
    result = utils.prep_plot_kwargs(ndcube_2d, *args)
    assert result == output
