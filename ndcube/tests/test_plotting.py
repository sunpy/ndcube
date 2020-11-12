import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
import sunpy.visualization.animator
from astropy.visualization.wcsaxes import WCSAxes

from ndcube.tests.helpers import figure_test


@figure_test
def test_plot_1D_cube(ndcube_1d_l):
    fig = plt.figure()
    ax = ndcube_1d_l.plot()
    assert isinstance(ax, WCSAxes)
    return fig


@figure_test
@pytest.mark.parametrize(("ndcube_4d", "cslice", "kwargs"),
                         (
                             ("ln_lt_l_t", np.s_[0, 0, 0, :], {}),
                             ("ln_lt_l_t", np.s_[0, 0, :, 0], {}),
                             ("ln_lt_l_t", np.s_[0, :, 0, 0], {}),
                             ("ln_lt_l_t", np.s_[:, 0, 0, 0], {}),

                             ("uncertainty", np.s_[0, 0, 0, :], {}),
                             ("unit_uncertainty", np.s_[0, 0, 0, :], {'data_unit': u.mJ}),

                             ("mask", np.s_[0, 0, 0, :], {'marker': 'o'}),),
                         indirect=["ndcube_4d"])
def test_plot_1D_cube_from_slice(ndcube_4d, cslice, kwargs):
    # TODO: The output for the spatial plots is inconsistent between the lat
    # slice and the lon slice.
    fig = plt.figure()

    sub = ndcube_4d[cslice]
    ax = sub.plot(**kwargs)
    assert isinstance(ax, WCSAxes)

    return fig


@figure_test
def test_plot_2D_cube(ndcube_2d_ln_lt):
    fig = plt.figure()
    ax = ndcube_2d_ln_lt.plot()
    assert isinstance(ax, WCSAxes)
    return fig


@figure_test
def test_plot_2D_cube_colorbar(ndcube_2d_ln_lt):
    fig = plt.figure()
    ax = ndcube_2d_ln_lt.plot()
    assert isinstance(ax, WCSAxes)
    plt.colorbar()
    return fig


@figure_test
@pytest.mark.parametrize(("ndcube_4d", "cslice", "kwargs"),
                         (
                             ("ln_lt_l_t", np.s_[0, 0, :, :], {}),
                             ("ln_lt_l_t", np.s_[0, :, :, 0], {}),
                             ("ln_lt_l_t", np.s_[:, :, 0, 0], {}),
                             ("unit_uncertainty", np.s_[0, 0, :, :], {'data_unit': u.mJ}),
                             ("mask", np.s_[0, :, 0, :], {}),),
                         indirect=["ndcube_4d"])
def test_plot_2D_cube_from_slice(ndcube_4d, cslice, kwargs):
    fig = plt.figure()

    sub = ndcube_4d[cslice]
    ax = sub.plot(**kwargs)
    assert isinstance(ax, WCSAxes)

    return fig


@figure_test
@pytest.mark.parametrize(("ndcube_4d", "cslice", "kwargs"),
                         (
                             ("ln_lt_l_t", np.s_[:, :, 0, :], {}),
                             ("ln_lt_l_t", np.s_[:, :, 0, :], {'plot_axes': [..., 'x']}),
                             ("ln_lt_l_t", None, {}),
                             ("ln_lt_l_t", None, {"plot_axes": [0, 0, 'x', 'y']}),
                             ("ln_lt_l_t", None, {"plot_axes": [0, 'x', 0, 'y']}),
                             ("ln_lt_l_t", np.s_[0, :, :, :], {}),
                             ("ln_lt_l_t", np.s_[:, :, :, :], {}),
                             ("unit_uncertainty", np.s_[0, :, :, :], {'data_unit': u.mJ}),
                             ("mask", np.s_[:, :, :, :], {}),),
                         indirect=["ndcube_4d"])
def test_animate_cube_from_slice(ndcube_4d, cslice, kwargs):
    if cslice:
        sub = ndcube_4d[cslice]
    else:
        sub = ndcube_4d
    ax = sub.plot(**kwargs)
    assert isinstance(ax, sunpy.visualization.animator.ArrayAnimatorWCS)

    return ax.fig


@pytest.mark.parametrize(("ndcube_4d", "cslice"),
                         [("ln_lt_l_t", np.s_[:, :, 0, 0])], indirect=["ndcube_4d"])
def test_mpl_axes(ndcube_4d, cslice):
    ndcube_2d = ndcube_4d[cslice]
    ax = plt.subplot(projection=ndcube_2d)
    assert isinstance(ax, WCSAxes)
    plt.close()
