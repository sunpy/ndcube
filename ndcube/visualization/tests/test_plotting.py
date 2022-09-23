import astropy.units as u
import matplotlib.pyplot as plt
import mpl_animators
import numpy as np
import pytest
from astropy.visualization.wcsaxes import WCSAxes
from astropy.wcs import WCS

from ndcube.ndcube import NDCube
from ndcube.tests.helpers import figure_test
from ndcube.visualization import PlotterDescriptor


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
def test_plot_2D_cube_custom_axis(ndcube_2d_ln_lt):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=ndcube_2d_ln_lt.wcs)
    ndcube_2d_ln_lt.plot(axes=ax)
    return fig


@figure_test
def test_plot_2D_cube_custom_axis_plot_axes(ndcube_2d_ln_lt):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=ndcube_2d_ln_lt.wcs)
    ndcube_2d_ln_lt.plot(axes=ax, plot_axes=('x', 'y'))
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
def test_animate_2D_cube(ndcube_2d_ln_lt):
    cube = ndcube_2d_ln_lt
    ax = cube.plot(plot_axes=[None, 'x'])
    assert isinstance(ax, mpl_animators.ArrayAnimatorWCS)

    return ax.fig


@figure_test
@pytest.mark.parametrize(("ndcube_4d", "cslice", "kwargs"),
                         (
                             ("ln_lt_l_t", np.s_[:, :, 0, :], {}),
                             ("ln_lt_l_t", np.s_[:, :, 0, :], {'plot_axes': [..., 'x']}),
                             ("ln_lt_l_t", None, {}),
                             ("ln_lt_l_t", None, {"plot_axes": [0, 0, 'x', 'y'], "axes_units": [None, None, u.pm, None]}),
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
    assert isinstance(ax, mpl_animators.ArrayAnimatorWCS)

    return ax.fig


@pytest.mark.parametrize(("ndcube_4d", "cslice"),
                         [("ln_lt_l_t", np.s_[:, :, 0, 0])], indirect=["ndcube_4d"])
def test_mpl_axes(ndcube_4d, cslice):
    ndcube_2d = ndcube_4d[cslice]
    ax = plt.subplot(projection=ndcube_2d)
    assert isinstance(ax, WCSAxes)
    plt.close()


def test_plotter_is_None(ndcube_1d_l):
    class NewCube(NDCube):
        plotter = PlotterDescriptor(default_type=None)

    cube = NewCube(np.zeros((1, 1)), wcs=WCS(naxis=2))
    assert cube.plotter is None

    with pytest.raises(NotImplementedError, match="no default plotting functionality is available"):
        cube.plot()

    # You can't (and shouldn't) set the plotter to None unless it's done at
    # descriptor init time:
    with pytest.raises(TypeError):
        ndcube_1d_l.plotter = None
