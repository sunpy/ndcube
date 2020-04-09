import matplotlib.pyplot as plt
import numpy as np
import pytest

import astropy.units as u
import sunpy.visualization.animator
from astropy.visualization.wcsaxes import WCSAxes


@pytest.mark.mpl_image_compare
def test_plot_1D_cube(ndcube_1d_simple):
    fig = plt.figure()
    ax = ndcube_1d_simple.plot()
    assert isinstance(ax, WCSAxes)
    return fig


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize(("ndcube_4d", "cslice", "kwargs"),
                         (
                             ("simple", np.s_[0,0,0,:], {}),
                             ("simple", np.s_[0,0,:,0], {}),
                             ("simple", np.s_[0,:,0,0], {}),
                             ("simple", np.s_[:,0,0,0], {}),

                             ("uncertainty", np.s_[0,0,0,:], {}),
                             ("unit_uncertainty", np.s_[0,0,0,:], {'data_unit': u.mJ}),

                             ("mask", np.s_[0,0,0,:], {'marker': 'o'}),
                         ),
                         indirect=["ndcube_4d"])
def test_plot_1D_cube_from_slice(ndcube_4d, cslice, kwargs):
    # TODO: The output for the spatial plots is inconsistent between the lat
    # slice and the lon slice.
    fig = plt.figure()

    sub = ndcube_4d[cslice]
    ax = sub.plot(**kwargs)
    assert isinstance(ax, WCSAxes)

    return fig


@pytest.mark.mpl_image_compare
def test_plot_2D_cube(ndcube_1d_simple):
    fig = plt.figure()
    ax = ndcube_1d_simple.plot()
    assert isinstance(ax, WCSAxes)
    return fig


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize(("ndcube_4d", "cslice", "kwargs"),
                         (
                             ("simple", np.s_[0,0,:,:], {}),
                             ("simple", np.s_[0,:,:,0], {}),
                             ("simple", np.s_[:,:,0,0], {}),

                             ("unit_uncertainty", np.s_[0,0,:,:], {'data_unit': u.mJ}),

                             ("mask", np.s_[0,:,0,:], {}),
                         ),
                         indirect=["ndcube_4d"])
def test_plot_2D_cube_from_slice(ndcube_4d, cslice, kwargs):
    fig = plt.figure()

    sub = ndcube_4d[cslice]
    ax = sub.plot(**kwargs)
    assert isinstance(ax, WCSAxes)

    return fig


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize(("ndcube_4d", "cslice", "kwargs", "bugged"),
                         (
                             ("simple", np.s_[:,:,0,:], {}, False),
                             ("simple", np.s_[:,:,0,:], {'plot_axes': [..., 'x']}, False),
                             ("simple", None, {}, False),
                             ("simple", None, {"plot_axes": [0,0,'x','y']}, False),
                             ("simple", None, {"plot_axes": [0,'x',0,'y']}, False),
                             ("simple", np.s_[0,:,:,:], {}, True),
                             ("simple", np.s_[:,:,:,:], {}, False),
                             ("unit_uncertainty", np.s_[0,:,:,:], {'data_unit': u.mJ}, True),
                             ("mask", np.s_[:,:,:,:], {}, False),
                         ),
                         indirect=["ndcube_4d"])
def test_animate_cube_from_slice(ndcube_4d, cslice, kwargs, bugged):
    if bugged:
        # Some of these require https://github.com/sunpy/sunpy/pull/3990
        pytest.importorskip("sunpy", minversion="1.1.3")

    if cslice:
        sub = ndcube_4d[cslice]
    else:
        sub = ndcube_4d
    ax = sub.plot(**kwargs)
    assert isinstance(ax, sunpy.visualization.animator.ArrayAnimatorWCS)

    return ax.fig
