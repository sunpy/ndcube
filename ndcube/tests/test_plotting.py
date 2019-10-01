import pytest
import numpy as np

import astropy.units as u
from astropy.visualization.wcsaxes import WCSAxes

import matplotlib.pyplot as plt


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
    # TODO: The output for the spatial plots is inconsistent between the lat
    # slice and the lon slice.
    fig = plt.figure()

    sub = ndcube_4d[cslice]
    ax = sub.plot(**kwargs)
    assert isinstance(ax, WCSAxes)

    return fig
