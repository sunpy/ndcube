import numpy as np
import pytest

import asdf

from ndcube.tests.helpers import assert_cubes_equal


@pytest.mark.parametrize("ndc",[("ndcube_gwcs_2d_ln_lt"),
                                ("ndcube_gwcs_3d_ln_lt_l"),
                                ("ndcube_gwcs_3d_ln_lt_l_ec_dropped_dim"),
                                ("ndcube_gwcs_3d_ln_lt_l_ec_q_t_gc"),
                                ("ndcube_gwcs_3d_rotated"),
                                ("ndcube_gwcs_4d_ln_lt_l_t")
                                ], indirect=("ndc",))
def test_serialization(ndc, tmp_path):
    file_path = tmp_path / "test.asdf"
    with asdf.AsdfFile() as af:
        af["ndcube_gwcs"] = ndc
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        assert_cubes_equal(af["ndcube_gwcs"], ndc)


@pytest.mark.xfail(reason="Serialization of sliced ndcube not supported")
def test_serialization_sliced_ndcube(ndcube_gwcs_3d_ln_lt_l, tmp_path):
    sndc = ndcube_gwcs_3d_ln_lt_l[np.s_[0, :, :]]
    file_path = tmp_path / "test.asdf"
    with asdf.AsdfFile() as af:
        af["ndcube_gwcs"] = sndc
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        assert_cubes_equal(af["ndcube_gwcs"], sndc)


@pytest.mark.xfail(reason="Serialization of ndcube with .wcs attribute as astropy.wcs.wcs.WCS not supported")
def test_serialization_ndcube_wcs(ndcube_3d_ln_lt_l, tmp_path):
    file_path = tmp_path / "test.asdf"
    with asdf.AsdfFile() as af:
        af["ndcube"] = ndcube_3d_ln_lt_l
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        assert_cubes_equal(af["ndcube"], ndcube_3d_ln_lt_l)
