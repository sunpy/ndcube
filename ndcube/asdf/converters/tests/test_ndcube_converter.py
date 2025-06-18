import asdf_astropy
import numpy as np
import pytest
from packaging.version import Version

import asdf
import astropy.wcs

from ndcube.tests.helpers import assert_cubes_equal


def test_serialization(all_ndcubes, tmp_path, all_ndcubes_names):
    # asdf_astropy doesn't save _naxis before this PR: https://github.com/astropy/asdf-astropy/pull/276
    if Version(asdf_astropy.__version__) < Version("0.8.0") and isinstance(all_ndcubes.wcs, astropy.wcs.WCS):
        all_ndcubes.wcs._naxis = [0, 0]

    if not isinstance(all_ndcubes.data, np.ndarray):
        pytest.skip("Can't save non-numpy array to ASDF.")

    file_path = tmp_path / "test.asdf"
    with asdf.AsdfFile() as af:
        af["ndcube"] = all_ndcubes
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        assert_cubes_equal(af["ndcube"], all_ndcubes)


@pytest.mark.parametrize("expected_cube", ["ndcube_gwcs_3d_ln_lt_l", "ndcube_3d_ln_lt_l"], indirect=True)
def test_serialization_sliced_ndcube(expected_cube, tmp_path):
    # This needs 0.8.0 of asdf_astropy to be able to save gwcs and to save array_shape on WCS
    pytest.importorskip("asdf_astropy", "0.8.0")

    sndc = expected_cube[np.s_[0, :, :]]
    file_path = tmp_path / "test.asdf"
    with asdf.AsdfFile() as af:
        af["ndcube_gwcs"] = sndc
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        assert_cubes_equal(af["ndcube_gwcs"], sndc)
