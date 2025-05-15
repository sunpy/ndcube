import numpy as np
import pytest
from gwcs import __version__ as gwcs_version
from packaging.version import Version

import asdf

from ndcube.tests.helpers import assert_cubes_equal


@pytest.mark.skipif(Version(gwcs_version) < Version("0.20"), reason="Requires gwcs>=0.20")
def test_serialization(all_ndcubes, tmp_path):
    if not isinstance(all_ndcubes.data, np.ndarray):
        pytest.skip("Can't save non-numpy array to ASDF.")

    file_path = tmp_path / "test.asdf"
    with asdf.AsdfFile() as af:
        af["ndcube"] = all_ndcubes
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        assert_cubes_equal(af["ndcube"], all_ndcubes)


def test_serialization_sliced_ndcube(ndcube_gwcs_3d_ln_lt_l, tmp_path):
    sndc = ndcube_gwcs_3d_ln_lt_l[np.s_[0, :, :]]
    file_path = tmp_path / "test.asdf"
    with asdf.AsdfFile() as af:
        af["ndcube_gwcs"] = sndc
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        assert_cubes_equal(af["ndcube_gwcs"], sndc)
