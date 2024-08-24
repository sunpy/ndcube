import pytest
from gwcs import __version__ as gwcs_version
from packaging.version import Version

import asdf

from ndcube.ndcube_sequence import NDCubeSequence
from ndcube.tests.helpers import assert_cubesequences_equal


@pytest.mark.skipif(Version(gwcs_version) < Version("0.20"), reason="Requires gwcs>=0.20")
def test_serialization(ndcube_gwcs_3d_ln_lt_l, ndcube_gwcs_3d_ln_lt_l_ec_q_t_gc, tmp_path):
    file_path = tmp_path / "test.asdf"
    ndcseq = NDCubeSequence([ndcube_gwcs_3d_ln_lt_l, ndcube_gwcs_3d_ln_lt_l_ec_q_t_gc], common_axis=1)
    with asdf.AsdfFile() as af:
        af["ndcube_gwcs"] = ndcseq
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        assert_cubesequences_equal(af["ndcube_gwcs"], ndcseq)
