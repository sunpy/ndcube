import pytest
from gwcs import __version__ as gwcs_version
from packaging.version import Version

import asdf

from ndcube.ndcollection import NDCollection
from ndcube.ndcube_sequence import NDCubeSequence
from ndcube.tests.helpers import assert_collections_equal


@pytest.fixture
def create_ndcollection_cube(ndcube_gwcs_3d_ln_lt_l, ndcube_gwcs_3d_ln_lt_l_ec_q_t_gc, ndcube_gwcs_3d_ln_lt_l_ec_dropped_dim):
    aligned_axes = ((1, 2), (1, 2), (1, 2))
    cube_collection = NDCollection([("cube0", ndcube_gwcs_3d_ln_lt_l),
                                    ("cube1", ndcube_gwcs_3d_ln_lt_l_ec_q_t_gc),
                                    ("cube2", ndcube_gwcs_3d_ln_lt_l_ec_dropped_dim)],
                                   aligned_axes=aligned_axes)

    return cube_collection


@pytest.mark.skipif(Version(gwcs_version) < Version("0.20"), reason="Requires gwcs>=0.20")
def test_serialization_cube(create_ndcollection_cube, tmp_path):
    ndcollection = create_ndcollection_cube
    file_path = tmp_path / "test.asdf"
    with asdf.AsdfFile() as af:
        af["ndcube_gwcs"] = ndcollection
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        assert_collections_equal(af["ndcube_gwcs"], ndcollection)


@pytest.fixture
def create_ndcollection_sequence(ndcube_gwcs_3d_ln_lt_l, ndcube_gwcs_3d_ln_lt_l_ec_dropped_dim):

    sequence02 = NDCubeSequence([ndcube_gwcs_3d_ln_lt_l, ndcube_gwcs_3d_ln_lt_l_ec_dropped_dim])
    sequence20 = NDCubeSequence([ndcube_gwcs_3d_ln_lt_l_ec_dropped_dim, ndcube_gwcs_3d_ln_lt_l])
    seq_collection = NDCollection([("seq0", sequence02), ("seq1", sequence20)], aligned_axes="all")
    return seq_collection


@pytest.mark.skipif(Version(gwcs_version) < Version("0.20"), reason="Requires gwcs>=0.20")
def test_serialization_sequence(create_ndcollection_sequence, tmp_path):
    ndcollection = create_ndcollection_sequence
    file_path = tmp_path / "test.asdf"
    with asdf.AsdfFile() as af:
        af["ndcube_gwcs"] = ndcollection
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        assert_collections_equal(af["ndcube_gwcs"], ndcollection)
