import pytest

@pytest.fixture
def sample_ndcube(ndcube_4d_ln_l_t_lt):
    return ndcube_4d_ln_l_t_lt

def test_ellipsis_usage(sample_ndcube):
    sliced_cube = sample_ndcube[..., 1]
    assert sliced_cube.data.shape == (5,10,12)

    with pytest.raises(IndexError, match="An index can only have a single ellipsis"):
        sample_ndcube[..., ..., 1]
