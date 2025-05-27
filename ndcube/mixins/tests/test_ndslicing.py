import pytest

@pytest.fixture
def sample_ndcube(ndcube_4d_ln_l_t_lt):
    return ndcube_4d_ln_l_t_lt

@pytest.mark.parametrize(("ndc","item","expected_shape"),
                         [
                             ("ndcube_4d_ln_l_t_lt", np.s_[..., 1], (5, 10, 12)),
                             ("ndcube_4d_ln_l_t_lt", np.s_[..., 1:, 1], (5, 10, 11)),
                             ("ndcube_4d_ln_l_t_lt", np.s_[1, ...], (10, 12, 8)),
                             ("ndcube_4d_ln_l_t_lt", np.s_[1, 1:, ...], (9, 12, 8)),
                             ("ndcube_4d_ln_l_t_lt", np.s_[1, ..., 1:], (10, 12, 7)),
                             ("ndcube_4d_ln_l_t_lt", np.s_[1, 1:, ..., 1:], (9, 12, 7)),
                         ],
                         indirect=("ndc",))
def test_ellipsis_usage(ndc):
    sliced_cube = ndc[item]
    assert sliced_cube.data.shape == expected_shape

    with pytest.raises(IndexError, match="An index can only have a single ellipsis"):
        sample_ndcube[..., ..., 1]
