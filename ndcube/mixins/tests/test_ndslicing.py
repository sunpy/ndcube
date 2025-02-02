import pytest
import numpy as np
from ndcube import NDCube
from astropy.wcs import WCS

@pytest.fixture
def sample_ndcube():
    data = np.random.rand(4, 4, 4)
    wcs = WCS(naxis=3)
    return NDCube(data, wcs)

def test_ellipsis_usage(sample_ndcube):
    sliced_cube = sample_ndcube[..., 1]
    assert sliced_cube.data.shape == (4, 4)

    with pytest.raises(IndexError, match="An index can only have a single ellipsis"):
        sample_ndcube[..., ..., 1]
