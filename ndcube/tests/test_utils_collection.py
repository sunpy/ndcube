
import astropy.units as u
import pytest

from ndcube.utils import collection as collection_utils


@pytest.mark.parametrize("data_dimensions1,data_dimensions2,data_axes1,data_axes2", [
    ([3., 4., 5.]*u.pix, [3., 5., 15.]*u.pix, (0, 2), (0, 1))])
def test_assert_aligned_axes_compatible(data_dimensions1, data_dimensions2,
                                        data_axes1, data_axes2):
    collection_utils.assert_aligned_axes_compatible(data_dimensions1, data_dimensions2,
                                                    data_axes1, data_axes2)


@pytest.mark.parametrize("data_dimensions1,data_dimensions2,data_axes1,data_axes2", [
    ([3., 4., 5.]*u.pix, [3., 5., 15.]*u.pix, (0, 1), (0, 1)),
    ([3., 4., 5.]*u.pix, [3., 5., 15.]*u.pix, (0, 1), (0, 1, 2))])
def test_assert_aligned_axes_compatible_error(data_dimensions1, data_dimensions2,
                                              data_axes1, data_axes2):
    with pytest.raises(ValueError):
        collection_utils.assert_aligned_axes_compatible(data_dimensions1, data_dimensions2,
                                                        data_axes1, data_axes2)
