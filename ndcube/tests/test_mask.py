import numpy as np
import pytest

from ndcube.mask import Mask


@pytest.fixture
def mask_simple():
    m =  Mask([("a", np.array([True, False, False])),
               ("b", np.array([True, False, False])),
               ("c", np.array([True, True, False]))])
    m.deactivate("b")
    return m


def test_active_masks(mask_simple):
    assert mask_simple.active_masks == ("a", "c")


def test_mask(mask_simple):
    expected = np.array([True, True, False])
    assert all(mask_simple.mask == expected)


def test_names(mask_simple):
    assert mask_simple.names == ("a", "b", "c")


def test_shape(mask_simple):
    assert mask_simple.shape == (3,)


def test_add(mask_simple):
    m = mask_simple
    d = np.array([True] * m.shape[0])
    e = np.array([False] * m.shape[0])
    m.add("d", d, activate=True)
    assert "d" in m
    assert all(m["d"] == d)
    assert m.is_active("d")
    m.add("e", e, activate=False)
    assert m.is_active("e") is False


def test_remove(mask_simple):
    m = mask_simple
    name = "a"
    m.remove(name)
    assert name not in m
    assert name not in m.active_masks


def test_activate(mask_simple):
    m = mask_simple
    name = "b"
    assert m.is_active(name) is False
    m.activate(name)
    assert m.is_active(name) is True


def test_deactivate(mask_simple):
    m = mask_simple
    name = "c"
    assert m.is_active(name) is True
    m.deactivate(name)
    assert m.is_active(name) is False


def test_is_active(mask_simple):
    m = mask_simple
    assert m.is_active("a") is True
    assert m.is_active("b") is False


def test_index_by_string(mask_simple):
    assert all(mask_simple["a"] == np.array([True, False, False]))


def test_slice(mask_simple):
    m = mask_simple
    item = slice(1, None)
    output = m[item]
    expected = Mask([("a", np.array([False, False])),
                     ("b", np.array([False, False])),
                     ("c", np.array([True, False]))])
    expected.deactivate("b")
    assert_masks_equal(output, expected)


def test_contains(mask_simple):
    m = mask_simple
    assert "a" in m
    assert "d" not in m


def test_len(mask_simple):
    assert len(mask_simple) == 3


def assert_masks_equal(mask1, mask2):
    assert mask1.names == mask2.names
    assert mask1._active.keys() == mask2._active.keys()
    assert mask1.active_masks == mask2.active_masks
    for key in mask1:
        assert all(mask1[key] == mask2[key])
