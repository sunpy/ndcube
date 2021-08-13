import numpy as np
import pytest

from ndcube.meta import Meta
from .helpers import assert_metas_equal


# Fixtures

@pytest.fixture
def basic_meta_values():
    return {"a": "hello",
            "b": list(range(10, 25, 10)),
            "c": np.array([[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400]]),
            "d": list(range(3, 13, 3))
            }


@pytest.fixture
def basic_comments():
    return {"a": "Comment A",
            "b": "Comment B",
            "c": "Comment C",
            }


@pytest.fixture
def basic_axes():
    return {"b": 0,
            "c": (1, 2),
            "d": (2,),
           }


@pytest.fixture
def basic_data_shape():
    return (2, 3, 4, 5)
  

@pytest.fixture
def basic_meta(basic_meta_values, basic_comments, basic_axes, basic_data_shape):
    return Meta(basic_meta_values, basic_comments, basic_axes, basic_data_shape)


@pytest.fixture
def no_shape_meta():
    return Meta({"a": "hello"})


def test_meta_values(basic_meta, basic_meta_values):
    meta = basic_meta
    expected_values = list(basic_meta_values.values())
    assert meta.meta_values == expected_values 


def test_comments(basic_meta, basic_comments):
    meta = basic_meta
    comments = basic_comments
    assert list(meta.comments.keys()) == list(comments.keys())
    assert list(meta.comments.values()) == list(comments.values())


def test_axes(basic_meta, basic_axes):
    meta = basic_meta
    axes = basic_axes
    axes["b"] = np.array([0])
    axes["c"] = np.asarray(axes["c"])
    axes["d"] = np.asarray(axes["d"])
    assert list(meta.axes.keys()) == list(axes.keys())
    for output_axis, expected_axis in zip(meta.axes.values(), axes.values()):
        assert all(output_axis == expected_axis)


def test_shape(basic_meta, basic_data_shape):
    meta = basic_meta
    shape = np.asarray(basic_data_shape)
    assert all(meta.shape == shape)


def test_slice_axis_with_no_meta(basic_meta):
    meta = basic_meta
    output = meta[:, :, :, 0]
    assert_metas_equal(output, meta)


def test_slice_away_independent_axis(basic_meta):
    meta = basic_meta
    # Get output
    sliced_axis = 0
    item = 0
    output = meta[item]
    # Build expected result.
    values = dict([(key, value[0]) for key, value in meta.items()])
    values["b"] = values["b"][0]
    comments = meta.comments
    axes = dict([(key, axis) for key, axis in meta.axes.items()])
    del axes["b"]
    axes["c"] -= 1
    axes["d"] -= 1
    shape = meta.shape[1:]
    print(values, comments, axes, shape)
    expected = Meta(values, comments, axes, shape)
    # Compare output and expected.
    assert_metas_equal(output, expected)


def test_slice_dependent_axes(basic_meta):
    meta = basic_meta
    print(meta["a"])
    # Get output
    output = meta[:, 1:3, 1]
    print(meta["a"])
    # Build expected result.
    values = dict([(key, value[0]) for key, value in meta.items()])
    values["c"] = values["c"][1:3, 1]
    values["d"] = values["d"][1]
    comments = meta.comments
    axes = dict([(key, axis) for key, axis in meta.axes.items()])
    axes["c"] = 1
    del axes["d"]
    shape = np.array([2, 2, 5])
    expected = Meta(values, comments, axes, shape)
    # Compare output and expected.
    assert_metas_equal(output, expected)


@pytest.mark.parametrize("meta, item, expected",
                         (
                             ("basic_meta", "a", "hello"),
                             ("basic_meta", "b", list(range(10, 25, 10))),
                         ),
                         indirect=("meta",))
def test_slice_by_str(meta, item, expected):
    meta = basic_meta
    assert meta["a"] == "hello"
    assert meta["b"] == list(range(10, 25, 10))


def test_add1(basic_meta):
    meta = basic_meta
    name = "z"
    value = 100
    comment = "Comment E"
    meta.add(name, value, comment=comment)
    assert name in meta.keys()
    assert meta[name] == value
    assert meta.comments[name] == comment
    assert name not in meta.axes.keys()


def test_add2(basic_meta):
    meta = basic_meta
    name = "z"
    value = list(range(2))
    axis = 0
    meta.add(name, value, axis=axis)
    assert name in meta.keys()
    assert meta[name] == value
    assert meta.axes[name] == np.array([axis])
    assert name not in meta.comments.keys()


def test_add_overwrite_error(basic_meta):
    meta = basic_meta
    with pytest.raises(KeyError):
        meta.add("a", "world")


def test_add_axis_without_shape(no_shape_meta):
    meta = no_shape_meta
    with pytest.raises(TypeError):
        meta.add("z", [100], axis=0)
