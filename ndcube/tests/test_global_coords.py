import astropy.units as u
import pytest

from ndcube.global_coords import GlobalCoords


@pytest.fixture
def gc():
    return GlobalCoords()


@pytest.fixture
def gc_coords(gc):
    coord1 = 1 * u.m
    coord2 = 2 * u.s
    gc.add('name1', 'custom:physical_type1', coord1)
    gc.add('name2', 'custom:physical_type2', coord2)
    return gc


def test_add(gc):
    coord1 = 1 * u.m
    coord2 = 2 * u.s
    gc.add('name1', 'custom:physical_type1', coord1)
    gc.add('name2', 'custom:physical_type2', coord2)
    assert gc.keys() == {'name1', 'name2'}
    assert gc.physical_types == dict((('name1', 'custom:physical_type1'), ('name2', 'custom:physical_type2')))


def test_remove(gc_coords):
    gc_coords.remove('name2')
    assert len(gc_coords) == 1
    assert gc_coords.keys() == {'name1'}
    assert gc_coords.physical_types == {'name1': 'custom:physical_type1'}


def test_overwrite(gc_coords):
    with pytest.raises(ValueError):
        coord2 = 2 * u.s
        gc_coords.add('name1', 'custom:physical_type2', coord2)


def test_iterating(gc_coords):
    for i, gc_item in enumerate(gc_coords):
        if i == 0:
            assert gc_item == 'name1'
        if i == 1:
            assert gc_item == 'name2'


def test_slicing(gc_coords):
    assert u.allclose(gc_coords['name1'], u.Quantity(1., u.m))


def test_physical_types(gc_coords):
    assert gc_coords.physical_types == dict((('name1', 'custom:physical_type1'), ('name2', 'custom:physical_type2')))


def test_len(gc_coords):
    assert len(gc_coords) == 2


def test_keys(gc_coords):
    assert gc_coords.keys() == {'name1', 'name2'}


def test_values(gc_coords):
    for value, expected in zip(gc_coords.values(), (1 * u.m, 2 * u.s)):
        assert u.allclose(value, expected)


def test_items(gc_coords):
    assert gc_coords.items() == {('name1', 1 * u.m), ('name2', 2 * u.s)}


def test_filter(gc_coords):
    filtered = gc_coords.filter_by_physical_type('custom:physical_type1')
    assert isinstance(filtered, GlobalCoords)
    assert len(filtered) == 1
    assert 'name1' in filtered
    assert u.allclose(filtered['name1'], 1 * u.m)
    assert filtered.physical_types == {'name1': 'custom:physical_type1'}
