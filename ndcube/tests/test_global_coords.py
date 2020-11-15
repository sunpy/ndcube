import astropy.units as u
import pytest

from ndcube.global_coords import GlobalCoords


@pytest.fixture
def gc():
    return GlobalCoords()


def test_add(gc):
    coord1 = 1 * u.m
    coord2 = 2 * u.s
    gc.add('name1', 'physical_type1', coord1)
    gc.add('name2', 'physical_type2', coord2)
    assert gc.names == ('name1', 'name2')
    assert gc.physical_types == ('physical_type1', 'physical_type2')


def test_remove(gc):
    coord1 = 1 * u.m
    coord2 = 2 * u.s
    gc.add('name1', 'physical_type1', coord1)
    gc.add('name2', 'physical_type2', coord2)
    gc.remove('name2')
    assert len(gc) == 1
    assert gc.names == ('name1',)
    assert gc.physical_types == ('physical_type1',)


def test_overwrite(gc):
    with pytest.raises(ValueError):
        coord1 = 1 * u.m
        coord2 = 2 * u.s
        gc.add('name1', 'physical_type1', coord1)
        gc.add('name1', 'physical_type2', coord2)


def test_iterating(gc):
    coord1 = 1 * u.m
    coord2 = 2 * u.s
    gc.add('name1', 'physical_type1', coord1)
    gc.add('name2', 'physical_type2', coord2)
    for i, gc_item in enumerate(gc):
        if i == 0:
            assert gc_item == 'name1'
        if i == 1:
            assert gc_item == 'name2'


def test_slicing(gc):
    coord1 = 1 * u.m
    gc.add('name1', 'physical_type1', coord1)
    assert gc['name1'] == ('physical_type1', u.Quantity(1., u.m))


def test_names(gc):
    coord1 = 1 * u.m
    coord2 = 2 * u.s
    gc.add('name1', 'physical_type1', coord1)
    gc.add('name2', 'physical_type2', coord2)
    assert gc.names == ('name1', 'name2')


def test_physical_types(gc):
    coord1 = 1 * u.m
    coord2 = 2 * u.s
    gc.add('name1', 'physical_type1', coord1)
    gc.add('name2', 'physical_type2', coord2)
    assert gc.physical_types == ('physical_type1', 'physical_type2')


def test_len(gc):
    coord1 = 1 * u.m
    coord2 = 2 * u.s
    gc.add('name1', 'physical_type1', coord1)
    gc.add('name2', 'physical_type2', coord2)
    assert len(gc) == 2


def test_get_coords(gc):
    coord1 = 1 * u.m
    coord2 = 2 * u.s
    gc.add('name1', 'physical_type1', coord1)
    gc.add('name2', 'physical_type2', coord2)
    value_list = list(gc.values())
    for i, element in enumerate(value_list):
        if i == 0:
            assert element[1] == coord1
        if i == 1:
            assert element[1] == coord2
