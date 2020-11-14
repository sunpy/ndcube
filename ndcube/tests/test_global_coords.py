import astropy.units as u
import pytest

from ndcube.global_coords import GlobalCoords


@pytest.fixture
def global_coords(ndcube_3d_ln_lt_l):
    return GlobalCoords(ndcube_3d_ln_lt_l)


def test_adding_global_coords(global_coords):
    coord1 = 1 * u.m
    coord2 = 2 * u.s
    global_coords.add('name1', 'physical_type1', coord1)
    global_coords.add('name2', 'physical_type2', coord2)
    assert global_coords.names == ('name1', 'name2')
    assert global_coords.physical_types == ('physical_type1', 'physical_type2')


def test_removing_global_coords(global_coords):
    coord1 = 1 * u.m
    coord2 = 2 * u.s
    global_coords.add('name1', 'physical_type1', coord1)
    global_coords.add('name2', 'physical_type2', coord2)
    global_coords.remove('name2')
    assert len(global_coords) == 1
    assert global_coords.names == ('name1',)
    assert global_coords.physical_types == ('physical_type1',)


def test_iterating(global_coords):
    coord1 = 1 * u.m
    coord2 = 2 * u.s
    global_coords.add('name1', 'physical_type1', coord1)
    global_coords.add('name2', 'physical_type2', coord2)
    for i, gc_item in enumerate(global_coords.keys()):
        if i == 0:
            assert (i, gc_item) == (0, 'name1')
        if i == 1:
            assert (i, gc_item) == (1, 'name2')


def test_slicing(global_coords):
    coord1 = 1 * u.m
    coord2 = 2 * u.s
    global_coords.add('name1', 'physical_type1', coord1)
    global_coords.add('name2', 'physical_type2', coord2)
    assert global_coords['name1']._all_coords == {'name1': ('physical_type1', u.Quantity(1., u.m))}
    assert global_coords['name2']._all_coords == {'name2': ('physical_type2', u.Quantity(2., u.s))}


def test_dict_keys_and_values(global_coords):
    coord1 = 1 * u.m
    coord2 = 2 * u.s
    global_coords.add('name1', 'physical_type1', coord1)
    global_coords.add('name2', 'physical_type2', coord2)
    assert global_coords.names == ('name1', 'name2')
    assert global_coords.physical_types == ('physical_type1', 'physical_type2')


def test_global_coords_len(global_coords):
    coord1 = 1 * u.m
    coord2 = 2 * u.s
    global_coords.add('name1', 'physical_type1', coord1)
    global_coords.add('name2', 'physical_type2', coord2)
    assert len(global_coords) == 2


def test_global_coords_remove_nonexistent_name(global_coords):
    coord1 = 1 * u.m
    coord2 = 2 * u.s
    global_coords.add('name1', 'physical_type1', coord1)
    global_coords.add('name2', 'physical_type2', coord2)
    with pytest.raises(ValueError):
        global_coords.remove('name0')


def test_global_coords_add_duplicated_name(global_coords):
    coord1 = 1 * u.m
    coord2 = 2 * u.s
    global_coords.add('name1', 'physical_type1', coord1)
    global_coords.add('name1', 'physical_type2', coord2)
    assert global_coords.names == ('name1',)
    assert global_coords.physical_types == ('physical_type2',)
