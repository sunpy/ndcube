import astropy.units as u
import pytest

from ndcube.global_coords import GlobalCoords


@pytest.fixture
def global_coords(ndcube_3d_ln_lt_l):
    return GlobalCoords(ndcube_3d_ln_lt_l)


def test_some_basic_coordinates(global_coords):
    coord1 = 1 * u.m
    coord2 = 2 * u.s
    global_coords.add('name1', 'physical_type1', coord1)
    global_coords.add('name2', 'physical_type2', coord2)
    assert global_coords.names == ('name1', 'name2')
    assert global_coords.physical_types == ('physical_type1', 'physical_type2')

    global_coords.remove('name2')
    assert len(global_coords) == 1


def test_iterating(global_coords):
    coord1 = 1 * u.m
    coord2 = 2 * u.s
    global_coords.add('name1', 'physical_type1', coord1)
    global_coords.add('name2', 'physical_type2', coord2)
    iter_obj = iter(global_coords)
    iter_list = []
    while True:
        try:
            # get the next item
            element = next(iter_obj)
            # do something with element
            iter_list.append(element)
        except StopIteration:
            # if StopIteration is raised, break from loop
            break
    assert iter_list == ['name1', 'name2']


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
