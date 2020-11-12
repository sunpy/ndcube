import astropy.units as u
from ndcube.global_coords import GlobalCoords

gc = GlobalCoords()  # Initialze an empty GlobalCoords instance

# Init
coord1 = 1 * u.m
coord2 = 2 * u.s


def test_some_basic_coordinates():
    gc.add('name1', 'physical_type1', coord1)
    gc.add('name2', 'physical_type2', coord2)
    assert gc.names == ('name1', 'name2')
    assert gc.physical_types == ('physical_type1', 'physical_type2')
    assert gc.coords == (u.Quantity(1., u.m), u.Quantity(2., u.s))

    gc_name1 = gc['name1']
    assert gc_name1.names == 'name1'
    assert gc_name1.coords == u.Quantity(1., u.m)

    assert gc['name1'].physical_types == 'physical_type1'

    gc.remove('name2')
    assert gc == gc_name1
