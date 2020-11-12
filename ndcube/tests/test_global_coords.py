import astropy.units as u

import ndcube
from ndcube.global_coords import GlobalCoords

gc = GlobalCoords(ndcube)  # Initialize an empty GlobalCoords instance


def test_some_basic_coordinates():
    coord1 = 1 * u.m
    coord2 = 2 * u.s
    gc.add('name1', 'physical_type1', coord1)
    gc.add('name2', 'physical_type2', coord2)
    assert gc.names == ['name1', 'name2']
    assert gc.physical_types == ['physical_type1', 'physical_type2']
    assert gc.coords == [u.Quantity(1., u.m), u.Quantity(2., u.s)]

    gc.remove('name2')
    assert len(gc) == 1
