import astropy.units as u

from ndcube.global_coords import GlobalCoords


@pytest.fixture
def global_coords(ndcube_3d_ln_lt_l):
    return GlobalCoords(ndcube_3d_ln_lt_l)


def test_some_basic_coordinates(global_coords):
    coord1 = 1 * u.m
    coord2 = 2 * u.s
    global_coords.add('name1', 'physical_type1', coord1)
    global_coords.add('name2', 'physical_type2', coord2)
    assert global_coords.names == ['name1', 'name2']
    assert global_coords.physical_types == ['physical_type1', 'physical_type2']
    assert global_coords.coords == [u.Quantity(1., u.m), u.Quantity(2., u.s)]

    global_coords.remove('name2')
    assert len(global_coords) == 1
