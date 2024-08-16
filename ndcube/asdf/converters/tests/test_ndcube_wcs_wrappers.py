"""
Tests for roundtrip serialization of NDCube with various GWCS types.

TODO: Add tests for the roundtrip serialization of NDCube with ResampledLowLevelWCS, ReorderedLowLevelWCS, and CompoundLowLevelWCS when using astropy.wcs.WCS.
"""

import pytest
from gwcs import __version__ as gwcs_version
from packaging.version import Version

import asdf

from ndcube import NDCube
from ndcube.conftest import data_nd
from ndcube.tests.helpers import assert_cubes_equal
from ndcube.wcs.wrappers import CompoundLowLevelWCS, ReorderedLowLevelWCS, ResampledLowLevelWCS


@pytest.fixture
def create_ndcube_resampledwcs(gwcs_3d_lt_ln_l):
    shape = (2, 3, 4)
    new_wcs = ResampledLowLevelWCS(wcs = gwcs_3d_lt_ln_l, factor=2 ,offset = 1)
    data = data_nd(shape)
    return NDCube(data = data, wcs =new_wcs)


@pytest.mark.skipif(Version(gwcs_version) < Version("0.20"), reason="Requires gwcs>=0.20")
def test_serialization_resampled(create_ndcube_resampledwcs, tmp_path):
    ndc = create_ndcube_resampledwcs
    file_path = tmp_path / "test.asdf"
    with asdf.AsdfFile() as af:
        af["ndcube"] = ndc
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        loaded_ndcube = af["ndcube"]

        loaded_resampledwcs = loaded_ndcube.wcs.low_level_wcs
        resampledwcs = ndc.wcs.low_level_wcs
        assert (loaded_resampledwcs._factor == resampledwcs._factor).all()
        assert (loaded_resampledwcs._offset == resampledwcs._offset).all()

        assert_cubes_equal(loaded_ndcube, ndc)

@pytest.fixture
def create_ndcube_reorderedwcs(gwcs_3d_lt_ln_l):
    shape = (2, 3, 4)
    new_wcs = ReorderedLowLevelWCS(wcs = gwcs_3d_lt_ln_l, pixel_order=[1, 2, 0] ,world_order=[2, 0, 1])
    data = data_nd(shape)
    return NDCube(data = data, wcs =new_wcs)



@pytest.mark.skipif(Version(gwcs_version) < Version("0.20"), reason="Requires gwcs>=0.20")
def test_serialization_reordered(create_ndcube_reorderedwcs, tmp_path):
    ndc = create_ndcube_reorderedwcs
    file_path = tmp_path / "test.asdf"
    with asdf.AsdfFile() as af:
        af["ndcube"] = ndc
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        loaded_ndcube = af["ndcube"]

        loaded_reorderedwcs = loaded_ndcube.wcs.low_level_wcs
        reorderedwcs = ndc.wcs.low_level_wcs
        assert (loaded_reorderedwcs._pixel_order == reorderedwcs._pixel_order)
        assert (loaded_reorderedwcs._world_order == reorderedwcs._world_order)

        assert_cubes_equal(loaded_ndcube, ndc)

@pytest.fixture
def create_ndcube_compoundwcs(gwcs_2d_lt_ln, time_and_simple_extra_coords_2d):

    shape = (1, 2, 3, 4)
    new_wcs = CompoundLowLevelWCS(gwcs_2d_lt_ln, time_and_simple_extra_coords_2d.wcs, mapping = [0, 1, 2, 3])
    data = data_nd(shape)
    return NDCube(data = data, wcs = new_wcs)

@pytest.mark.skipif(Version(gwcs_version) < Version("0.20"), reason="Requires gwcs>=0.20")
def test_serialization_compoundwcs(create_ndcube_compoundwcs, tmp_path):
    ndc = create_ndcube_compoundwcs
    file_path = tmp_path / "test.asdf"
    with asdf.AsdfFile() as af:
        af["ndcube"] = ndc
        af.write_to(file_path)

    with asdf.open(file_path) as af:
        loaded_ndcube = af["ndcube"]
        assert_cubes_equal(loaded_ndcube, ndc)
        assert (loaded_ndcube.wcs.low_level_wcs.mapping.mapping == ndc.wcs.low_level_wcs.mapping.mapping)
        assert (loaded_ndcube.wcs.low_level_wcs.atol == ndc.wcs.low_level_wcs.atol)
