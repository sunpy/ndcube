"""
This file contains the entry points for asdf.
"""
import importlib.resources as importlib_resources

from asdf.extension import ManifestExtension
from asdf.resource import DirectoryResourceMapping


def get_resource_mappings():
    """
    Get the resource mapping instances for myschemas
    and manifests.  This method is registered with the
    asdf.resource_mappings entry point.

    Returns
    -------
    list of collections.abc.Mapping
    """
    from ndcube.asdf import resources
    resources_root = importlib_resources.files(resources)
    return [
        DirectoryResourceMapping(
            resources_root / "schemas", "asdf://sunpy.org/ndcube/schemas/"),
        DirectoryResourceMapping(
            resources_root / "manifests", "asdf://sunpy.org/ndcube/manifests/"),
    ]


def get_extensions():
    """
    Get the list of extensions.
    """
    from ndcube.asdf.converters.compoundwcs_converter import CompoundConverter
    from ndcube.asdf.converters.extracoords_converter import ExtraCoordsConverter
    from ndcube.asdf.converters.globalcoords_converter import GlobalCoordsConverter
    from ndcube.asdf.converters.ndcollection_converter import NDCollectionConverter
    from ndcube.asdf.converters.ndcube_converter import NDCubeConverter
    from ndcube.asdf.converters.ndcubesequence_converter import NDCubeSequenceConverter
    from ndcube.asdf.converters.reorderedwcs_converter import ReorderedConverter
    from ndcube.asdf.converters.resampled_converter import ResampledConverter
    from ndcube.asdf.converters.tablecoord_converter import (
        QuantityTableCoordinateConverter,
        SkyCoordTableCoordinateConverter,
        TimeTableCoordConverter,
    )
    ndcube_converters = [
        NDCubeConverter(),
        ExtraCoordsConverter(),
        TimeTableCoordConverter(),
        QuantityTableCoordinateConverter(),
        SkyCoordTableCoordinateConverter(),
        GlobalCoordsConverter(),
        ResampledConverter(),
        ReorderedConverter(),
        CompoundConverter(),
        NDCubeSequenceConverter(),
        NDCollectionConverter(),
        ]
    _manifest_uri = "asdf://sunpy.org/ndcube/manifests/ndcube-0.1.0"

    return [
        ManifestExtension.from_uri(_manifest_uri, converters=ndcube_converters)
    ]
