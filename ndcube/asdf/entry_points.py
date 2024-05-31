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
    from ndcube.asdf.converters.ndcube_converter import NDCubeConverter
    from ndcube.asdf.converters.extracoords_converter import ExtraCoordsConverter
    from ndcube.asdf.converters.tablecoord_converter import TimeTableCoordConverter
    from ndcube.asdf.converters.tablecoord_converter import QuantityTableCoordinateConverter
    from ndcube.asdf.converters.tablecoord_converter import SkyCoordTableCoordinateConverter

    ndcube_converters = [NDCubeConverter(),ExtraCoordsConverter(),TimeTableCoordConverter(),QuantityTableCoordinateConverter(),SkyCoordTableCoordinateConverter()]
    _manifest_uri = "asdf://sunpy.org/ndcube/manifests/ndcube-0.1.0"

    return [
        ManifestExtension.from_uri(_manifest_uri, converters=ndcube_converters)
    ]
