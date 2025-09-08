import warnings

from asdf.extension import Converter


class NDCubeConverter(Converter):
    tags = ["tag:sunpy.org:ndcube/ndcube-*"]
    types = ["ndcube.ndcube.NDCube"]

    def from_yaml_tree(self, node, tag, ctx):
        from ndcube.ndcube import NDCube

        ndcube = NDCube(
            node["data"],
            node["wcs"],
            meta=node.get("meta"),
            mask=node.get("mask"),
            unit=node.get("unit"),
            uncertainty=node.get("uncertainty"),
        )
        if "extra_coords" in node:
            ndcube._extra_coords = node["extra_coords"]
        if "global_coords" in node:
            ndcube._global_coords = node["global_coords"]

        return ndcube

    def to_yaml_tree(self, ndcube, tag, ctx):
        """
        Notes
        -----
        This methods serializes the primary components of the NDCube object,
        including the `data`, `wcs`, `extra_coords`, and `global_coords` attributes.
        Issues a warning if unsupported attributes are present.

        Warnings
        --------
        UserWarning
            Warns if the NDCube object has a 'psf' attribute that will not be
            saved in the ASDF serialization.
            This ensures that users are aware of potentially important information
            that is not included in the serialized output.
        """
        from astropy.wcs.wcsapi import BaseHighLevelWCS

        node = {}
        node["data"] = ndcube.data
        if isinstance(ndcube.wcs, BaseHighLevelWCS):
            node["wcs"] = ndcube.wcs.low_level_wcs
        else:
            node["wcs"] = ndcube.wcs
        if not ndcube.extra_coords.is_empty:
            node["extra_coords"] = ndcube.extra_coords
        if ndcube.global_coords._all_coords:
            node["global_coords"] = ndcube.global_coords
        if ndcube.meta:
            node["meta"] = ndcube.meta
        if ndcube.mask is not None:
            node["mask"] = ndcube.mask
        if ndcube.unit is not None:
            node["unit"] = ndcube.unit
        if ndcube.uncertainty is not None:
            node["uncertainty"] = ndcube.uncertainty

        if getattr(ndcube, 'psf') is not None:
            warnings.warn("Attribute 'psf' is present but not being saved in ASDF serialization.", UserWarning)

        return node
