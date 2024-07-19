import warnings

from asdf.extension import Converter


class NDCubeConverter(Converter):
    tags = ["tag:sunpy.org:ndcube/ndcube/ndcube-*"]
    types = ["ndcube.ndcube.NDCube"]

    def from_yaml_tree(self, node, tag, ctx):
        from ndcube.ndcube import NDCube

        ndcube = NDCube(node["data"], node["wcs"])
        ndcube._extra_coords = node["extra_coords"]
        ndcube._global_coords = node["global_coords"]

        return ndcube

    def to_yaml_tree(self, ndcube, tag, ctx):
        """
        Notes
        -----
        This methods serializes the primary components of the NDCube object,
        including the `data`, `wcs`, `extra_coords`, and `global_coords` attributes.
        Issues a warning if unsupported attributes (uncertainty, mask, meta, unit) are present,
        which are not currently serialized to ASDF.

        Warnings
        --------
        UserWarning
            Warns if the NDCube object has attributes 'uncertainty', 'mask', 'meta',
            or 'unit' that are present but not being saved in the ASDF serialization.
            This ensures that users are aware of potentially important information
            that is not included in the serialized output.
        """
        node = {}
        node["data"] = ndcube.data
        node["wcs"] = ndcube.wcs
        node["extra_coords"] = ndcube.extra_coords
        node["global_coords"] = ndcube.global_coords

        attributes = ['uncertainty', 'mask', 'unit']
        for attr in attributes:
            if getattr(ndcube, attr) is not None:
                warnings.warn(f"Attribute '{attr}' is present but not being saved in ASDF serialization.", UserWarning)

        if len(ndcube.meta) > 0:
            warnings.warn("Attribute 'meta' is present but not being saved in ASDF serialization.", UserWarning)

        return node
