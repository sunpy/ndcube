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
        node = {}
        node["data"] = ndcube.data
        node["wcs"] = ndcube.wcs
        node["extra_coords"] = ndcube.extra_coords
        node["global_coords"] = ndcube.global_coords

        return node
