import numpy as np

from asdf.extension import Converter


class NDCubeConverter(Converter):
    tags = ["tag:sunpy.org:ndcube/ndcube/NDCube-*"]
    types = ["ndcube.ndcube.NDCube"]

    def from_yaml_tree(self, node, tag, ctx):
        from ndcube.ndcube import NDCube

        data = np.asanyarray(node["data"])
        wcs = node["wcs"]
        ndcube = NDCube(data, wcs)
        ndcube._extra_coords = node["extra_coords"]

        return ndcube

    def to_yaml_tree(self, ndcube, tag, ctx):
        node = {}
        node["data"] = np.asarray(ndcube.data)
        node["wcs"] = ndcube.wcs
        node["extra_coords"] = ndcube.extra_coords

        return node
