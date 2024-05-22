from asdf.extension import Converter
import numpy as np


class NDCubeConverter(Converter):
    tags = ["tag:sunpy.org:ndcube/ndcube/NDCube-*"]
    types = ["ndcube.ndcube.NDCube"]

    def from_yaml_tree(self, node, tag, ctx):
        from ndcube import ndcube
        data = np.asanyarray(node["data"])
        wcs = node["wcs"]
        return(ndcube.NDCube(data, wcs))

    def to_yaml_tree(self, ndcube, tag, ctx):
        node = {}
        node["data"] =np.asarray(ndcube.data)
        node["wcs"] = ndcube.wcs
        return node
