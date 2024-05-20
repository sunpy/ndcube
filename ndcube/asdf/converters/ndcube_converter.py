from asdf.extension import Converter
import numpy as np


class NDCubeConverter(Converter):
    tags = ["tag:sunpy.org:ndcube/ndcube/NDCube-*"]

    @property
    def types(self):
        from ndcube.ndcube import NDCube
        return [NDCube]

    def select_tag(self, obj, tags, ctx):
        # Sort the tags in reverse alphabetical order and pick the first (i.e.
        # the one with the highest version). This assumes that all the tags for
        # this converter are named the same other than the version number.
        tags = list(sorted(tags, reverse=True))
        return tags[0]

    def from_yaml_tree(self, node, tag, ctx):
        from ndcube import ndcube
        data = np.asanyarray(node["data"])
        wcs = node["wcs"]
        return(ndcube.NDCube(data, wcs))

    def to_yaml_tree(self, ndcube, tag, ctx):
        node = {}
        node["data"] = np.asarray(ndcube.data)
        node["wcs"] = ndcube.wcs
        return node
