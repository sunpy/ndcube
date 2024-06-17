from typing import OrderedDict

from asdf.extension import Converter


class GlobalCoordsConverter(Converter):
    tags = ["tag:sunpy.org:ndcube/global_coords/GlobalCoords-*"]
    types = ["ndcube.global_coords.GlobalCoords"]

    def from_yaml_tree(self, node, tag, ctx):
        from ndcube.global_coords import GlobalCoords

        globalcoords = GlobalCoords()
        if node.get("internal_coords") is not None:
            globalcoords._internal_coords = OrderedDict(node.get("internal_coords"))
        globalcoords._ndcube = node["ndcube"]

        return globalcoords

    def to_yaml_tree(self, globalcoords, tag, ctx):
        node = {}
        node["ndcube"] = globalcoords._ndcube
        if globalcoords._internal_coords:
            node["internal_coords"] = dict(globalcoords._internal_coords)
        # Todo: Include `_all_coords` as a node key to preserve the dropped dimensions
        # after ndcube support serialization of sliced NDCube object to asdf.

        return node
