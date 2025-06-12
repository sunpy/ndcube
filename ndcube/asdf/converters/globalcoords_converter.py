from asdf.extension import Converter


class GlobalCoordsConverter(Converter):
    tags = ["tag:sunpy.org:ndcube/global_coords/globalcoords-*"]
    types = ["ndcube.global_coords.GlobalCoords"]

    def from_yaml_tree(self, node, tag, ctx):
        from ndcube.global_coords import GlobalCoords

        globalcoords = GlobalCoords()
        if "internal_coords" in node:
            globalcoords._internal_coords = node["internal_coords"]
        globalcoords._ndcube = node["ndcube"]

        return globalcoords

    def to_yaml_tree(self, globalcoords, tag, ctx):
        node = {}
        node["ndcube"] = globalcoords._ndcube
        if globalcoords._internal_coords:
            node["internal_coords"] = globalcoords._internal_coords

        return node
