from asdf.extension import Converter


class ExtraCoordsConverter(Converter):
    tags = ["tag:sunpy.org:ndcube/extra_coords/extra_coords/extracoords-*"]
    types = ["ndcube.extra_coords.extra_coords.ExtraCoords"]

    def from_yaml_tree(self, node, tag, ctx):
        from ndcube.extra_coords.extra_coords import ExtraCoords
        extra_coords = ExtraCoords()
        extra_coords._wcs = node.get("wcs")
        extra_coords._mapping = node.get("mapping")
        extra_coords._lookup_tables = node.get("lookup_tables", [])
        extra_coords._dropped_tables = node.get("dropped_tables")
        extra_coords._ndcube = node.get("ndcube")
        return extra_coords

    def to_yaml_tree(self, extracoords, tag, ctx):
        node = {}
        if extracoords._wcs is not None:
            node["wcs"] = extracoords._wcs
        if extracoords._mapping is not None:
            node["mapping"] = extracoords._mapping
        if extracoords._lookup_tables:
            node["lookup_tables"] = extracoords._lookup_tables
        if extracoords._dropped_tables is not None:
            node["dropped_tables"] = extracoords._dropped_tables
        node["ndcube"] = extracoords._ndcube
        return node
