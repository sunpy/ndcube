from asdf.extension import Converter


class ReorderedConverter(Converter):
    tags = ["tag:sunpy.org:ndcube/reorderedwcs-*"]
    types = ["ndcube.wcs.wrappers.reordered_wcs.ReorderedLowLevelWCS"]

    def from_yaml_tree(self, node, tag, ctx):
        from ndcube.wcs.wrappers import ReorderedLowLevelWCS

        return ReorderedLowLevelWCS(
            wcs=node["wcs"],
            pixel_order=node["pixel_order"],
            world_order=node["world_order"],
        )

    def to_yaml_tree(self, reorderedwcs, tag, ctx):
        node = {}
        node["wcs"] = reorderedwcs._wcs
        node["pixel_order"] = reorderedwcs._pixel_order
        node["world_order"] = reorderedwcs._world_order
        return node
