from asdf.extension import Converter


class ReorderedConverter(Converter):
    tags = ["tag:sunpy.org:ndcube/reorderedwcs-0.1.0"]
    types = ["ndcube.wcs.wrappers.reordered_wcs.ReorderedLowLevelWCS"]

    def from_yaml_tree(self, node, tag, ctx):
        from ndcube.wcs.wrappers import ReorderedLowLevelWCS

        reorderedwcs = ReorderedLowLevelWCS(wcs=node["wcs"],
                                            pixel_order = node.get("pixel_order"),
                                            world_order = node.get("world_order")
                                            )
        return reorderedwcs
    def to_yaml_tree(self, reorderedwcs, tag, ctx):
        node={}
        node["wcs"] = reorderedwcs._wcs
        node["pixel_order"] = (reorderedwcs._pixel_order)
        node["world_order"] = (reorderedwcs._world_order)
        return node
