from asdf.extension import Converter


class ResampledConverter(Converter):
    tags = ["tag:sunpy.org:ndcube/resampledwcs-*"]
    types = ["ndcube.wcs.wrappers.resampled_wcs.ResampledLowLevelWCS"]

    def from_yaml_tree(self, node, tag, ctx):
        from ndcube.wcs.wrappers import ResampledLowLevelWCS

        return ResampledLowLevelWCS(
            wcs=node["wcs"],
            offset=node.get("offset"),
            factor=node.get("factor"),
        )

    def to_yaml_tree(self, resampledwcs, tag, ctx):
        node = {}
        node["wcs"] = resampledwcs._wcs
        node["factor"] = resampledwcs._factor
        node["offset"] = resampledwcs._offset

        return node
