from asdf.extension import Converter


class CompoundConverter(Converter):
    tags = ["tag:sunpy.org:ndcube/compoundwcs-0.1.0"]
    types = ["ndcube.wcs.wrappers.compound_wcs.CompoundLowLevelWCS"]

    def from_yaml_tree(self, node, tag, ctx):
        from ndcube.wcs.wrappers import CompoundLowLevelWCS

        return(CompoundLowLevelWCS(*node["wcs"], mapping = node.get("mapping"), pixel_atol = node.get("atol")))

    def to_yaml_tree(self, compoundwcs, tag, ctx):
        node={}
        node["wcs"] = compoundwcs._wcs
        node["mapping"] = compoundwcs.mapping.mapping
        node["atol"] = compoundwcs.atol
        return node
