import numpy as np

from asdf.extension import Converter


class NDMetaConverter(Converter):
    tags = ["tag:sunpy.org:ndcube/meta/ndmeta-*"]
    types = ["ndcube.meta.NDMeta"]

    def from_yaml_tree(self, node, tag, ctx):
        from ndcube.meta import NDMeta
        axes = {k: np.array(v) for k, v in node["axes"].items()}
        meta = NDMeta(node["meta"], node["key_comments"], axes, node["data_shape"])
        meta._original_meta = node["original_meta"]
        return meta

    def to_yaml_tree(self, meta, tag, ctx):
        node = {}
        node["meta"] = dict(meta)
        node["key_comments"] = meta.key_comments
        node["axes"] = meta.axes
        node["data_shape"] = meta.data_shape
        node["original_meta"] = meta._original_meta  # not the MappingProxy object
        return node
