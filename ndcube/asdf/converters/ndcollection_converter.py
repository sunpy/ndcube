from asdf.extension import Converter


class NDCollectionConverter(Converter):
    tags = ["tag:sunpy.org:ndcube/ndcube/ndcollection-*"]
    types = ["ndcube.ndcollection.NDCollection"]

    def from_yaml_tree(self, node, tag, ctx):
        from ndcube.ndcollection import NDCollection

        key_value_pairs = list(zip(node["keys"], node["value"]))
        aligned_axes = list(node.get("aligned_axes").values())
        aligned_axes = tuple(tuple(lst) for lst in aligned_axes)
        ndcollection = NDCollection(key_value_pairs,
                                    meta=node.get("meta"),
                                    aligned_axes = aligned_axes)
        return ndcollection

    def to_yaml_tree(self, ndcollection, tag, ctx):
        node = {}
        node["keys"] = tuple(ndcollection.keys())
        node["value"] = tuple(ndcollection.values())
        if ndcollection.meta is not None:
            node["meta"] = ndcollection.meta
        if ndcollection._aligned_axes is not None:
            node["aligned_axes"] = ndcollection._aligned_axes

        return node
