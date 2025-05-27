from asdf.extension import Converter


class NDCollectionConverter(Converter):
    tags = ["tag:sunpy.org:ndcube/ndcube/ndcollection-*"]
    types = ["ndcube.ndcollection.NDCollection"]

    def from_yaml_tree(self, node, tag, ctx):
        from ndcube.ndcollection import NDCollection

        aligned_axes = list(node.get("aligned_axes").values())
        aligned_axes = tuple(tuple(lst) for lst in aligned_axes)
        return NDCollection(node["items"], meta=node.get("meta"), aligned_axes=aligned_axes)

    def to_yaml_tree(self, ndcollection, tag, ctx):
        node = {}
        node["items"] = dict(ndcollection)
        if ndcollection.meta is not None:
            node["meta"] = ndcollection.meta
        if ndcollection._aligned_axes is not None:
            node["aligned_axes"] = ndcollection._aligned_axes

        return node
