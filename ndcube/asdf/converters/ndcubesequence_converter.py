from asdf.extension import Converter


class NDCubeSequenceConverter(Converter):
    tags = ["tag:sunpy.org:ndcube/ndcube_sequence-*"]
    types = ["ndcube.ndcube_sequence.NDCubeSequence"]

    def from_yaml_tree(self, node, tag, ctx):
        from ndcube.ndcube_sequence import NDCubeSequence  # noqa: PLC0415

        return NDCubeSequence(node["data"],
                              meta=node.get("meta"),
                              common_axis=node.get("common_axis"))

    def to_yaml_tree(self, ndcseq, tag, ctx):
        node = {}
        node["data"] = ndcseq.data
        if ndcseq.meta is not None:
            node["meta"] = ndcseq.meta
        if ndcseq._common_axis is not None:
            node["common_axis"] = ndcseq._common_axis

        return node
