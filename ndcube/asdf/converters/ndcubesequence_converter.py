from typing import Any

from asdf.extension import Converter


class NDCubeSequenceConverter(Converter):  # type: ignore[misc]
    tags = ["tag:sunpy.org:ndcube/ndcubesequence-*"]
    types = ["ndcube.ndcube_sequence.NDCubeSequence"]

    def from_yaml_tree(self, node: dict[str, Any], tag: str, ctx: Any) -> Any:
        from ndcube.ndcube_sequence import NDCubeSequence

        return NDCubeSequence(node["data"],
                              meta=node.get("meta"),
                              common_axis=node.get("common_axis"))

    def to_yaml_tree(self, ndcseq: Any, tag: str, ctx: Any) -> dict[str, Any]:
        node: dict[str, Any] = {}
        node["data"] = ndcseq.data
        if ndcseq.meta is not None:
            node["meta"] = ndcseq.meta
        if ndcseq._common_axis is not None:
            node["common_axis"] = ndcseq._common_axis

        return node
