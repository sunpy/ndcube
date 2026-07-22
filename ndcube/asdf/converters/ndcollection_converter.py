from typing import Any

from asdf.extension import Converter


class NDCollectionConverter(Converter):  # type: ignore[misc]
    tags = ["tag:sunpy.org:ndcube/ndcollection-*"]
    types = ["ndcube.ndcollection.NDCollection"]

    def from_yaml_tree(self, node: dict[str, Any], tag: str, ctx: Any) -> Any:
        from ndcube.ndcollection import NDCollection

        aligned_axes_list = list(node.get("aligned_axes", {}).values())
        aligned_axes = tuple(tuple(lst) for lst in aligned_axes_list)
        return NDCollection(node["items"], meta=node.get("meta"), aligned_axes=aligned_axes)

    def to_yaml_tree(self, ndcollection: Any, tag: str, ctx: Any) -> dict[str, Any]:
        node: dict[str, Any] = {}
        node["items"] = dict(ndcollection)
        if ndcollection.meta is not None:
            node["meta"] = ndcollection.meta
        if ndcollection._aligned_axes is not None:
            node["aligned_axes"] = ndcollection._aligned_axes

        return node
