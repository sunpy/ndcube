from typing import Any

from asdf.extension import Converter


class ReorderedConverter(Converter):  # type: ignore[misc]
    tags = ["tag:sunpy.org:ndcube/reorderedwcs-*"]
    types = ["ndcube.wcs.wrappers.reordered_wcs.ReorderedLowLevelWCS"]

    def from_yaml_tree(self, node: dict[str, Any], tag: str, ctx: Any) -> Any:
        from ndcube.wcs.wrappers import ReorderedLowLevelWCS

        return ReorderedLowLevelWCS(
            wcs=node["wcs"],
            pixel_order=node["pixel_order"],
            world_order=node["world_order"],
        )

    def to_yaml_tree(self, reorderedwcs: Any, tag: str, ctx: Any) -> dict[str, Any]:
        node: dict[str, Any] = {}
        node["wcs"] = reorderedwcs._wcs
        node["pixel_order"] = reorderedwcs._pixel_order
        node["world_order"] = reorderedwcs._world_order
        return node
