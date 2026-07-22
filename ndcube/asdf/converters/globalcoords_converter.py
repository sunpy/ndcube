from typing import Any

from asdf.extension import Converter


class GlobalCoordsConverter(Converter):  # type: ignore[misc]
    tags = ["tag:sunpy.org:ndcube/global_coords/globalcoords-*"]
    types = ["ndcube.global_coords.GlobalCoords"]

    def from_yaml_tree(self, node: dict[str, Any], tag: str, ctx: Any) -> Any:
        from ndcube.global_coords import GlobalCoords

        globalcoords = GlobalCoords()
        if "internal_coords" in node:
            globalcoords._internal_coords = node["internal_coords"]  # pyright: ignore[reportPrivateUsage]
        globalcoords._ndcube = node["ndcube"]  # pyright: ignore[reportPrivateUsage]

        return globalcoords

    def to_yaml_tree(self, globalcoords: Any, tag: str, ctx: Any) -> dict[str, Any]:
        node: dict[str, Any] = {}
        node["ndcube"] = globalcoords._ndcube  # pyright: ignore[reportPrivateUsage]
        if globalcoords._internal_coords:  # pyright: ignore[reportPrivateUsage]
            node["internal_coords"] = globalcoords._internal_coords  # pyright: ignore[reportPrivateUsage]

        return node
