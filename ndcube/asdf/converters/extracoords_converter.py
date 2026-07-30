from typing import Any

from asdf.extension import Converter


class ExtraCoordsConverter(Converter):  # type: ignore[misc]
    tags = ["tag:sunpy.org:ndcube/extra_coords/extra_coords/extracoords-*"]
    types = ["ndcube.extra_coords.extra_coords.ExtraCoords"]

    def from_yaml_tree(self, node: dict[str, Any], tag: str, ctx: Any) -> Any:
        from ndcube.extra_coords.extra_coords import ExtraCoords
        extra_coords = ExtraCoords()
        extra_coords._wcs = node.get("wcs")  # pyright: ignore[reportPrivateUsage]
        extra_coords._mapping = node.get("mapping")  # pyright: ignore[reportPrivateUsage]
        extra_coords._lookup_tables = node.get("lookup_tables", [])  # pyright: ignore[reportPrivateUsage]
        extra_coords._dropped_tables = node.get("dropped_tables", [])  # pyright: ignore[reportPrivateUsage]
        extra_coords._ndcube = node.get("ndcube")  # pyright: ignore[reportPrivateUsage]
        return extra_coords

    def to_yaml_tree(self, extracoords: Any, tag: str, ctx: Any) -> dict[str, Any]:
        node: dict[str, Any] = {}
        if extracoords._wcs is not None:  # pyright: ignore[reportPrivateUsage]
            node["wcs"] = extracoords._wcs  # pyright: ignore[reportPrivateUsage]
        if extracoords._mapping is not None:  # pyright: ignore[reportPrivateUsage]
            node["mapping"] = extracoords._mapping  # pyright: ignore[reportPrivateUsage]
        if extracoords._lookup_tables:  # pyright: ignore[reportPrivateUsage]
            node["lookup_tables"] = extracoords._lookup_tables  # pyright: ignore[reportPrivateUsage]
        if extracoords._dropped_tables is not None:  # pyright: ignore[reportPrivateUsage]
            node["dropped_tables"] = extracoords._dropped_tables  # pyright: ignore[reportPrivateUsage]
        node["ndcube"] = extracoords._ndcube  # pyright: ignore[reportPrivateUsage]
        return node
