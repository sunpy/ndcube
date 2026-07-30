from typing import Any

from asdf.extension import Converter


class CompoundConverter(Converter):  # type: ignore[misc]
    tags = ["tag:sunpy.org:ndcube/compoundwcs-*"]
    types = ["ndcube.wcs.wrappers.compound_wcs.CompoundLowLevelWCS"]

    def from_yaml_tree(self, node: dict[str, Any], tag: str, ctx: Any) -> Any:
        from ndcube.wcs.wrappers import CompoundLowLevelWCS

        return CompoundLowLevelWCS(*node["wcs"], mapping=node.get("mapping"), pixel_atol=node.get("atol"))  # type: ignore[arg-type]

    def to_yaml_tree(self, compoundwcs: Any, tag: str, ctx: Any) -> dict[str, Any]:
        node: dict[str, Any] = {}
        node["wcs"] = compoundwcs._wcs
        node["mapping"] = compoundwcs.mapping.mapping
        node["atol"] = compoundwcs.atol
        return node
