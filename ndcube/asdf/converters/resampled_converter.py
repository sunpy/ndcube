from typing import Any

from asdf.extension import Converter


class ResampledConverter(Converter):  # type: ignore[misc]
    tags = ["tag:sunpy.org:ndcube/resampledwcs-*"]
    types = ["ndcube.wcs.wrappers.resampled_wcs.ResampledLowLevelWCS"]

    def from_yaml_tree(self, node: dict[str, Any], tag: str, ctx: Any) -> Any:
        from ndcube.wcs.wrappers import ResampledLowLevelWCS

        return ResampledLowLevelWCS(
            wcs=node["wcs"],
            offset=node["offset"],
            factor=node["factor"],
        )

    def to_yaml_tree(self, resampledwcs: Any, tag: str, ctx: Any) -> dict[str, Any]:
        node: dict[str, Any] = {}
        node["wcs"] = resampledwcs._wcs
        node["factor"] = resampledwcs._factor
        node["offset"] = resampledwcs._offset

        return node
