

from typing import Any
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from astropy.wcs.wcsapi import BaseLowLevelWCS
from astropy.wcs.wcsapi.wrappers.base import BaseWCSWrapper

__all__ = ['ReorderedLowLevelWCS']


class ReorderedLowLevelWCS(BaseWCSWrapper):  # type: ignore[misc]
    """
    A wrapper for a low-level WCS object that has re-ordered
    pixel and/or world axes.

    Parameters
    ----------
    wcs : `~astropy.wcs.wcsapi.BaseLowLevelWCS`
        The original WCS for which to reorder axes
    pixel_order : iterable
        The indices of the original axes in the order of the
        new WCS.
    world_order : iterable
        The indices of the original axes in the order of the
        new WCS.
    """

    def __init__(self, wcs: BaseLowLevelWCS, pixel_order: Sequence[int], world_order: Sequence[int]) -> None:
        if sorted(pixel_order) != list(range(wcs.pixel_n_dim)):
            raise ValueError(f'pixel_order should be a permutation of {list(range(wcs.pixel_n_dim))}')
        if sorted(world_order) != list(range(wcs.world_n_dim)):
            raise ValueError(f'world_order should be a permutation of {list(range(wcs.world_n_dim))}')
        self._wcs = wcs
        self._pixel_order = pixel_order
        self._world_order = world_order
        self._pixel_order_inv = np.argsort(pixel_order)
        self._world_order_inv = np.argsort(world_order)

    @property
    def world_axis_physical_types(self) -> list[Any]:
        return [self._wcs.world_axis_physical_types[idx] for idx in self._world_order]

    @property
    def world_axis_units(self) -> list[Any]:
        return [self._wcs.world_axis_units[idx] for idx in self._world_order]

    @property
    def pixel_axis_names(self) -> list[Any]:
        return [self._wcs.pixel_axis_names[idx] for idx in self._pixel_order]

    @property
    def world_axis_names(self) -> list[Any]:
        return [self._wcs.world_axis_names[idx] for idx in self._world_order]

    def pixel_to_world_values(self, *pixel_arrays: Any) -> list[Any]:
        ordered_pixel_arrays = [pixel_arrays[idx] for idx in self._pixel_order_inv]
        world_arrays = self._wcs.pixel_to_world_values(*ordered_pixel_arrays)
        return [world_arrays[idx] for idx in self._world_order]

    def world_to_pixel_values(self, *world_arrays: Any) -> list[Any]:
        ordered_world_arrays = [world_arrays[idx] for idx in self._world_order_inv]
        pixel_arrays = self._wcs.world_to_pixel_values(*ordered_world_arrays)
        return [pixel_arrays[idx] for idx in self._pixel_order]

    @property
    def world_axis_object_components(self) -> list[Any]:
        return [self._wcs.world_axis_object_components[idx] for idx in self._world_order]

    @property
    def pixel_shape(self) -> tuple[Any, ...] | None:
        if self._wcs.pixel_shape:
            return tuple([self._wcs.pixel_shape[idx] for idx in self._pixel_order])
        return None

    @property
    def pixel_bounds(self) -> tuple[Any, ...] | None:
        if self._wcs.pixel_bounds:
            return tuple([self._wcs.pixel_bounds[idx] for idx in self._pixel_order])
        return None

    @property
    def axis_correlation_matrix(self) -> npt.NDArray[np.bool_]:
        return np.asarray(self._wcs.axis_correlation_matrix[self._world_order][:, self._pixel_order])
