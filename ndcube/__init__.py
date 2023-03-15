"""
ndcube
======

A base package for multi-dimensional contiguous and non-contiguous coordinate-aware arrays.

* Homepage: https://github.com/sunpy/ndcube
* Documentation: https://docs.sunpy.org/projects/ndcube/
"""
from .extra_coords.extra_coords import ExtraCoords, ExtraCoordsABC
from .global_coords import GlobalCoords, GlobalCoordsABC
from .ndcollection import NDCollection
from .ndcube import NDCube, NDCubeBase
from .ndcube_sequence import NDCubeSequence, NDCubeSequenceBase
from .version import version as __version__

__all__ = ['NDCube', 'NDCubeSequence', "NDCollection", "ExtraCoords", "GlobalCoords", "ExtraCoordsABC", "GlobalCoordsABC", "NDCubeBase", "NDCubeSequenceBase", "__version__"]
