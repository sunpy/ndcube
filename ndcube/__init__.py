from .extra_coords import ExtraCoords
from .global_coords import GlobalCoords
from .ndcollection import NDCollection
from .ndcube import NDCube
from .ndcube_sequence import NDCubeSequence

try:
    from .version import __version__
except ImportError:
    __version__ = "unknown"

__all__ = ['NDCube', 'NDCubeSequence', "NDCollection", "ExtraCoords", "GlobalCoords"]
