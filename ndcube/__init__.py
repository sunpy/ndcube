from .ndcollection import NDCollection
from .ndcube import NDCube
from .ndcube_sequence import NDCubeSequence
from .extra_coords import ExtraCoords

try:
    from .version import __version__
except ImportError:
    __version__ = "unknown"

__all__ = ['NDCube', 'NDCubeSequence', "NDCollection", "ExtraCoords"]
