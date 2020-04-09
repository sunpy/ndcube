from .ndcube import NDCube, NDCubeOrdered
from .ndcube_sequence import NDCubeSequence
from .ndcollection import NDCollection

try:
    from .version import __version__
except ImportError:
    __version__ = "unknown"

__all__ = ['NDCube', 'NDCubeSequence', "NDCollection"]
