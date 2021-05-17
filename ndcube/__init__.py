"""
ndcube
======

A base package for multi-dimensional contiguous and non-contiguous coordinate-aware arrays.

* Homepage: https://github.com/sunpy/ndcube
* Documentation: https://docs.sunpy.org/projects/ndcube/
"""
import sys

# Enforce Python version check during package import.
# Must be done before any ndcube imports
__minimum_python_version__ = "3.7"


class UnsupportedPythonError(Exception):
    """Running on an unsupported version of Python."""


if sys.version_info < tuple(int(val) for val in __minimum_python_version__.split('.')):
    # This has to be .format to keep backwards compatibly.
    raise UnsupportedPythonError(
        "sunpy does not support Python < {}".format(__minimum_python_version__))


from .extra_coords import ExtraCoords
from .global_coords import GlobalCoords
from .ndcollection import NDCollection
from .ndcube import NDCube
from .ndcube_sequence import NDCubeSequence
from .version import version as __version__

__all__ = ['NDCube', 'NDCubeSequence', "NDCollection", "ExtraCoords", "GlobalCoords"]
