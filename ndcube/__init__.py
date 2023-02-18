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
__minimum_python_version__ = "3.8"


class UnsupportedPythonError(Exception):
    """Running on an unsupported version of Python."""


if sys.version_info < tuple(int(val) for val in __minimum_python_version__.split('.')):
    # This has to be .format to keep backwards compatibly.
    raise UnsupportedPythonError(
        "ndcube does not support Python < {}".format(__minimum_python_version__))

from .extra_coords import ExtraCoordsABC, ExtraCoords  # NOQA
from .global_coords import GlobalCoordsABC, GlobalCoords  # NOQA
from .ndcollection import NDCollection  # NOQA
from .ndcube import NDCube, NDCubeBase  # NOQA
from .ndcube_sequence import NDCubeSequence, NDCubeSequenceBase  # NOQA
from .version import version as __version__  # NOQA

__all__ = ['NDCube', 'NDCubeSequence', "NDCollection", "ExtraCoords", "GlobalCoords"]
