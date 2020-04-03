# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._sunpy_init import *   # noqa
# ----------------------------------------------------------------------------

# Enforce Python version check during package import.
# This is the same check as the one at the top of setup.py
import sys
from distutils.version import LooseVersion

__minimum_python_version__ = "3.6"

__all__ = ['NDCube', 'NDCubeSequence', "NDCollection"]


class UnsupportedPythonError(Exception):
    pass


if LooseVersion(sys.version) < LooseVersion(__minimum_python_version__):
    raise UnsupportedPythonError("ndcube does not support Python < {}"
                                 .format(__minimum_python_version__))

if not _SUNPY_SETUP_:   # noqa
    # For egg_info test builds to pass, put package imports here.
    from .ndcube import NDCube, NDCubeOrdered
    from .ndcube_sequence import NDCubeSequence
    from .ndcollection import NDCollection
