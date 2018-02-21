# Licensed under a 2-clause BSD style license - see LICENSE.rst

"""
This is a SunPy affiliated package.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._sunpy_init import *
# ----------------------------------------------------------------------------

if not _ASTROPY_SETUP_:
    # For egg_info test builds to pass, put package imports here.
    from .ndcube import NDCube, NDCubeOrdered
    from .ndcube_sequence import NDCubeSequence
