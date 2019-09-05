# Licensed under a 3-clause BSD style license - see LICENSE.rst

__all__ = ['__version__']

# this indicates whether or not we are in the package's setup.py
try:
    _SUNPY_SETUP_
except NameError:
    import builtins
    builtins._SUNPY_SETUP_ = False

try:
    from .version import version as __version__
except ImportError:
    __version__ = ''


if not _SUNPY_SETUP_:
    import os
    from sunpy.tests.runner import SunPyTestRunner

    self_test = SunPyTestRunner.make_test_runner_in(os.path.dirname(__file__))
    self_test.__test__ = False
    __all__ += ["self_test"]
