"""
This module provides errors/exceptions and warnings of general use for NDCube.

Exceptions that are specific to a given subpackage should **not** be here,
but rather in the particular package.
"""
import warnings

__all__ = ["NDCubeWarning", "NDCubeDeprecationWarning", "warn_user", "warn_deprecated"]


class NDCubeWarning(UserWarning):
    """
    A general NDCube warning
    """

class NDCubeDeprecationWarning(FutureWarning, NDCubeWarning):
    """
    A warning class to indicate a deprecated feature.
    """


def warn_user(msg, stacklevel=1):
    """
    Raise a `NDCubeWarning`.

    Parameters
    ----------
    msg : str
        Warning message.
    stacklevel : int
        This is interpreted relative to the call to this function,
        e.g. ``stacklevel=1`` (the default) sets the stack level in the
        code that calls this function.
    """
    warnings.warn(msg, NDCubeWarning , stacklevel + 1)


def warn_deprecated(msg, stacklevel=1):
    """
    Raise a `NDCubeDeprecationWarning`.

    Parameters
    ----------
    msg : str
        Warning message.
    stacklevel : int
        This is interpreted relative to the call to this function,
        e.g. ``stacklevel=1`` (the default) sets the stack level in the
        code that calls this function.
    """
    warnings.warn(msg, NDCubeDeprecationWarning , stacklevel + 1)
