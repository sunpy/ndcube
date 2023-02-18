#!/usr/bin/env python
from setuptools import setup  # isort:skip
import os
from itertools import chain

try:
    # Recommended for setuptools 61.0.0+
    # (though may disappear in the future)
    from setuptools.config.setupcfg import read_configuration
except ImportError:
    from setuptools.config import read_configuration

################################################################################
# Programmatically generate some extras combos.
################################################################################
extras = read_configuration("setup.cfg")['options']['extras_require']


# All is everything but tests and docs
exclude_keys = ("tests", "docs")
ex_extras = dict(filter(lambda i: i[0] not in exclude_keys, extras.items()))
# Concatenate all the values together for 'all'
extras['all'] = list(chain.from_iterable(ex_extras.values()))
# Dev is everything
extras['dev'] = list(chain(*extras.values()))

setup(
    extras_require=extras,
    use_scm_version={'write_to': os.path.join('ndcube', '_version.py')},
)
