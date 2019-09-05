#!/usr/bin/env python

# Licensed under a 3-clause BSD style license - see LICENSE.rst

import builtins

# Ensure that astropy-helpers is available
import ah_bootstrap  # noqa

from setuptools import setup
from setuptools.config import read_configuration

from astropy_helpers.setup_helpers import register_commands, get_package_info
from astropy_helpers.version_helpers import generate_version_py

# Store the package name in a built-in variable so it's easy
# to get from other parts of the setup infrastructure
builtins._ASTROPY_PACKAGE_NAME_ = read_configuration('setup.cfg')['metadata']['name']

# Create a dictionary with setup command overrides. Note that this gets
# information about the package (name and version) from the setup.cfg file.
cmdclass = register_commands()

# Override the default Astropy Test Command
try:
    from sunpy.tests.setup_command import SunPyTest
    # Overwrite the Astropy Testing framework
    cmdclass['test'] = type('SunPyTest', (SunPyTest,),
                            {'package_name': 'ndcube'})
except Exception:
    # Catch everything, if it doesn't work, we still want the package to install.
    pass

# Programmatically generate some extras combos.
extras = read_configuration("setup.cfg")['options']['extras_require']

# Dev is everything
from itertools import chain
extras['dev'] = list(chain(*extras.values()))

# All is everything but tests and docs
exclude_keys = ("tests", "docs", "dev")
ex_extras = dict(filter(lambda i: i[0] not in exclude_keys, extras.items()))
# Concatenate all the values together for 'all'
extras['all'] = list(chain.from_iterable(ex_extras.values()))

# Get configuration information from all of the various subpackages.
# See the docstring for setup_helpers.update_package_files for more
# details.
package_info = get_package_info()
setup(extras_require=extras, use_scm_version=True, cmdclass=cmdclass, **package_info)
