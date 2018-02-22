`NDCube`
========

|Latest Version| |Build Status| |Build status| |codecov| |Research
software impact| |Powered by NumFOCUS| |Powered by Sunpy|

.. |Powered by Sunpy| image:: http://img.shields.io/badge/powered%20by-SunPy-orange.svg?style=flat
   :target: http://www.sunpy.org
.. |Latest Version| image:: https://img.shields.io/pypi/v/ndcube.svg
   :target: https://pypi.python.org/pypi/ndcube/
.. |Build Status| image:: https://secure.travis-ci.org/sunpy/sunpy.svg
   :target: http://travis-ci.org/sunpy/ndcube
.. |Build status| image:: https://ci.appveyor.com/api/projects/status/github/sunpy/ndcube?branch=master&svg=true
   :target: https://ci.appveyor.com/project/sunpy/ndcube
.. |codecov| image:: https://codecov.io/gh/sunpy/ndcube/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/sunpy/ndcube
.. |Powered by NumFOCUS| image:: https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A
   :target: http://numfocus.org

NDcube is a a open-source package for multi-dimensional contiguious and non-contiguious spatially aware arrays.

Installation
------------

If you want to develop NDcube you will need to install from git.
The best way to do this is to create a new conda environment and install the git
version of SunPy in it:

::

    $ conda config --append channels conda-forge
    $ conda create -n ndcube-dev python sunpy hypothesis pytest-mock
    $ source activate ndcube-dev
    $ git clone https://github.com/sunpy/ndcube.git ndcube-git
    $ cd ndcube-git
    $ pip install -e .

For detailed (general) installation instructions, see the `installation guide`_ in
the SunPy docs.

Getting Help
------------

For more information or to ask questions about SunPy, check out:

-  `SunPy Mailing List`_
-  `SunPy Matrix Channel`_

Contributing
------------

If you would like to get involved, start by joining the `SunPy mailing list`_ and check out the `Developer’s Guide`_ section of the SunPy docs.
Stop by our chat room `#sunpy:matrix.org`_ if you have any questions.
Help is always welcome so let us know what you like to work on, or check out the `issues page`_ for the list of known outstanding items.

For more information on contributing to SunPy, please read our `contributing guide`_.

Code of Conduct
---------------

When you are interacting with the SunPy community you are asked to follow our `Code of Conduct`_.

License
-------

This project is Copyright (c) SunPy Developers and licensed under the terms of the BSD 3-Clause license. See the licenses folder for more information.

.. _installation guide: http://docs.sunpy.org/en/stable/guide/installation/index.html
.. _SunPy Matrix Channel: https://riot.im/app/#/room/#sunpy:matrix.org
.. _SunPy mailing list: https://groups.google.com/forum/#!forum/sunpy
.. _Developer’s Guide: http://docs.sunpy.org/en/latest/dev_guide/index.html
.. _`#sunpy:matrix.org`: https://riot.im/app/#/room/#sunpy:matrix.org
.. _issues page: https://github.com/sunpy/ndcube/issues
.. _contributing guide: http://docs.sunpy.org/en/stable/dev_guide/newcomers.html#newcomers
.. _Code of Conduct: http://docs.sunpy.org/en/stable/coc.html
