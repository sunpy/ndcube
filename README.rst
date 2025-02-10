******
ndcube
******

|Latest Version| |codecov| |matrix| |Powered by NumFOCUS| |Powered by SunPy|

.. |Latest Version| image:: https://img.shields.io/pypi/v/ndcube.svg
   :target: https://pypi.python.org/pypi/ndcube/
   :alt: It is up to date, we promise
.. |matrix| image:: https://img.shields.io/matrix/ndcube:openastronomy.org.svg?colorB=%23FE7900&label=Chat&logo=matrix&server_fqdn=openastronomy.modular.im
   :target: https://app.element.io/#/room/#ndcube:openastronomy.org
   :alt: join us on #ndcube:openastronom.org on matrix
.. |codecov| image:: https://codecov.io/gh/sunpy/ndcube/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/sunpy/sunpy
   :alt: Best code cov this side of mars
.. |Powered by NumFOCUS| image:: https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A
   :target: https://numfocus.org
   :alt: Go give them money
.. |Powered by SunPy| image:: http://img.shields.io/badge/powered%20by-SunPy-orange.svg?style=flat
   :target: http://www.sunpy.org
   :alt: SunPy
.. image:: https://joss.theoj.org/papers/10.21105/joss.05296/status.svg
   :target: https://doi.org/10.21105/joss.05296
   :alt: JOSS

``ndcube`` is an open-source SunPy affiliated package for manipulating, inspecting and visualizing multi-dimensional contiguous and non-contiguous coordinate-aware data arrays.

It combines data, uncertainties, units, metadata, masking, and coordinate transformations into classes with unified slicing and generic coordinate transformations and plotting/animation capabilities.
It is designed to handle data of any number of dimensions and axis types (e.g. spatial, temporal, spectral, etc.) whose relationship between the array elements and the real world can be described by World Coordinate System (WCS) translations.

Usage of Generative AI
======================

We expect authentic engagement in our community.
Be wary of posting output from Large Language Models or similar generative AI as comments on GitHub or any other platform, as such comments tend to be formulaic and low quality content.
If you use generative AI tools as an aid in developing code or documentation changes, ensure that you fully understand the proposed changes and can explain why they are the correct approach and an improvement to the current state.

Acknowledging ndcube
====================
If you use ``ndcube`` is your work, we kindly ask you to `acknowledge ndcube`_ in your
publications and presentations.

.. _acknowledge ndcube: https://docs.sunpy.org/projects/ndcube/en/stable/acknowledging.html

Installation
============

For detailed installation instructions, see the `installation guide`_ in the ndcube docs.

.. _installation guide: https://docs.sunpy.org/projects/ndcube/en/stable/installation.html

Getting Help
============

For more information or to ask questions about ndcube, check out:

-  `ndcube Documentation`_
-  `ndcube Element Channel`_

.. _ndcube Documentation: https://docs.sunpy.org/projects/ndcube/
.. _ndcube Element Channel: https://app.element.io/#/room/#ndcube:openastronomy.org

Contributing
============

If you would like to get involved, check out the `Newcomers Guide`_ section of the sunpy docs.
This shows how to get setup with a "sunpy" workflow but the same applies for ndcube, you will just need to replace sunpy with ndcube.

Help is always welcome so let us know what you like to work on, or check out the `issues page`_ for the list of known outstanding items.

.. _Newcomers Guide: https://docs.sunpy.org/en/latest/dev_guide/contents/newcomers.html
.. _issues page: https://github.com/sunpy/ndcube/issues

Code of Conduct
===============

When you are interacting with the SunPy community you are asked to follow our `Code of Conduct`_.

.. _Code of Conduct: https://sunpy.org/coc
