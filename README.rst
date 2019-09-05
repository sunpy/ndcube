ndcube
======

.. image:: http://img.shields.io/badge/powered%20by-SunPy-orange.svg?style=flat
    :target: http://www.sunpy.org
    :alt: Powered by SunPy Badge

ndcube is an open-source SunPy affiliated package for manipulating,
inspecting and visualizing multi-dimensional contiguous and non-contiguous
coordinate-aware data arrays. It combines data, uncertainties, units, metadata,
masking, and coordinate transformations into classes with unified slicing and
generic coordinate transformations and plotting/animation capabilities. It is
designed to handle data of any number of dimensions and axis types (e.g.
spatial, temporal, spectral, etc.) whose relationship between the array elements
and the real world can be described by World Coordinate System (WCS)
translations. See the `ndcube docs`_ for more details.

Installation
------------

ndcube requires Python 3.5+, SunPy 0.8+, astropy and matplotlib.

Stable Version
##############

There are two options for installing the stable version of ndcube. The first is
via the anaconda distribution using the conda-forge channel::

   $ conda install --channel conda-forge ndcube

For more information on installing the anaconda distribution, see the
`anaconda website`_.

To update ndcube do::

   $ conda update ndcube

The second option for installing the stable verison of ndcube is via pip::

    $ pip install ndcube

Then to update ndcube do::

   $ pip install ndcube --upgrade

Development Version
###################

The stable version of ndcube will be relatively reliable. However, if you value
getting the latest updates immediately over reliablility, or want to contribute
to the development of ndcube, you will need to install the bleeding edge version
via github. The recommended way to set up your system is to first fork the
`ndcube github repository`_ to your github account and then clone your forked
repo to your local machine. This setup has the advantage of being able to push
any changes you make in local version to your github account. From there you can
issue pull requests to have your changes merged into the main repo and thus
shared with other users. You can also set up a remote between your local version
and the main repo so that you can stay updated with the latest changes to
ndcube. Let's step through how to do this.

Once you've forked the main `ndcube github repository`_ to your github account,
create a conda environment on your local machine to hold the ndcube bleeding
edge version and activate that environment. Type the following into a terminal::

    $ conda config --append channels conda-forge
    $ conda create -n ndcube-dev python sunpy hypothesis pytest-mock
    $ source activate ndcube-dev

Next clone the ndcube repo from your github account to a new
directory.  Let's call it ``ndcude-git``::

    $ git clone https://github.com/your-github-name/ndcube.git ndcube-git

To install, change into the new directory and run the install script::

    $ cd ndcube-git
    $ pip install -e .

Finally add a remote to the main repo so you can pull the latest
version::

   $ git remote add upstream https://github.com/sunpy/ndcube.git

Then to ensure you stay up-to-date with the latest version of ndcube,
regularly do::

   $ git pull upstream master

To push any changes you make to your github account by doing::

   $ git push origin branch-name

where ``branch-name`` is the name of the branch you're working on.  Then
from your github account you can request your changes to be merged to
the main repo.  For more information on on git version control,
github, and issuing pull requests, see `SunPy's version control guide`_.

Getting Help
------------

As a SunPy-affiliated package, ndcube relies on the SunPy support
infrastructure.  To pose questions to ndcube and SunPy developers and
to get annoucements regarding ndcube and SunPy in general, sign up to
the

- `SunPy Mailing List`_

To get quicker feedback and chat directly to ndcube and SunPy
developers check out the

- `SunPy Matrix Channel`_.

Contributing
------------

If you would like to get involved, start by joining the `SunPy mailing
list`_ and check out the `Developer’s Guide`_ section of the SunPy
docs.  Stop by our chat room `#sunpy:matrix.org`_ if you have any
questions. Help is always welcome so let us know what you like to work
on, or check out the `issues page`_ for the list of known outstanding
items.

For more information on contributing to ncdube or the SunPy
organization, please read the SunPy `contributing guide`_.

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source
projects is a fantastic way to advance one's coding skills. Writing perfect code
isn't the measure of a good developer (that would disqualify all of us!); it's
trying to create something, making mistakes, and learning from those
mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project (and yes - that includes giving feedback about the contribution
process). Some of these contributions may be the most valuable to the project as
a whole, because you're coming to the project with fresh eyes, so you can see
the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by
`Adrienne Lowe <https://github.com/adriennefriend>`_ for a
`PyCon talk <https://www.youtube.com/watch?v=6Uj746j9Heo>`_, and was adapted by
ndcube based on its use in the README file for the
`MetPy project <https://github.com/Unidata/MetPy>`_.


Code of Conduct
---------------

When you are interacting with the SunPy community you are asked to
follow our `Code of Conduct`_.

License
-------

This project is Copyright (c) SunPy Developers and licensed under the
terms of the BSD 2-Clause license. See the licenses folder for more
information.

.. _ndcube docs: http://docs.sunpy.org/projects/ndcube/
.. _installation guide: http://docs.sunpy.org/en/stable/guide/installation/index.html
.. _SunPy Matrix Channel: https://riot.im/app/#/room/#sunpy:matrix.org
.. _`#sunpy:matrix.org`: https://riot.im/app/#/room/#sunpy:matrix.org
.. _SunPy mailing list: https://groups.google.com/forum/#!forum/sunpy
.. _Developer’s Guide: http://docs.sunpy.org/en/latest/dev_guide/index.html
.. _issues page: https://github.com/sunpy/ndcube/issues
.. _contributing guide: http://docs.sunpy.org/en/stable/dev_guide/newcomers.html#newcomers
.. _Code of Conduct: http://docs.sunpy.org/en/stable/coc.html
.. _anaconda website: https://docs.anaconda.com/anaconda/install.html
.. _`ndcube github repository`: https://github.com/sunpy/ndcube
.. _`SunPy's version control guide`: http://docs.sunpy.org/en/stable/dev_guide/version_control.html
