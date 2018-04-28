============
Installation
============

ndcube requires Python 3.5+, SunPy 0.9+, astropy and matplotlib.

Installing the Stable Version
-----------------------------
There are two options for installing the stable version of ndcube. The first is
via the anaconda distribution using the conda-forge channel.  For more
information on installing the anaconda distribution, see the `anaconda
website`_.

.. code-block:: console

		$ conda install --channel conda-forge ndcube

To update ndcube do:

.. code-block:: console

		$ conda update ndcube

The second option for installing the stable version of ndcube is via
pip.

.. code-block:: console

		$ pip install ndcube

Then to update ndcube do:

.. code-block:: console

		$ pip install ndcube --upgrade

.. _dev_install:

Installing the Development Version
----------------------------------

The stable version of ndcube will be reliable. However, if you value
getting the latest updates over reliablility, or want to contribute
to the development of ndcube, you will need to install the development
version via `GitHub`_. Let's step through how to do this using
anaconda.  For information on installing the anaconda
distribution, see the `anaconda website`_.

First, create a conda environment on your local machine to hold the
ndcube bleeding edge version. Using a new environment allows you to
keep your root environment for stable package releases.  Let's call
the new conda environment ``ndcube-dev``. Type the following into a
terminal:

.. code-block:: console

		$ conda config --append channels conda-forge
		$ conda create -n ndcube-dev sunpy hypothesis pytest-mock pip sphinx coverage ipython jupyter

Be sure to activate the environment, i.e. switch into it.  In Linux or MacOS, type:

.. code-block:: console

		$ source activate ndcube-dev

In Windows, type:

.. code-block:: console

		> activate ndcube-dev

Next clone the ndcube repo from GitHub to a new directory.  Let's call
it ndcude-git.

.. code-block:: console

		$ git clone https://github.com/sunpy/ndcube.git ndcube-git

To install, change into the new directory and run the install script.

.. code-block:: console

		$ cd ndcube-git
		$ pip install -e .

Voila!  The ndcube development version is now installed!  Be sure you
get the latest updates by regularly doing:

.. code-block:: console

		$ git pull origin master

.. _anaconda website: https://docs.anaconda.com/anaconda/install.html
.. _GitHub: https://github.com/
.. _ndcube GitHub repository: https://github.com/sunpy/ndcube
