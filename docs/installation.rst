============
Installation
============

ndcube requires Python 3.5+, SunPy 0.8+, astropy and matplotlib.

Installing the Stable Version
-----------------------------
There are two options for installing the stable version of ndcube. The first is
via the anaconda distribution using the conda-forge channel.

.. code-block:: console

		$ conda install --channel conda-forge ndcube

For more information on installing the anaconda distribution, see the
`anaconda website`_.

To update ndcube do:

.. code-block:: console

		$ conda update ndcube

The second option for installing the stable verison of ndcube is via
pip.

.. code-block:: console

		$ pip install ndcube

Then to update ndcube do:

.. code-block:: console

		$ pip install ndcube --upgrade

.. _dev_install:

Installing the Development Version
----------------------------------

The stable version of ndcube will be relatively reliable. However, if you value
getting the latest updates over reliablility, or want to contribute
to the development of ndcube, you will need to install the bleeding edge version
via `GitHub`_. The recommended way to set up your system is to first fork the
`ndcube GitHub repository`_ (repo) to your GitHub account and then
clone your forked repo to your local machine. This setup has the
advantage that you can push any changes you make in your local version
to your GitHub account.  From there your changes can be merged
into the main repo via a pull request and hence shared with other
users.  See :ref:`contributing_code` to learn more about this process.
You can also set up a remote between your local version and the main
repo so that you can stay updated with the latest changes to
ndcube. Let's step through how to do this.

Once you've forked the main `ndcube GitHub repository`_ to your github account,
create a conda environment on your local machine to hold the ndcube bleeding
edge version. Let's call the conda environment ndcube-dev. Type the
following into a terminal:

.. code-block:: console

		$ conda config --append channels conda-forge
		$ conda create -n ndcube-dev python sunpy hypothesis pytest-mock

Be sure to activate the environment.  In Linux or MacOS, type:

.. code-block:: console

		$ source activate ndcube-dev

In Windows, type:

.. code-block:: console

		> activate ndcube-dev

Next clone the ndcube repo from your github account to a new
directory.  Let's call it ``ndcude-git``.

.. code-block:: console

		$ git clone https://github.com/your-github-name/ndcube.git ndcube-git

To install, change into the new directory and run the install script.

.. code-block:: console

		$ cd ndcube-git
		$ pip install -e .

Finally add a remote to the main repo so you can pull the latest
version.

.. code-block:: console

		$ git remote add upstream https://github.com/sunpy/ndcube.git

Then to ensure you stay up-to-date with the latest version of ndcube,
regularly do:

.. code-block:: console

		$ git pull upstream master

If you wish to make changes to ndcube, it is strongly recommended that
you create a new branch and keep the master branch as a copy of the
latest upstream master branch.  See the :ref:`contributing_code`.

.. _anaconda website: https://docs.anaconda.com/anaconda/install.html
.. _GitHub: https://github.com/
.. _ndcube GitHub repository: https://github.com/sunpy/ndcube
