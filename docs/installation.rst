.. _installation:

*********************
Installing ``ndcube``
*********************

`ndcube` requires Python >=3.9, ``astropy``>=5.0, ``numpy``>=1.21 and ``gwcs``>=0.18.

Installing the release version
------------------------------

There are two options for installing the release version of `ndcube`.

The first is via the miniforge distribution using the conda-forge channel.
(The anaconda distribution can also be used but because ``miniforge`` is more lightweight, we recommend it.)
For more information on installing ``miniforge`` see the `miniforge website`_.

.. code-block:: console

    $ conda install ndcube

To update ndcube do:

.. code-block:: console

    $ conda update ndcube

The second option for installing the stable version of ndcube is via ``pip``.

.. code-block:: console

        $ pip install ndcube

Then to update ndcube do:

.. code-block:: console

        $ pip install ndcube --upgrade

Please see the `sunpy installation guide for more general installation help <https://docs.sunpy.org/en/stable/installation.html>`__.

.. _dev_install:

Installing the development version
----------------------------------

If you want to contribute to the development of `ndcube`, you will need to install the development version via `ndcube GitHub repository`_.
Let's step through how to do this using miniforge.
For information on installing the miniforge distribution, see the `miniforge website`_.

First, create a conda environment on your local machine to hold the ndcube bleeding edge version.
Using a new environment allows you to keep your root environment for stable package releases.
Let's call the new conda environment ``ndcube-dev``.
Type the following into a terminal:

.. code-block:: console

    $ conda create -n ndcube-dev pip

Be sure to activate the environment, i.e. switch into it.

.. code-block:: console

    $ conda activate ndcube-dev

Next clone the ndcube repo from GitHub to a new directory.
Let's call it "ndcude-git".

.. warning::

    If you want to develop ndcube, you should fork the repository and then clone your fork here and not the main ndcube repository.

.. code-block:: console

    $ git clone https://github.com/sunpy/ndcube.git ndcube-git

To install, change into the new directory and run the install script.

.. code-block:: console

        $ cd ndcube-git
        $ pip install -e .[dev]

Voila!
The ndcube development version is now installed!
Be sure you get the latest updates by regularly doing:

.. code-block:: console

    $ git pull origin main

.. _miniforge website: https://github.com/conda-forge/miniforge#download
.. _ndcube GitHub repository: https://github.com/sunpy/ndcube
