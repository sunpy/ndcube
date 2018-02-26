============
Installation
============

Stable Version
--------------
There are two options for installing the stable version of ndcube.
The first is via the anaconda distribution using the `conda-forge
channel`:
::
   $ conda install --channel conda-forge ndcube
For more information on installing the anaconda distribution, see the
`anaconda website`_.

To update ndcube do:
::
   $ conda update ndcube

The second option for installing the stable verison of ndcube is via
pip:
::
    $ pip install ndcube
Then to update ndcube do:
::
   $ pip install ndcube --upgrade

Bleeding Edge Version
---------------------

The stable version of ndcube will be relatively reliable.  However, if
you value getting the latest updates immediately over reliablility, or
want to contribute to the development of ndcube, you will need to
install the bleeding edge version via github.  The recommended way to
set up your system is to first fork the `ndcube github repository`_ to
your github account and then clone your forked repo to your local
machine.  This setup has the advantage of being able to push any
changes you make in local version to your github account.  From
there you can issue pull requests to have your changes merged into the
main repo and thus shared with other users.  You can also set up a
remote between your local version and the main repo so that you can
stay updated with the latest changes to ndcube.  Let's step through
how to do this.

Once you've forked the main `ndcube github repository`_ to your github
account, create a conda environment on your local machine to hold the
ndcube bleeding edge version and activate that environment.  Type the
following into a terminal:
::
    $ conda config --append channels conda-forge
    $ conda create -n ndcube-dev python sunpy hypothesis pytest-mock
    $ source activate ndcube-dev

Next clone the ndcube repo from your github account to a new
directory.  Let's call it ``ndcude-git``:
::
    $ git clone https://github.com/your-github-name/ndcube.git ndcube-git

To install, change into the new directory and run the install script:
::
    $ cd ndcube-git
    $ pip install -e .

Finally add a remote to the main repo so you can pull the latest
version:
::
   $ git remote add upstream https://github.com/sunpy/ndcube.git

Then to ensure you stay up-to-date with the latest version of ndcube,
regularly do:
::
   $ git pull upstream master

To push any changes you make to your github account by doing:
::
   $ git push origin branch-name
where ``branch-name`` is the name of the branch you're working on.  Then
from your github account you can request your changes to be merged to
the main repo.  For more information on on git version control,
github, and issuing pull requests, see `SunPy's version control
guide`_.

.. _anaconda website: https://docs.anaconda.com/anaconda/install.html
.. _`ndcube github repository`: https://github.com/sunpy/ndcube
.. _`SunPy's version control guide`: http://docs.sunpy.org/en/stable/dev_guide/version_control.html
