The SunPy project is very happy to announce a new release of "ndcube".

ndcube is a package built for handling, inspecting and visualizing a wide
variety of data, of any number of dimensions, along with coordinate information
described by WCS. The `~ndcube.NDCube` provides functionality for slicing
the data, WCS and other metadata simultaneously, plotting and animating,
and associating extra coordinate information along any axis.

In addition, the ndcube package provides the `~nddata.NDCubeSequence`
class for representing sequences of `~ndcube.NDCube` objects where the
coordinate information may or may not align, and accessing these sequences in a
way consistent with a singular cube.

This release of ndcube contains 95 commits in 12 merged pull requests closing 5 issues from 5 people, 2 of which are first time contributors to ndcube.

The people who have contributed to the code for this release are:

    Daniel Ryan
    David Stansby  *
    Nabil Freij
    P. L. Lim  *
    Stuart Mumford (boooooooo)

Where a * indicates their first contribution to ndcube.

For more information about ndcube see the `documentation <http://docs.sunpy.org/projects/ndcube/>`__.

ndcube can be installed from pip or conda using the following commands::

  $ conda install -c conda-forge ndcube

  $ pip install ndcube
