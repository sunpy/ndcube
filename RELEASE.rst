The SunPy project is very happy to announce a new release of "ndcube".

ndcube is a package built for handling, inspecting and visualizing a wide
variety of data, of any number of dimensions, along with coordinate information
described by FITS-WCS. The `~ndcube.NDCube` object is based on the astropy
`~astropy.nddata.NDData` class, and adds functionality for slicing both the data
and the WCS simultaneously, plotting with matplotlib and support for extra
coordinate information along any of the axes not described by WCS.

In addition the ndcube package provides the `~nddata.NDCubeSequence`
class for representing sequences of `~ndcube.NDCube` objects where the
coordinate information may or may not align, and accessing these sequences in a
way consistent with a singular cube.

This release of ndcube contains 126 commits in 28 merged pull requests closing
52 issues from 5 people, 2 of which are first time contributors to ndcube.

The people who have contributed to the code for this release are:

    Daniel Ryan
    Gabe Shafiq  *
    Nabil Freij
    Stuart Mumford
    Yash Sharma  *

Where a * indicates their first contribution to ndcube.

For more information about ndcube see the `documentation <http://docs.sunpy.org/projects/ndcube/>`_.

ndcube can be installed from pip or conda using the following commands::


  $ conda install -c conda-forge ndcube

  $ pip install ndcube

