The SunPy project is very happy to announce the release of a new package "ndcube".

ndcube is a package built for handling, inspecting and visualizing a wide
variety of data, of any number of dimensions, along with coordinate information
described by FITS-WCS. The `~ndcube.NDCube` object is based on the astropy
`~astropy.nddata.NDData` class, and adds functionality for slicing both the data
and the WCS simultaneously, plotting with matplotlib and support for extra
coordinate information along any of the axes not described by WCS.

In addition to this the ndcube package provides the `~nddata.NDCubeSequence`
class for representing sequences of `~ndcube.NDCube` objects where the
coordinate information may or may not align, and accessing these sequences in a
way consistent with a singular cube.

The ndcube development has been lead by Daniel Ryan with contributions from the
following people:

*  Daniel Ryan
*  Ankit Baruah
*  Stuart Mumford
*  Mateo Inchaurrandieta
*  Nabil Freij
*  Drew Leonard
*  Shelbe Timothy

A lot of the design and development of ndcube was done as part of Ankit's Google
Summer of Code project in the summer of 2017.

For more information about ndcube see the `documentation <http://docs.sunpy.org/projects/ndcube/>`_.

ndcube can be installed from pip or conda using the following commands::


  $ conda install -c conda-forge ndcube

  $ pip install ndcube

