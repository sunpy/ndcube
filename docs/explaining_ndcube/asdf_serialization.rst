.. _asdf_serialization:

*************************
Saving ND objects to ASDF
*************************

:ref:`asdf` is an extensible format for validating and saving complex scientific data along with its metadata.
`ndcube` provides schemas and converters for all the ND objects (`~ndcube.NDCube`, `~ndcube.NDCubeSequence` and `~ndcube.NDCollection`) as well as for various WCS and table objects required by them.
To make use of these, simply save an ND object to an ASDF file and it will be correctly serialized.
ASDF files save a "tree" which is a `dict`.
You can save any number of cubes in your ASDF by adding them to the dictionary.

.. expanding-code-block:: python
  :summary: Click to reveal/hide instantiation of the NDCube.

  >>> import numpy as np
  >>> import asdf
  >>> import astropy.wcs
  >>> from ndcube import NDCube

  >>> # Define data array.
  >>> data = np.random.rand(4, 4, 5)

  >>> # Define WCS transformations in an astropy WCS object.
  >>> wcs = astropy.wcs.WCS(naxis=3)
  >>> wcs.wcs.ctype = 'WAVE', 'HPLT-TAN', 'HPLN-TAN'
  >>> wcs.wcs.cunit = 'Angstrom', 'deg', 'deg'
  >>> wcs.wcs.cdelt = 0.2, 0.5, 0.4
  >>> wcs.wcs.crpix = 0, 2, 2
  >>> wcs.wcs.crval = 10, 0.5, 1
  >>> wcs.wcs.cname = 'wavelength', 'HPC lat', 'HPC lon'

  >>> # Now instantiate the NDCube
  >>> my_cube = NDCube(data, wcs=wcs)


.. code-block:: python

  >>> my_tree = {"mycube": my_cube}
  >>> with asdf.AsdfFile(tree=my_tree) as f:  # doctest:  +SKIP
  ...     f.write_to("somefile.asdf")  # doctest:  +SKIP


What's Supported and What Isn't
===============================

We aim to support all features of `ndcube` when saving and loading to ASDF.
However, because it is possible to create `ndcube` objects with many different components (for example dask arrays) which aren't part of the `ndcube` package these may not be supported.
Many common components of `ndcube` classes are supported in the `asdf_astropy <https://asdf-astropy.readthedocs.io/en/stable/>`__ package, such as `astropy.wcs.WCS`, `astropy.wcs.wcsapi.SlicedLowLevelWCS` and uncertainty objects.

The only component of the `ndcube.NDCube` class which is never saved is the ``.psf`` attribute.

`ndcube` implements converters and schemas for the following objects:

* `~ndcube.NDCube`
* `~ndcube.NDCubeSequence`
* `~ndcube.NDCollection`
* `~ndcube.NDMeta`
* `~ndcube.GlobalCoords`
* `~ndcube.ExtraCoords`
* `~ndcube.extra_coords.TimeTableCoordinate`
* `~ndcube.extra_coords.QuantityTableCoordinate`
* `~ndcube.extra_coords.SkyCoordTableCoordinate`
* `~ndcube.extra_coords.MultipleTableCoordinate`
* `~ndcube.wcs.wrappers.ReorderedLowLevelWCS`
* `~ndcube.wcs.wrappers.ResampledLowLevelWCS`
* `~ndcube.wcs.wrappers.CompoundLowLevelWCS`
