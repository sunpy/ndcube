.. _reproject:

=======================
Reprojecting ND Objects
=======================

Reprojecting allows you to transform your ND Objects to use coordinates described by another WCS object.
Users can use this feature to resample the resolution of ND Objects by providing an appropriate target WCS. It also enables them to remove tiny differences between similar `~ndcube.NDCube` objects to get them onto the same grid.

.. _cube_reproject:

Reprojecting an NDCube
======================

Reprojecting returns a new `~ndcube.NDCube` object that has been transformed to use the provided ``target_wcs``.
The ``target_wcs`` must be compatible with the WCS that is already associated with your `~ndcube.NDCube`. This means that it should represent the same physical axes and in the same order.

To reproject an `~ndcube.NDCube`, simply do:

.. expanding-code-block:: python
  :summary: Expand to see <span class="pre">my_cube</span> instantiated.

  >>> import astropy.wcs
  >>> import numpy as np
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

  >>> # Instantiate NDCube with supporting data.
  >>> my_cube = NDCube(data, wcs=wcs)

.. code-block:: python

  >>> # Create a target WCS object with new transformations.
  >>> target_wcs = astropy.wcs.WCS(naxis=3)
  >>> target_wcs.wcs.ctype = 'WAVE', 'HPLT-TAN', 'HPLN-TAN'
  >>> target_wcs.wcs.cunit = 'Angstrom', 'deg', 'deg'
  >>> target_wcs.wcs.cdelt = 0.1, 0.5, 0.4
  >>> target_wcs.wcs.crpix = 0, 2, 2
  >>> target_wcs.wcs.crval = 10, 0.5, 1

  >>> reprojected_cube = my_cube.reproject(target_wcs=target_wcs, shape_out=(4, 4, 10))

In the above example, the ``CDELT1`` parameter of the ``target_wcs`` was modified to 0.1 (from 0.2). This allowed us to upsample the wavelength axis by a factor of 2.
Accordingly, the shape of the data was updated to ``(4, 4, 10)`` from the initial shape ``(4, 4, 5)``. The wavelength axis goes last since the pixel shape and WCS shape are represented in opposite directions.

Currently, this method does not handle the ``mask``, ``extra_coords``, and ``uncertainty`` attributes. These values are dropped from the returned `~ndcube.NDCube`.

.. _cube_sequence_reproject:

Reprojecting an NDCubeSequence
==============================

`~ndcube.NDCubeSequence` offers an interesting application of this functionality. The `~ndcube.NDCubeSequence.combine_cubes` method reprojects all the cubes in an `~ndcube.NDCubeSequence` to a common WCS object, and then combines them all to return a single `~ndcube.NDCube`.
It does this by extracting the sequence axis of the `~ndcube.NDCubeSequence` and adding it to the WCS, hence creating an (N+1)-dimensional `~ndcube.NDCube`. This extra dimension can be used just like any other existing axes.

This method can be used as:

.. expanding-code-block:: python
  :summary: Expand to see <span class="pre">my_cube</span> instantiated.

  >>> import astropy.wcs
  >>> import numpy as np
  >>> from ndcube import NDCubeSequence

  >>> # Define data arrays.
  >>> data0 = np.random.rand(4, 4, 5)
  >>> data1 = np.random.rand(4, 4, 5)
  >>> data2 = np.random.rand(4, 4, 5)

  >>> # Define WCS transformations in an astropy WCS object.
  >>> wcs0 = astropy.wcs.WCS(naxis=3)
  >>> wcs0.wcs.ctype = 'WAVE', 'HPLT-TAN', 'HPLN-TAN'
  >>> wcs0.wcs.cunit = 'Angstrom', 'deg', 'deg'
  >>> wcs0.wcs.cdelt = 0.2, 0.5, 0.4
  >>> wcs0.wcs.crpix = 0, 2, 2
  >>> wcs0.wcs.crval = 10, 0.5, 1

  >>> wcs1 = astropy.wcs.WCS(naxis=3)
  >>> wcs1.wcs.ctype = 'WAVE', 'HPLT-TAN', 'HPLN-TAN'
  >>> wcs1.wcs.cunit = 'Angstrom', 'deg', 'deg'
  >>> wcs1.wcs.cdelt = 0.2, 0.5, 0.4
  >>> wcs1.wcs.crpix = 0, 2, 2
  >>> wcs1.wcs.crval = 10, 0.5, 1

  >>> wcs2 = astropy.wcs.WCS(naxis=3)
  >>> wcs2.wcs.ctype = 'WAVE', 'HPLT-TAN', 'HPLN-TAN'
  >>> wcs2.wcs.cunit = 'Angstrom', 'deg', 'deg'
  >>> wcs2.wcs.cdelt = 0.2, 0.5, 0.4
  >>> wcs2.wcs.crpix = 0, 2, 2
  >>> wcs2.wcs.crval = 10, 0.5, 1

  >>> # Instantiate NDCubes with supporting data.
  >>> cube0 = NDCube(data0, wcs=wcs0)
  >>> cube1 = NDCube(data1, wcs=wcs1)
  >>> cube2 = NDCube(data2, wcs=wcs2)

  >>> # Instantiate NDCubeSequence
  >>> my_sequence = NDCubeSequence([cube0, cube1, cube2])

.. code-block:: python

  >>> combined_cube = my_sequence.combine_cubes(common_wcs_index=2)

By default, this method uses the first `~ndcube.NDCube` in the `~ndcube.NDCubeSequence` for the common WCS object, but it can be controlled using the ``common_wcs_index`` parameter as shown above.

