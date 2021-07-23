.. _reproject:

=======================
Reprojecting ND Objects
=======================

Reprojecting allows you to transform your ND Objects to a coordinate grid described by another WCS object.
Using this feature it is possible to regrid ND Objects by providing an appropriate target WCS, for operations such as resampling or alignment.
It also enables putting similar `~ndcube.NDCube` objects onto the same grid for more direct comparison.

.. _cube_reproject:

Reprojecting an NDCube
======================

Reprojecting returns a new `~ndcube.NDCube` object that has been transformed to use the provided ``target_wcs``.
The ``target_wcs`` must be compatible with the WCS that is already associated with your `~ndcube.NDCube`.
This means that it should represent the same physical axes and in the same order.

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

  >>> reprojected_cube = my_cube.reproject_to(target_wcs=target_wcs, shape_out=(4, 4, 10))

In the above example, the ``CDELT1`` parameter of the ``target_wcs`` was modified to 0.1 (from 0.2).
This allowed us to upsample the wavelength axis by a factor of 2.
Accordingly, the shape of the data was updated to ``(4, 4, 10)`` from the initial shape ``(4, 4, 5)``.
The wavelength axis goes last since the aray shape and WCS shape are represented in opposite directions.

Currently, this method does not handle the ``mask``, ``extra_coords``, and ``uncertainty`` attributes.
These values are dropped from the returned `~ndcube.NDCube`.
