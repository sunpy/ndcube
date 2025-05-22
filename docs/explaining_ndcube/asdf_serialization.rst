.. _asdf_serialization

*************************
Saving ND objects to ASDF
*************************

The `Advanced Scientific Data Format (ASDF)<https://www.asdf-format.org/en/latest/>`_ is an extensible format for validating and saving complex scientific data along with its metadata.
`ndcube` provides schemas and converters for all the ND objects (NDCube, NDCubeSequence and NDCollection) as well as for various WCS and table objects required by them.
To make use of these, simply save an ND object to an ASDF file and it will be correctly serialized.

.. code-block:: python
  >>> import numpy as np
  >>> import asdf
  >>> import astropy.wcs
  >>> from ndcube import NDCube
  >>> 
  >>> # Define data array.
  >>> data = np.random.rand(4, 4, 5)
  >>> 
  >>> # Define WCS transformations in an astropy WCS object.
  >>> wcs = astropy.wcs.WCS(naxis=3)
  >>> wcs.wcs.ctype = 'WAVE', 'HPLT-TAN', 'HPLN-TAN'
  >>> wcs.wcs.cunit = 'Angstrom', 'deg', 'deg'
  >>> wcs.wcs.cdelt = 0.2, 0.5, 0.4
  >>> wcs.wcs.crpix = 0, 2, 2
  >>> wcs.wcs.crval = 10, 0.5, 1
  >>> wcs.wcs.cname = 'wavelength', 'HPC lat', 'HPC lon'
  >>> 
  >>> # Now instantiate the NDCube
  >>> my_cube = NDCube(data, wcs=wcs)
  >>> 
  >>> # Save the NDCube to an ASDF file
  >>> with asdf.AsdfFile(tree={"mycube": my_cube}) as f:
  >>>     f.write_to("somefile.asdf")
