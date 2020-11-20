.. _data_classes:

============
Data Classes
============
ndcube provides its features via its data classes: `~ndcube.NDCube`, `~ndcube.NDCubeSequence` and `~ndcube.NDCollection`.  This section describes the purpose of each and how they are structured and instantiated.

.. _ndcube:

NDCube
======
ndcube's primary data class is `~ndcube.NDCube`.  It's designed for managing a single data array and set of WCS transformations.  `~ndcube.NDCube` provides unified slicing, visualization, coordinate conversion APIs as well as APIs for inspecting the data, coordinate transformations and metadata. `~ndcube.NDCube` does this in a way that is not specific to any number or physical type of axis.  It can therefore be used for any type of data (e.g. images, spectra, timeseries, etc.) so long as those data are represented by an array and a set of WCS transformations. Moreover, `~ndcube.NDCube` is agnostic to the fundamental array type in which the data is stored, as long as it behaves like a numpy array.
Meanwhile, the WCS object can be any class, as long as it adhere's to the AstroPy `wcsapi (APE 14) <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ specification. These features and flexibility make `~ndcube.NDCube` ideal for subclassing when creating tools for specific data types of data.  It enables developers and scientists to focus on developing the tools needed for their specific research while leveraging standarized APIs for non-data-type-specific functionalities (e.g. slicing).

Initialize an NDCube
--------------------
To initialize the most basic `~ndcube.NDCube` object, we need is a `numpy.ndarray`-like array containing the data and an APE-14-compliant WCS object (e.g. `astropy.wcs.WCS`) describing the coordinate transformations to and from array-elements. Let's create a 3-D array of data with shape ``(3, 4, 5)`` where every value is 1::

  >>> import numpy as np
  >>> data = np.ones((3, 4, 5))

Now let's create an `astropy.wcs.WCS` object.  Let the first world axis be wavelength, the second be helioprojective longitude, the third be helioprojective latitude. Remember that due to convention, the order WCS axes is reversed relative to the data array.  This means that the first two array axes will correspond to helioprojective latitude and longitude and the third to wavelength.::

  >>> import astropy.wcs
  >>> wcs_input_dict = {
  ... 'CTYPE1': 'WAVE    ', 'CUNIT1': 'Angstrom', 'CDELT1': 0.2, 'CRPIX1': 0, 'CRVAL1': 10, 'NAXIS1': 5,
  ... 'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5, 'NAXIS2': 4,
  ... 'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 3}
  >>> input_wcs = astropy.wcs.WCS(wcs_input_dict)

Now we can create an `~ndcube.NDCube`.::

  >>> from ndcube import NDCube
  >>> my_cube = NDCube(data, input_wcs)

The data array is stored in ``mycube.data`` while the WCS object is stored in ``my_cube.wcs``.  However, when manipulating/slicing the data it is better to slice the object as a whole.  (See section on :ref:`ndcube_slicing`.)  So the ``.data`` attribute should only be used to access specific raw data values.

Thanks to its inheritance from `astropy.nddata.NDData`, `~ndcube.NDCube` can also hold additional supplementary data including: metadata located at `NDCube.meta`;
an uncertainty array located at `NDCube.uncertainty` (subclass of `astropy.nddata.NDUncertainty`) describing the uncertainty of each data array value;
a data unit (`astropy.units.Unit` or unit `str`);
and a mask (boolean array), located at `NDCube.mask`, marking reliable and unreliable pixels.
Note that in keeping the convention of `numpy.ma.masked_array` ``True`` means that the corresponding data array axis is masked, i.e. it is bad data, while ``False`` signifies good data.::

  >>> from astropy.nddata import StdDevUncertainty
  >>> uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))
  >>> mask = np.zeros_like(my_cube.data, dtype=bool)
  >>> meta = {"Description": "This is example NDCube metadata."}
  >>> my_cube = NDCube(data, input_wcs, uncertainty=uncertainty, mask=mask,
  ...                  meta=meta, unit=u.ct)

Dimensions and Physical Types
..............................

`~ndcube.NDCube` has useful properties for inspecting its data shape and
axis types, `~ndcube.NDCube.dimensions` and `~ndcube.NDCube.array_axis_physical_types`::

  >>> my_cube.dimensions
  <Quantity [3., 4., 5.] pix>
  >>> my_cube.array_axis_physical_types
  [('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
   ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
   ('em.wl',)]

`~ndcube.NDCube.dimensions` returns an `~astropy.units.Quantity` of pixel units giving the length of each dimension in the `~ndcube.NDCube` while `~ndcube.NDCube.array_axis_physical_types` returns tuples of strings denoting the types of physical properties represented by each array axis.  The tuples are arranged in array axis order as they map to array axes.  As more than one physical type can be associated with an array axis, the length of each tuple can be greater than 1.  This is the case for the 1st and 2nd array array axes which are associated with the coupled world axes of helioprojective latitude and longitude. The axis names are in accordance with the International Virtual Observatory Alliance (IVOA) 
`UCD1+ controlled vocabulary <http://www.ivoa.net/documents/REC/UCD/UCDlist-20070402.html>`_.

`~ndcube.NDCube` provides many more helpful features, specifically regarding coordinate transformations and visualization.  See the :ref:plotting and :ref:coordinates sections.


.. _ndcubesequence:

NDCubeSequence
==============


.. _ndcollection:

NDCollection
============
