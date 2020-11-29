.. _data_classes:

==========
ND Objects
==========
ndcube provides its features via its data objects: `~ndcube.NDCube`, `~ndcube.NDCubeSequence` and `~ndcube.NDCollection`.
This section describes the purpose of each and how they are structured and instantiated.
To learn how to slice, visualize, and perform coordinate transformations with these classes, see the :ref:`slicing`, :ref:`plotting` and :ref:`coordinates` sections.

.. _ndcube:

NDCube
======
ndcube's primary data class is `~ndcube.NDCube`.
It's designed for managing a single data array and set of WCS transformations.
`~ndcube.NDCube` provides unified slicing, visualization, coordinate conversion APIs as well as APIs for inspecting the data, coordinate transformations and metadata.
`~ndcube.NDCube` does this in a way that is not specific to any number or physical type of axis.
It can therefore be used for any type of data (e.g. images, spectra, timeseries, etc.) so long as those data are represented by an array and a set of WCS transformations.
This makes `~ndcube.NDCube` ideal as a base class for classes represent specific types of data, e.g. images.
It enables developers and scientists to focus on developing what's needed for their specific research while leveraging standarized APIs for non-data-type-specific functionalities (e.g. slicing).
Moreover, `~ndcube.NDCube` is agnostic to the fundamental array type in which the data is stored, as long as it behaves like a numpy array.
Meanwhile, the WCS object can be any class, as long as it adhere's to the AstroPy `wcsapi (APE 14) <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ specification.

Thanks to its inheritance from `astropy.nddata.NDData`, `~ndcube.NDCube` can hold optional supplementary data in addition to its data array and primary WCS transformations.
These include:
general metadata (located at ``.meta``);
the unit of the data (an `astropy.units.Unit` or unit `str` located at ``.unit``);
the uncertainty of each data value (subclass of `astropy.nddata.NDUncertainty` located at ``.uncertainty``);
and a mask marking unreliable data values (boolean array located at ``mask``).
Note that in keeping the convention of `numpy.ma.masked_array`, ``True`` means that the corresponding data value is masked, i.e. it is bad data, while ``False`` signifies good data.
`~ndcube.NDCube` also provides classes for representing additional coordinates not included in the primary WCS object.
These are `~ndcube.ExtraCoords` (located at ``.extra_coords``) - for additional coordinates associated with specific data axes - and `~ndcube.GlobalCoords` (located at ``.global_coords``) for scalar coordinates associated with the `~ndcube.NDCube` as a whole.
These are discussed in :ref:`coordinates`.

The figure below tries to make it easier to visualize and `~ndcube.NDCube` instance and the relationships between its components.
Array-based components are in blue (``.data``, ``.uncertainty``, and ``.mask``), metadata components in green (``.meta`` and ``.unit``), and coordinate components in red (``.wcs``, ``.extra_coords``, and ``.global_coords``).
Yellow ovals represent methods for inspecting, visualizing, and analyzing the `~ndcube.NDCube`.

.. image:: images/ndcube_diagram.png
  :width: 800
  :alt: Components of an NDCube


Initialize an NDCube
--------------------
To initialize the most basic `~ndcube.NDCube` object, we need is a `numpy.ndarray`-like array containing the data and an APE-14-compliant WCS object (e.g. `astropy.wcs.WCS`) describing the coordinate transformations to and from array-elements.
Let's create a 3-D array of data with shape ``(3, 4, 5)`` with random values and a WCS object with axes of wavelength, helioprojective longitude, and helioprojective latitude.  Remember that due to convention, the order of WCS axes is reversed relative to the data array.

.. include:: code_block/instantiate_simple_ndcube.rst

The data array is stored in ``mycube.data`` while the WCS object is stored in ``my_cube.wcs``.
The ``.data`` attribute should only be used to access specific raw data values.
When manipulating/slicing the data it is better to slice the `~ndcube.NDCube` instance as a whole so as to ensure that supporting data - e.g. coordinates, uncertainties, mask - remain consistent.
(See :ref:`cube_slicing`.)

To instantiate a more complex `~ndcube.NDCube` with metadata, a data unit, uncertainties and a mask, we can  the following:

.. include:: code_block/instantiate_ndcube.rst

Generating `~ndcube.ExtraCoords` and `~ndcube.GlobalCoords` objects and attaching them to your `~ndcube.NDCube` is demonstrated in the :ref:`extra_coords` and :ref:`global_coords` sections.

Dimensions and Physical Types
-----------------------------

`~ndcube.NDCube` has useful properties for inspecting its data shape and axis types, `~ndcube.NDCube.dimensions` and `~ndcube.NDCube.array_axis_physical_types`.

.. code-block:: python

  >>> my_cube.dimensions
  <Quantity [3., 4., 5.] pix>
  >>> my_cube.array_axis_physical_types
  [('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
   ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
   ('em.wl',)]

`~ndcube.NDCube.dimensions` returns a `~astropy.units.Quantity` of pixel units giving the length of each dimension in the `~ndcube.NDCube`, `~ndcube.NDCube.array_axis_physical_types` returns tuples of strings denoting the types of physical properties represented by each array axis.
The tuples are arranged in array axis order, while the physical types inside each tuple are returned in world order.
As more than one physical type can be associated with an array axis, the length of each tuple can be greater than 1.
This is the case for the 1st and 2nd array array axes which are associated with the coupled world axes of helioprojective latitude and longitude.
The axis names are in accordance with the International Virtual Observatory Alliance (IVOA) `UCD1+ controlled vocabulary <http://www.ivoa.net/documents/REC/UCD/UCDlist-20070402.html>`_.

`~ndcube.NDCube` provides many helpful features, specifically regarding coordinate transformations, slicing and visualization.
See the :ref:`cube_coordinates`, :ref:`cube_slicing` and :ref:`cube_plotting` sections.


.. _ndcubesequence:

NDCubeSequence
==============
`~ndcube.NDCubeSequence` is a class for handling multiple `~ndcube.NDCube` objects as if they were one contiguous data set.
The `~ndcube.NDCube` objects within an `~ndcube.NDCubeSequence` must be have the same shape and physical types associated with each axis.
They must also be arranged in some order.
The direction in which the cubes are ordered is referred to as the "sequence axis".
For example, say we have four images with a shape of 512 x 512 represented by four 2-D `~ndcube.NDCube` objects.
Let's also say they that were taken at different times, but that their WCS transformations only describe their celestial coordinates.
We can place these `~ndcube.NDCube` objects into a `~ndcube.NDCubeSequence` where the sequence axis acts as a 3rd axis representing time.
Thus, the data set has an effective shape of ``(4, 512, 512)``.
This is shown in panel a) in the figure below.
The cubes are represented as blue squares (representing its array-based data) inset with a smaller red square (representing its coordinates and metadata).
The 2-D cubes are stacked in a 3rd dimension labeled "sequence axis".

.. image:: images/ndcubesequence_diagram.png
  :width: 400
  :alt: Schematic of an NDCubeSequence and its two configurations.

However, let's also say that the images represent tiles in a mosaic that, when combined, form a map of the sky much larger than the field of view of the instrument.
Thus the images represent adjacent regions of the sky.
In that case the cubes are not only ordered in time, but also along one of their spatial axes.
Another way of saying this is that the sequence axis is parallel to one of the cubes' axes.
The cube axis that's parallel to the sequence axis is known as the common axis.
Let's say in our example that the common axis is the x-axis of the cubes.
Thus, we can also treat the data set as if it were a single image with a shape of ``(2048, 512)``.
See panel b) of the figure above.

Setting a common axis is optional and if one is not set it simply means can only treat the data in configuration a) in the figure above.
However if a common axis is set, it means the users can treat the data in configuration a) or b).
`~ndcube.NDCubeSequence` has different versions of its methods whose names are prefixed with ``cube_like`` that account for the common axis.
Equivalent non-cube-like methods do not.
This allows users to switch back and forth between configurations a) and b) as their use case demands.
This flexibility makes `~ndcube.NDCubeSequence` a powerful tool when handling complex N-D dimensional data described by different but comparable coordinate transformations.

Initializing an NDCubeSequence
------------------------------
To initialize the most basic `~ndcube.NDCubeSequence` object, all you need is a list of `~ndcube.NDCube` instances.
Let's first define three 3-D NDCubes for slit-spectrograph data as we did in the :ref:`ndcube` section of this tutorial.

.. code-block:: python

  >>> # Define data for cubes
  >>> import numpy as np
  >>> data0 = np.random.rand((3, 4, 5))
  >>> data1 = data0 * 2
  >>> data2 = data1 * 2

  >>> # Define WCS object for all cubes.
  >>> import astropy.wcs
  >>> wcs_input_dict = {
  ... 'CTYPE1': 'WAVE    ', 'CUNIT1': 'Angstrom', 'CDELT1': 0.2, 'CRPIX1': 0, 'CRVAL1': 10, 'NAXIS1': 5,
  ... 'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5, 'NAXIS2': 4,
  ... 'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 3}
  >>> input_wcs = astropy.wcs.WCS(wcs_input_dict)

  >>> from ndcube import NDCube, NDCubeSequence
  >>> my_cube0 = NDCube(data0, input_wcs)
  >>> my_cube1 = NDCube(data1, input_wcs)
  >>> my_cube2 = NDCube(data2, input_wcs)

Creating an `~ndcube.NDCubeSequence` is simply a case of providing the list of `~ndcube.NDCube` objects to the `~ndcube.NDCubeSequence` class.
We also have the option of providing some sequence-level metadata.
This is in addition to anything located in the ``.meta`` objects of the NDCubes.

.. code-block:: python

  >>> my_sequence_metadata = {"Description": "This is some sample NDCubeSequence metadata."}
  >>> my_sequence = NDCubeSequence([my_cube0, my_cube1, my_cube2], meta=my_sequence_metadata)

The `~ndcube.NDCube` instances are stored in ``my_sequence.data`` while the metadata is stored at ``my_sequence.meta``.
If we wanted to define a common cube axis, we must set it during instantiation.
Let's reinstantiate the `~ndcube.NDCubeSequence` with the common axis as the first cube axis.
Additionally, let's also provide some sequence-level metadata.

.. code-block:: python

  >>> my_sequence = NDCubeSequence([my_cube0, my_cube1, my_cube2], common_axis=0)

.. _dimensions:

Dimensions and Physical Types
-----------------------------

Analagous to `ndcube.NDCube.dimensions`, there is also a `ndcube.NDCubeSequence.dimensions` property for easily inspecting the shape of an `~ndcube.NDCubeSequence` instance

.. code-block:: python

  >>> my_sequence.dimensions
  (<Quantity 3. pix>, <Quantity 3. pix>, <Quantity 4. pix>, <Quantity 5. pix>)

Slightly differently to `ndcube.NDCube.dimensions`, `ndcube.NDCubeSequence.dimensions` returns a tuple of `astropy.units.Quantity` instances with pixel units, giving the length of each axis.
To see the dimensionality of the sequence in the cube-like paradigm, i.e. taking into account the common axis, use the `ndcube.NDCubeSequence.cube_like_dimensions` property.

.. code-block:: python

  >>> my_sequence.cube_like_dimensions
  <Quantity [9., 4., 5.] pix>

Equivalent to `ndcube.NDCube.array_axis_physical_types`, `ndcube.NDCubeSequence.array_axis_physical_types` returns a list of tuples of physical axis types.
The same `IVOA UCD1+ controlled words <http://www.ivoa.net/documents/REC/UCD/UCDlist-20070402.html>`_ are used for the cube axes as is used in `ndcube.NDCube.array_axis_physical_types`.
The sequence axis is given the label ``'meta.obs.sequence'`` as it is the IVOA UCD1+ controlled word that best describes it.
To call, simply do:

.. code-block:: python

  >>> my_sequence.array_axis_physical_types
  [('meta.obs.sequence',),
   ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'), ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
   ('em.wl',)]

Once again, we can see the physical types associated with each axis in the cube-like paradigm be calling `ndcube.NDCubeSequence.cube_like_array_axis_physical_types`.

.. code-block:: python

  >>> my_sequence.cube_like_array_axis_physical_types
  [('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
   ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
   ('em.wl',)]

Explode Along Axis
------------------
During analysis of some data - say of a stack of images - it may be necessary to make some different fine-pointing adjustments to each image that isn't accounted for the in the original WCS translations, e.g. due to satellite wobble.
If these changes are not describable with a single WCS object, it may be desirable to break up the N-D sub-cubes of an `~ndcube.NDCubeSequence` into an sequence of sub-cubes with dimension N-1.
This would enable a separate WCS object to be associated with each image and hence allow individual pointing adjustments.

Rather than manually dividing the datacubes up and deriving the corresponding WCS object for each exposure, `~ndcube.NDCubeSequence`
provides a useful method, `~ndcube.NDCubeSequence.explode_along_axis`.
To call it, simply provide the number of the data cube axis along which you wish to break up the sub-cubes.

.. code-block:: python

  >>> exploded_sequence = my_sequence.explode_along_axis(0)

Assuming we are using the same ``my_sequence`` as above, with dimensions of ``(<Quantity 3.0 pix>, <Quantity 3.0 pix>, <Quantity 4.0 pix>, <Quantity 5.0 pix>)``, the ``exploded_sequence`` will be an `~ndcube.NDCubeSequence` of nine 2-D NDCubes each with shape ``(<Quantity 4.0 pix>, <Quantity 5.0 pix>)``.

.. code-block:: python

  >>> # Check old and new shapes of the squence
  >>> my_sequence.dimensions
  (<Quantity 3. pix>, <Quantity 3. pix>, <Quantity 4. pix>, <Quantity 5. pix>)
  >>> exploded_sequence.dimensions
  (<Quantity 9. pix>, <Quantity 4. pix>, <Quantity 5. pix>)

Note that an `~ndcube.NDCubeSequence` can be exploded along any axis.  A common axis need not be defined.

To learn how to slice `~ndcube.NDCubeSequence` instances and manipulate sequence coordinates, the :ref:`sequence_slicing` and :ref:`sequence_coordinates` sections.

.. _ndcollection:

NDCollection
============
`~ndcube.NDCollection` is a container class for grouping `~ndcube.NDCube` or `~ndcube.NDCubeSequence` instances in an unordered way.
`~ndcube.NDCollection` therefore is differs from `~ndcube.NDCubeSequence` in that the objects contained are not considered to be in any order, are not assumed to represent measurements of the same physical property, and they can have different dimensionalities.
However `~ndcube.NDCollection` is more powerful than a simple `dict` because it enables us to identify axes that are aligned between the objects and hence provides some limited slicing functionality.
(See :ref:`collection_slicing` to for more on slicing.)

One possible application of `~ndcube.NDCollection` is linking observations with derived data products.
Let's say we have a 3D `~ndcube.NDCube` representing space-space-wavelength.
Then let's say we fit a spectral line in each pixel's spectrum and extract its linewidth.
Now we have a 2D spatial map of linewidth with the same spatial axes as the original 3D cube.
There is a clear relationship between these two objects and so it makes sense to store them together.
An `~ndcube.NDCubeSequence` is not appropriate here as the physical properties represented by the two objects is different, they do not have an order within their common coordinate space, and they do not have the same dimensionality.
Instead let's use an `~ndcube.NDCollection`.

Let's use ``my_cube`` defined above as our observations cube and define a "linewidth cube".

.. code-block:: python

  >>> # Define derived linewidth NDCube
  >>> linewidth_data = np.random.rand((3, 4)) / 2 # dummy data
  >>> linewidth_wcs_dict = {
  ...    'CTYPE1': 'HPLT-TAN', 'CUNIT1': 'deg', 'CDELT1': 0.5, 'CRPIX1': 2, 'CRVAL1': 0.5, 'NAXIS1': 20,
  ...    'CTYPE2': 'HPLN-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.4, 'CRPIX2': 2, 'CRVAL2': 1, 'NAXIS2': 10}
  >>> linewidth_wcs = astropy.wcs.WCS(linewidth_wcs_dict)
  >>> linewidth_cube = NDCube(linewidth_data, linewidth_wcs)

To combine these ND objects into an `~ndcube.NDCollection`, simply supply a sequence of ``(key, value)`` pairs in the same way that you initialize and dictionary.

.. code-block:: python

  >>> from ndcube import NDCollection
  >>> my_collection = NDCollection([("observations", my_cube), ("linewidths", linewidth_cube)])

To access each ND object in ``my_collection`` index it with the name of the desired object, just like a `dict`:

.. code-block:: python

  >>> my_collection["observations"]  # doctest: +SKIP

And just like a `dict` we can see the different names available using the ``keys`` method:

.. code-block:: python

  >>> my_collection.keys()
  dict_keys(['observations', 'linewidths'])

Aligned Axes
------------
`~ndcube.NDCollection` is more powerful than a simple dictionary because it allows us to link common aligned axes between the ND objects.
In our example above, the linewidth object's axes are aligned with the first two axes of observation object.
Let's instantiate our collection again, but this time declare those axes to be aligned.
Note that aligned axes must have the same lengths.

.. code-block:: python

  >>> my_collection = NDCollection(
  ...    [("observations", my_cube), ("linewidths", linewidth_cube)], aligned_axes=(0, 1))

We can see which axes are aligned by inpecting the ``aligned_axes`` attribute:

.. code-block:: python

  >>> my_collection.aligned_axes
  {'observations': (0, 1), 'linewidths': (0, 1)}

As you can see, this gives us the axes for each ND object separately.
We should read this as the 1st axis in the ``observations`` tuple is aligned with the first axis in the ``'linewidths'`` tuple, as so on.
Therefore in this case, the axis 0 of both ND objects are aligned, as are axis 1 in both objects.
However, the mapping can be more complicated.
Let's say we reversed the axes of our ``linewidths`` ND object for some reason:

.. code-block:: python

  >>> linewidth_wcs_dict_reversed = {
  ...    'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5, 'NAXIS2': 20,
  ...    'CTYPE1': 'HPLN-TAN', 'CUNIT1': 'deg', 'CDELT1': 0.4, 'CRPIX1': 2, 'CRVAL1': 1, 'NAXIS1': 10}
  >>> linewidth_wcs_reversed = astropy.wcs.WCS(linewidth_wcs_dict_reversed)
  >>> linewidth_cube_reversed = NDCube(linewidth_data.transpose(), linewidth_wcs_reversed)

We can still define an `~ndcube.NDCollection` with aligned axes by supplying a tuple of tuples, giving the aligned axes of each ND object separately.
In this case, the 1st axis of the ``observations`` cube is aligned with the 2nd axis of the ``linewidths`` cube and vice versa.

.. code-block:: python

   >>> my_collection_reversed = NDCollection(
   ...    [("observations", my_cube), ("linewidths", linewidth_cube_reversed)],
   ...    aligned_axes=((0, 1), (1, 0)))
   >>> my_collection_reversed.aligned_axes
   {'observations': (0, 1), 'linewidths': (1, 0)}

Because aligned axes must have the same lengths, we can get the lengths of the aligned axes by using the ``aligned_dimensions`` property.

.. code-block:: python

  >>> my_collection.aligned_dimensions
  <Quantity [3., 4.] pix>

Note that this only tells us the lengths of the aligned axes.
To see the lengths of the non-aligned axes, e.g. the spectral axis of the ``observations`` object, you must inspect that ND object individually.

We can also see the physical properties to which the aligned axes correspond by using the `~ndcube.NDCollection.aligned_axis_physical_types` property.

.. code-block:: python

  >>> my_collection.aligned_axis_physical_types  # doctest: +SKIP
  [('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat'), ('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat')]

This returns the a `list` of `tuple` giving the physical types that correspond to each aligned axis.
For each aligned axis, only physical types are associated with all the cubes in the collection are returned.
Note that there is no there is no requirement that all aligned axes must represent the same physical types.
They just have to be the same length.
Therefore, is it possible that this property returns no physical types.

The real power behind `~ndcube.NDCollection.aligned_axes` is that it enables all objects within the `~ndcube.NDCollection` to be sliced along the aligned axes simultaneously from the `~ndcube.NDCollection` level.
This allows users to quickly and accurately crop their entire data set to a region of interest, thereby speeding up their analysis workflow.
See the :ref:`collection_slicing` to see this in action.

Editing NDCollections
---------------------

Because `~ndcube.NDCollection` inherits from `dict`, we can edit the collection using many of the same methods.
These have the same or analagous APIs to the `dict` versions and include `del`, `~ndcube.NDCollection.pop`, and `~ndcube.NDCollection.update`.
Some `dict` methods may not be implemented on `~ndcube.NDCollection` if they are not consistent with its design.
