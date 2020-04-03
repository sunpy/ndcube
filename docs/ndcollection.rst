.. _ndcollection:

============
NDCollection
============

`~ndcube.NDCollection` is a container class for grouping `~ndcube.NDCube` or 
`~ndcube.NDCubeSequence` instances together.
It does not imply an ordered relationship between its constituent ND objects
like `~ndcube.NDCubeSequence`.
Instead it links ND objects in an unordered way like a Python dictionary.
This has many possible uses, for example, linking observations with derived
data products.

Let's say we have a 3D `~ndcube.NDCube` representing space-space-wavelength.
Then let's say we fit a spectral line in each pixel's spectrum and extract
its linewidth.
Now we have a 2D spatial map of linewidth with the same spatial axes
as the original 3D cube.
However the physical properties represented by the data are different.
They do not have an order within their common coordinate space.
And they do not have the same dimensionality as the 2nd cube's spectral axis
has been collapsed.
Therefore is it not appropriate to combine them in an `~ndcube.NDCubeSequence`.
This is where `~ndcube.NDCollection` comes in handy.
It allows us to name each ND object and combine them into a single container,
just like a dictionary.
In fact `~ndcube.NDCollection` inherits from `dict`.

Initialization
--------------
To see how we initialize an `~ndcube.NDCollection`, let's first define a couple
of `~ndcube.NDCube` instances representing the situation above, i.e. a 3D
space-space-spectral cube and a 2D space-space cube that share spatial axes.
Let there be 10x20 spatial pixels and 30 pixels along the spectral axis.

.. code-block:: python
  
  >>> import numpy as np
  >>> from astropy.wcs import WCS
  >>> from ndcube import NDCube

  >>> # Define observations NDCube.
  >>> data = np.ones((10, 20, 30)) # dummy data
  >>> obs_wcs_dict = {
  ...    'CTYPE1': 'WAVE    ', 'CUNIT1': 'Angstrom', 'CDELT1': 0.2, 'CRPIX1': 0, 'CRVAL1': 10, 'NAXIS1': 30,
  ...    'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5, 'NAXIS2': 20,
  ...    'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 10}
  >>> obs_wcs = WCS(obs_wcs_dict)
  >>> obs_cube = NDCube(data, obs_wcs)
  
  >>> # Define derived linewidth NDCube
  >>> linewidth_data = np.ones((10, 20)) / 2 # dummy data
  >>> linewidth_wcs_dict = {
  ...    'CTYPE1': 'HPLT-TAN', 'CUNIT1': 'deg', 'CDELT1': 0.5, 'CRPIX1': 2, 'CRVAL1': 0.5, 'NAXIS1': 20,
  ...    'CTYPE2': 'HPLN-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.4, 'CRPIX2': 2, 'CRVAL2': 1, 'NAXIS2': 10}
  >>> linewidth_wcs = WCS(linewidth_wcs_dict)
  >>> linewidth_cube = NDCube(linewidth_data, linewidth_wcs)

Combine these ND objects into an `~ndcube.NDCollection` by supplying a sequence of
``(key, value)`` pairs in the same way that you initialize and dictionary.

.. code-block:: python

  >>> from ndcube import NDCollection
  >>> my_collection = NDCollection([("observations", obs_cube), ("linewidths", linewidth_cube)])

Data Access
-----------

Key Access
**********
To access each ND object in ``my_collection`` we can index with the name of the desired object,
just like a `dict`:

.. code-block:: python

  >>> my_collection["observations"]  # doctest: +SKIP

And just like a `dict` we can see the different names available using the ``keys`` method:

.. code-block:: python

  >>> my_collection.keys()
  dict_keys(['observations', 'linewidths'])

Aligned Axes & Slicing
**********************

Aligned Axes
^^^^^^^^^^^^

`~ndcube.NDCollection` is more powerful than a simple dictionary because it
allows us to link common aligned axes between the ND objects.
In our example above, the linewidth object's axes are aligned with the
first two axes of observation object.  Let's instantiate our collection again,
but this time declare those axes to be aligned.

.. code-block:: python

  >>> my_collection = NDCollection(
  ...    [("observations", obs_cube), ("linewidths", linewidth_cube)], aligned_axes=(0, 1))

We can see which axes are aligned by inpecting the ``aligned_axes`` attribute:

.. code-block:: python

  >>> my_collection.aligned_axes
  {'observations': (0, 1), 'linewidths': (0, 1)}

As you can see, this gives us the aligned axes for each ND object separately.
We should read this as the 0th axes of both ND objects are aligned, as are the
1st axes of both objects.
Because each ND object's set of aligned axes is stored separately,
aligned axes do not have to be in the same order in both objects.
Let's say we reversed the axes of our ``linewidths`` ND object for some reason:

.. code-block:: python

  >>> linewidth_wcs_dict_reversed = {
  ...    'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5, 'NAXIS2': 20,
  ...    'CTYPE1': 'HPLN-TAN', 'CUNIT1': 'deg', 'CDELT1': 0.4, 'CRPIX1': 2, 'CRVAL1': 1, 'NAXIS1': 10}
  >>> linewidth_wcs_reversed = WCS(linewidth_wcs_dict_reversed)
  >>> linewidth_cube_reversed = NDCube(linewidth_data.transpose(), linewidth_wcs_reversed)

We can still define an `~ndcube.NDCollection` with aligned axes by supplying
a tuple of tuples, giving the aligned axes of each ND object separately.
In this case, the 0th axis of the ``observations`` object is aligned with the 1st
axis of the ``linewidths`` object and vice versa.

.. code-block:: python

   >>> my_collection_reversed = NDCollection(
   ...    [("observations", obs_cube), ("linewidths", linewidth_cube_reversed)],
   ...    aligned_axes=((0, 1), (1, 0)))
   >>> my_collection_reversed.aligned_axes
   {'observations': (0, 1), 'linewidths': (1, 0)}

Aligned axes must have the same lengths.
We can see the lengths of the aligned axes by using the ``aligned_dimensions``
property.

.. code-block:: python

  >>> my_collection.aligned_dimensions
  <Quantity [10., 20.] pix>

Note that this only tells us the lengths of the aligned axes.  To see the
lengths of the non-aligned axes, e.g. the spectral axis of the ``observations``
object, you must inspect that ND object individually.

We can also see the physical properties to which the aligned axes correspond
by using the ``aligned_world_axis_physical_types`` property.

.. code-block:: python

  >>> my_collection.aligned_world_axis_physical_types
  ('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat')

Note that this method simply returns the world physical axis types of one of
the ND objects.  However, there is no requirement that all aligned axes must
represent the same physical types.
They just have to be the same length.

Slicing
^^^^^^^

Defining aligned axes enables us to slice those axes of all the ND objects in
the collection by using the standard Python slicing API.

.. code-block:: python

  >>> sliced_collection = my_collection[1:3, 3:8]
  >>> sliced_collection.keys()
  dict_keys(['observations', 'linewidths'])
  >>> sliced_collection.aligned_dimensions
  <Quantity [2., 5.] pix>

Note that we still have the same number of ND objects, but both have
been sliced using the inputs provided by the user.
Also note that slicing takes account of and updates the aligned axis information.
Therefore a self-consistent result would be obtained even if the aligned axes
are not in order.

.. code-block:: python

  >>> sliced_collection_reversed = my_collection_reversed[1:3, 3:8]
  >>> sliced_collection_reversed.keys()
  dict_keys(['observations', 'linewidths'])
  >>> sliced_collection_reversed.aligned_dimensions
  <Quantity [2., 5.] pix>

Editing NDCollection
--------------------

Because `~ndcube.NDCollection` inherits from `dict`, we can edit the 
collection using many of the same methods.
These have the same or analagous APIs to the ``dict`` versions and 
include ``del``, `~ndcube.NDCollection.pop`, and `~ndcube.NDCollection.update`.
Some `dict` methods may not be implemented on `~ndcube.NDCollection`
if they are not consistent with its design.
