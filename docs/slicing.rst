.. _slicing:

======================
Slicing ndcube Objects
======================
Arguably the most useful feature ndcube provides is the slicing of its data classes.  Users can apply the standard slicing notation to ND objects (or real world coordinates in the case of `ndcube.NDCube.crop`) which alter the data and WCS transformations consistently and simultaneously.  This enables users to rapidly and reliably identify and extract regions of interest in their data, thereby allowing them to move closer to the speed of thought during their analysis.

.. _cube_slicing::

Slicing NDCubes
===============
The `~ndcube.NDCube` slicing infrastructure accurately and consistently returns a new `~ndcube.NDCube` with all relevant infomation altered accordingly.  This includes the data, WCS transformations, `~ndcube.ExtraCoords`, uncertainty and mask.  This is achieved by applying the standard slicing API to the `~ndcube.NDCube`.  Because many of the inspection properties such `~ndcube.NDCube.dimensions` and `~ndcube.NDCube.array_axis_physical_types` are calculated on the fly, the information they return after slicing is also consistent with the sliced `~ndcube.NDCube`.  This makes `~ndcube.NDCube`'s slicing infrastructure very powerful.

To demonstrate `~ndcube.NDCube`'s slicing in action, let's first create an `~ndcube.NDCube` instance.

.. code-block:: python

  >>> import astropy.wcs
  >>> import numpy as np
  >>> from ndcube import NDCube
  >>> data = np.ones((3, 4, 5))
  >>> wcs_input_dict = {
  ... 'CTYPE1': 'WAVE    ', 'CUNIT1': 'Angstrom', 'CDELT1': 0.2, 'CRPIX1': 0, 'CRVAL1': 10, 'NAXIS1': 5,
  ... 'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5, 'NAXIS2': 4,
  ... 'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 3}
  >>> input_wcs = astropy.wcs.WCS(wcs_input_dict)
  >>> my_cube = NDCube(data, input_wcs)

To slice ``my_cube``, simply do something like:

.. code-block:: python

  >>> my_cube_roi = my_cube[0:2, 1:4, 1:4]

Slicing can also reduce the dimensionality of an `~ndcube.NDCube`, e.g.:

.. code-block:: python

  >>> my_2d_cube = my_cube[1:2, 1:3, 0]

This will create a 2-D `~ndcube.NDCube`.  In this case, we have sliced away the wavelength axis which means that wavelength is no longer part of the WCS transformations.  However, the value of wavelength at the location along the axis at which the `~ndcube.NDCube` was sliced can now be accessed via `~ndcube.NDCube.global_coords`.

.. code-block:: python

  >>> my_2d_cube.global_coords['em.wl']  # doctest: +SKIP

.. _ndcube_crop::

Cropping with Real World Coordinates
------------------------------------
In addition to slicing by index, `~ndcube.NDCube` supports slicing by real world coordinates via the `~ndcube.NDCube.crop` method.  This takes two iterables of high level astropy objects -- e.g. `~astropy.time.Time`, `~astropy.coordinates.SkyCoord`, `~astropy.coordinates.SpectralCoord`, `~astropy,units.Quantity` etc. -- which relate to the physical types of the axes in the cube.  Each iterable describes a single location in the data array in real world coordinates.  The first iterable describes the lower corner of the region of interest and thus contains the lower limit of each real world coordinate.  The second iterable represents the upper corner of the region of interest and thus contains the upper limit of each real world coordinate.  The crop method indentifies the smallest rectangular region in the data array that contains both the lower and upper limits in all the real world coordinates, and crops the `~ndcube.NDCube` to that region. It does not rebin or interpolate the data.  The order of the high level coordinate objects in each iterable must be the same as that expected by `astropy.wcs.WCS.world_to_array_index`, namely in world order.::

  >>> import astropy.units as u
  >>> from astropy.coordinates import SkyCoord, SpectralCoord
  >>> from sunpy.coordinates.frames import Helioprojective
  >>> wave_range = SpectralCoord([1.04e-9, 1.08e-9], unit=u.m)
  >>> sky_range = SkyCoord(Tx=[1, 1.5], Ty=[0.5, 1.5], unit=u.deg, frame=Helioprojective)
  >>> lower_corner = [wave_range[0], sky_range[0]]
  >>> upper_corner = [wave_range[-1], sky_range[-1]]
  >>> my_cube_roi = my_cube.crop(lower_corner, upper_corner)

.. _slicing_sequence::

Slicing NDCubeSequences
=======================
As with `~ndcube.NDCube`, `~ndcube.NDCubeSequence` is sliced by applying the standard slicing API.  When an `~ndcube.NDCubeSequence`, it determines which cubes should be kept from the slice input for the sequence axis, then passes the rest of the slicing off to desired NDCubes.  Thus the data arrays, WCS transformations, masks, uncertainty arrays, and extra coordinates are all altered accorindingly in each relevant sub-cube.  Say we have three NDCubes in an `~ndcube.NDCubeSequence`, each of shape ``(3, 4, 5)``.

.. code-block:: python

  >>> # Define data for cubes
  >>> import numpy as np
  >>> data0 = np.ones((3, 4, 5))
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
  >>> my_sequence = NDCubeSequence([my_cube0, my_cube1, my_cube2])

Now say we want to obtain a region of interest from the 2nd and 3rd cubes in the sequence.
Let's say the region of interest in each cube is defined as the 1st slice
along the 1st cube dimension, between the 2nd and 3rd pixels (inclusive)
in the 2nd dimension, and between the 2nd and 4th pixels (inclusive)
in the 3rd dimension. This would be a cumbersome slicing operation
if treating the sub-cubes independently. (This would be made even worse
without the power of `~ndcube.NDCube` where the data arrays, WCS
objects, masks, uncertainty arrays, etc. would all have to be sliced
independently!) However, with `~ndcube.NDCubeSequence` this becomes as
simple as indexing a single array::

  >>> regions_of_interest_in_sequence = my_sequence[1:3, 0, 1:3, 1:4]
  >>> regions_of_interest_in_sequence.dimensions
  (<Quantity 2. pix>, <Quantity 2. pix>, <Quantity 3. pix>)
  >>> regions_of_interest_in_sequence.array_axis_physical_types
  [('meta.obs.sequence',),
   ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
   ('em.wl',)]

This will return a new `~ndcube.NDCubeSequence` with 2 2-D NDCubes,
one for each region of interest from each original sub-cube.
If we want our region of interest to only apply to a single sub-cube,
and we index the sequence axis with an `int`, an `~ndcube.NDCube` is returned::

  >>> roi_from_single_subcube = my_sequence[1, 0, 1:3, 1:4]
  >>> roi_from_single_subcube.dimensions
  <Quantity [2., 3.] pix>
  >>> roi_from_single_subcube.array_axis_physical_types
  [('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
   ('em.wl',)]

However, as with numpy slicing, we can induce the slicing operation to return
an `~ndcube.NDCubeSequence` by supplying a length-1 `slice` to the sequence
axis, rather than an `int`. This sequence will still represent the same region
of interest from the same single sub-cube, but the sequence axis will have a
length of 1, rather than be removed.::

  >>> roi_length1_sequence = my_sequence[0:1, 0, 1:3, 1:4]
  >>> roi_length1_sequence.dimensions
  (<Quantity 1. pix>, <Quantity 2. pix>, <Quantity 3. pix>)
  >>> roi_length1_sequence.array_axis_physical_types
  [('meta.obs.sequence',),
   ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
   ('em.wl',)]

Cube-like Slicing
-----------------
As explained in the :ref:`ndcubesequence` section, we can think of the cubes in an `~ndcube.NDCubeSequence` as being concatenated along one of the cubes' axes if we set a common axis.  Therefore it would be useful to be able to slice the sequence as though it were one
large concatenated cube.  This can be achieved with the `ndcube.NDCubeSequence.index_as_cube` property.  Note that if a common axis is set, we do not have to slice this way.  Instead, we simply have the option of using regular slicing or `ndcube.NDCubeSequence.index_as_cube`.  Let's re-instantiate our `~ndcube.NDCubeSequence` with a common axis of ``0``.

.. code-block:: python

  >>> my_sequence = NDCubeSequence([my_cube0, my_cube1, my_cube2], common_axis=0)

Recall that, ``my_sequence`` has a shape of ``(<Quantity 3. pix>, <Quantity 3. pix>, <Quantity 4. pix>, <Quantity 5. pix>)``.  Therefore is has ``cube-like`` dimensions of ``(<Quantity 9. pix>, <Quantity 4. pix>, <Quantity 5. pix>)`` where the first sub-cube extends along the 0th cube-like axis from 0 to 3, the second from 3 to 6 and the third from 6 to 9.

.. code-block:: python

  >>> my_sequence.cube_like_dimensions
  <Quantity [9., 4., 5.] pix>

Now say we want to extract the same region of interest as above, i.e. ``my_sequence[1, 0, 1:3, 1:4]``.  This can be achieved by entering:

.. code-block:: python

  >>> roi_from_single_subcube = my_sequence.index_as_cube[3, 1:3, 1:4]
  >>> roi_from_single_subcube.dimensions
  <Quantity [2., 3.] pix>
  >>> roi_from_single_subcube.array_axis_physical_types
  [('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
   ('em.wl',)]

This returns the same `~ndcube.NDCube` as above.  However, also as above,
we can induce the return type to be an `~ndcube.NDCubeSequence` by supplying
a length-1 `slice`.  As before, the same region of interest from the same
sub-cube is represeted, just with sequence and common axes of length 1.::

  >>> roi_length1_sequence = my_sequence.index_as_cube[3:4, 1:3, 1:4]
  >>> roi_length1_sequence.dimensions
  (<Quantity 1. pix>, <Quantity 1. pix>, <Quantity 2. pix>, <Quantity 3. pix>)
  >>> roi_length1_sequence.array_axis_physical_types
  [('meta.obs.sequence',),
   ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
   ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
   ('em.wl',)]

In the case the entire region came from a single sub-cube.  However,
`~ndcube.NDCubeSequence.index_as_cube` also works when the region of
interest spans multiple sub-cubes in the sequence.  Say we want the
same region of interest in the 2nd and 3rd cube dimensions, but this
time from the final slice along the 1st cube axis of the 1st sub-cube
the whole 2nd sub-cube and the 1st slice of the 3rd sub-cube.
In cube-like indexing this corresponds to slices 2 to 7 along to the
1st cube axis::

  >>> roi_across_subcubes = my_sequence.index_as_cube[2:7, 1:3, 1:4]
  >>> roi_across_subcubes.dimensions
  (<Quantity 3. pix>,
   <Quantity [1., 3., 1.] pix>,
   <Quantity 2. pix>,
   <Quantity 3. pix>)
  >>> roi_across_subcubes.array_axis_physical_types
  [('meta.obs.sequence',),
   ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
   ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
   ('em.wl',)]

Notice that since the sub-cubes are now of different lengths along the
common axis, the corresponding `~astropy.units.Quantity` gives the
lengths of each cube individually.

.. _collection_slicing::

Slicing NDCollections
=====================
Recall from the :ref:`ndcollection` section that members of an `~ndcube.NDCollection` can be accessed by slicing it with a string giving the member's name.

.. code-block:: python

  >>> my_collection['observations']  # doctest: +SKIP

However, also recall that we can mark axes of the member ND objects that are aligned.  The value in this is that it enables users to slice all the members of the collection simultaneously from the `~ndcube.NDCollection` level.  This can only be done for aligned axes.  Non-aligned axes must be sliced separately.  Nonethless, `~ndcube.NDCollection`'s slicing capability represents one of its greatest advantages over a simple Python `dict`, making it a powerful tool for rapidly and reliably cropping multiple components of a data set to a region of interest.  This has the potential to drastically speed up analysis workflows.

To demonstrate, let's instantiate an `~ndcube.NDCollection` with aligned axes, as we did in the :ref:`ndcollection` section.  (We have already defined ``my_cube`` in the :ref:`cube_slicing` section.)

.. code-block:: python

  >>> # Define derived linewidth NDCube to link with my_cube, defined above, in an NDCollection.
  >>> linewidth_data = np.ones((3, 4)) / 2 # dummy data
  >>> linewidth_wcs_dict = {
  ...    'CTYPE1': 'HPLT-TAN', 'CUNIT1': 'deg', 'CDELT1': 0.5, 'CRPIX1': 2, 'CRVAL1': 0.5, 'NAXIS1': 20,
  ...    'CTYPE2': 'HPLN-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.4, 'CRPIX2': 2, 'CRVAL2': 1, 'NAXIS2': 10}
  >>> linewidth_wcs = astropy.wcs.WCS(linewidth_wcs_dict)
  >>> linewidth_cube = NDCube(linewidth_data, linewidth_wcs)

  >>> # Enter my_cube, defined in a previous section, with the cube defined just above.
  >>> from ndcube import NDCollection
  >>> my_collection = NDCollection([("observations", my_cube), ("linewidths", linewidth_cube)],
  ...                              aligned_axes=(0, 1))

To slice an `~ndcube.NDCollection` you can simply do the following:

.. code-block:: python

  >>> sliced_collection = my_collection[1:3, 3:8]
  >>> sliced_collection.keys()
  dict_keys(['observations', 'linewidths'])
  >>> sliced_collection.aligned_dimensions
  <Quantity [2.0, 1.0] pix>

Note that we still have the same number of ND objects, but both have been sliced using the inputs provided by the user.  The slicing takes account of and updates the aligned axis information.  Therefore a self-consistent result would be obtained even if the aligned axes are not in order.

.. code-block:: python

  >>> linewidth_wcs_dict_reversed = {
  ...    'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5, 'NAXIS2': 20,
  ...    'CTYPE1': 'HPLN-TAN', 'CUNIT1': 'deg', 'CDELT1': 0.4, 'CRPIX1': 2, 'CRVAL1': 1, 'NAXIS1': 10}
  >>> linewidth_wcs_reversed = astropy.wcs.WCS(linewidth_wcs_dict_reversed)
  >>> linewidth_cube_reversed = NDCube(linewidth_data.transpose(), linewidth_wcs_reversed)

  >>> my_collection_reversed = NDCollection([("observations", my_cube),
  ...                                        ("linewidths", linewidth_cube_reversed)],
  ...                                       aligned_axes=((0, 1), (1, 0)))

  >>> sliced_collection_reversed = my_collection_reversed[1:3, 3:8]
  >>> sliced_collection_reversed.keys()
  dict_keys(['observations', 'linewidths'])
  >>> sliced_collection_reversed.aligned_dimensions
  <Quantity [2.0, 1.0] pix>
