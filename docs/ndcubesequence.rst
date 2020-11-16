.. doctest-skip-all::
    all

.. _ndcubesequence:

==============
NDCubeSequence
==============

`~ndcube.NDCubeSequence` is a class for handling multiple
`~ndcube.NDCube` objects as though they were one contiguous data set.
Another way of thinking about it is that `~ndcube.NDCubeSequence`
provides the ability to manipulate a data set described by multiple
separate WCS transformations.

Regarding implementation, an `~ndcube.NDCubeSequence` instance is
effectively a list of `~ndcube.NDCube` instances with some helper
methods attached.

Initialization
==============

To initialize the most basic `~ndcube.NDCubeSequence` object, all you
need is a list of `~ndcube.NDCube` instances.  So let us first define
three 3-D NDCubes for slit-spectrograph data as we did in the NDCube
section of this tutorial.  First we define the data arrays and WCS objects::

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

Let's also define an extra coordinate of time assigned to the 0th cube
data axis and another label coordinate assigned to the cubes as
wholes.  (See NDCube section of this guide of more detail.) Let the
slices along the 0th axis be separated by one minute and the slices in
preceding cube are followed directly in time by the slices in the next::

  >>> from datetime import datetime, timedelta
  >>> timestamps0 = [datetime(2000, 1, 1)+timedelta(minutes=i) for i in range(data0.shape[0])]
  >>> timestamps1 = [timestamps0[-1]+timedelta(minutes=i+1) for i in range(data1.shape[0])]
  >>> timestamps2 = [timestamps1[-1]+timedelta(minutes=i+1) for i in range(data2.shape[0])]
  >>> extra_coords_input0 = [("time", 0, timestamps0), ("label", None, "hello")]
  >>> extra_coords_input1 = [("time", 0, timestamps1), ("label", None, "world")]
  >>> extra_coords_input2 = [("time", 0, timestamps2), ("label", None, "!")]

Now we can define our cubes.

  >>> from ndcube import NDCube
  >>> from ndcube import NDCubeSequence
  >>> # Define a mask such that all array elements are unmasked.
  >>> mask = np.empty(data0.shape, dtype=object)
  >>> mask[:, :, :] = False
  >>> cube_meta = {"Description": "This is example NDCube metadata."}
  >>> my_cube0 = NDCube(data0, input_wcs, uncertainty=np.sqrt(data0),
  ...                          mask=mask, meta=cube_meta, unit=None,
  ...                          extra_coords=extra_coords_input0)
  INFO: uncertainty should have attribute uncertainty_type. [astropy.nddata.nddata]
  >>> my_cube1 = NDCube(data1, input_wcs, uncertainty=np.sqrt(data1),
  ...                          mask=mask, meta=cube_meta, unit=None,
  ...                          extra_coords=extra_coords_input1)
  INFO: uncertainty should have attribute uncertainty_type. [astropy.nddata.nddata]
  >>> my_cube2 = NDCube(data2, input_wcs, uncertainty=np.sqrt(data2),
  ...                          mask=mask, meta=cube_meta, unit=None,
  ...                          extra_coords=extra_coords_input2)
  INFO: uncertainty should have attribute uncertainty_type. [astropy.nddata.nddata]

N.B. The above warnings are due to the fact that
`astropy.nddata.uncertainty` is recommended to have an
``uncertainty_type`` attribute giving a string describing the type of
uncertainty.  However, this is not required.  Also note that due to
laziness, we have used the same WCS translations in each
`~ndcube.NDCube` instance above.  However, it would be more common for
each `~ndcube.NDCube` instance to have a different WCS, and in that
case the usefulness of `~ndcube.NDCubeSequence` is more
pronounced. Nonetheless, this case can still be used to adequately
demonstrate the capabilities of `~ndcube.NDCubeSequence`.

Finally, creating an `~ndcube.NDCubeSequence` becomes is simple::

  >>> my_sequence = NDCubeSequence([my_cube0, my_cube1, my_cube2])

While, each `~ndcube.NDCube` in the `~ndcube.NDCubeSequence` can have
its own meta, it is also possible to supply additional metadata upon
initialization of the `~ndcube.NDCubeSequence`.  This metadata may be
common to all sub-cubes or is specific to the sequence rather than the
sub-cubes. This metadata is input as a dictionary::

  >>> my_sequence_metadata = {"Description": "This is some sample NDCubeSequence metadata."}
  >>> my_sequence = NDCubeSequence([my_cube0, my_cube1, my_cube2],
  ...                              meta=my_sequence_metadata)

and stored in the ``my_sequence.meta`` attribute.  Meanwhile, the
`~ndcube.NDCube` instances are stored in ``my_sequence.data``.
However, analgously to `~ndcube.NDCube`, it is strongly advised that
the data is manipulated by slicing the `~ndcube.NDCubeSequence` rather
than more manually delving into the ``.data`` attribute.  For more
explanation, see the section on :ref:`sequence_slicing`.

Common Axis
===========

It is possible (although not required) to set a common axis of the
`~ndcube.NDCubeSequence`.  A common axis is defined as the axis of the
sub-cubes parallel to the axis of the sequence.

For example, assume the 0th axis of the sub-cubes, ``my_cube0``,
``my_cube1`` and ``my_cube2`` in the `~ndcube.NDCubeSequence`,
``my_sequence``, represent time as we have indicated by setting the
``time`` extra coordinate. In this case, ``my_cube0`` represents
observations taken from a period directly before ``my_cube1`` and
``my_cube2`` and the sub-cubes are  ordered chronologically in the
sequence.  Then moving along the 0th axis of one sub-cube and moving
along the sequence axis from one cube to the next both represent
movement in time.  The difference is simply the size of the steps.
Therefore it can be said that the 0th axis of the sub-cubes is common
to the sequence.

To define a common axis, set the kwarg during intialization of
the `~ndcube.NDCubeSequence` to the desired data axis number::

  >>> my_sequence = NDCubeSequence([my_cube0, my_cube1, my_cube2],
  ...                              meta=my_sequence_metadata, common_axis=0)

Defining a common axis enables the full range of the
`~ndcube.NDCubeSequence` features to be utilized including
`ndcube.NDCubeSequence.plot`,
`ndcube.NDCubeSequence.common_axis_extra_coords`, and
`ndcube.NDCubeSequence.index_as_cube`. See following sections for
more details on these features.

.. _dimensions:

Dimensions
==========

Analagous to `ndcube.NDCube.dimensions`, there is also a
`ndcube.NDCubeSequence.dimensions` property for
easily inspecting the shape of an `~ndcube.NDCubeSequence` instance::

  >>> my_sequence.dimensions
  (<Quantity 3. pix>, <Quantity 3. pix>, <Quantity 4. pix>, <Quantity 5. pix>)

Slightly differently to `ndcube.NDCube.dimensions`,
`ndcube.NDCubeSequence.dimensions` returns a tuple of
`astropy.units.Quantity` instances with pixel units, giving the length
of each axis.  This is in constrast to the single
`~astropy.units.Quantity` returned by `~ndcube.NDCube`. This is
because `~ndcube.NDCubeSequence` supports sub-cubes of different
lengths along the common axis if it is set.  In that case, the
corresponding quantity in the dimensions tuple will have a length
greater than 1 and list the length of each sub-cube along the common
axis.

Equivalent to `ndcube.NDCube.array_axis_physical_types`,
`ndcube.NDCubeSequence.array_axis_physical_types` returns a list of
tuples of physical axis types.  The same `IVOA UCD1+` controlled words
<http://www.ivoa.net/documents/REC/UCD/UCDlist-20070402.html> are
used for the cube axes as is used in
`ndcube.NDCube.array_axis_physical_types`.  The sequence axis is given
the label ``'meta.obs.sequence'`` as it is the IVOA UCD1+ controlled
word that best describes it.  To call, simply do::

  >>> my_sequence.array_axis_physical_types
  [('meta.obs.sequence',), ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'), ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'), ('em.wl',)]

.. _sequence_slicing:

Slicing
=======

As with `~ndcube.NDCube`, slicing an `~ndcube.NDCubeSequence` using
the standard slicing API simulataneously slices the data arrays, WCS
objects, masks, uncertainty arrays, etc. in each relevant sub-cube.
For example, say we have three NDCubes in an `~ndcube.NDCubeSequence`,
each of shape ``(3, 4, 5)``.  Say we want to obtain a region of
interest from the 1st (not 0th) and 2nd cubes in the sequence.
Let's say the region of interest in each cube is defined as the 0th slice
along the 0th cube dimension, between the 1st and 2nd pixels (inclusive)
in the 2nd dimension and between the 1st and 3rd pixels (inclusive)
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
  [('meta.obs.sequence',), ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'), ('em.wl',)]

This will return a new `~ndcube.NDCubeSequence` with 2 2-D NDCubes,
one for each region of interest from each original sub-cube.
If we want our region of interest to only apply to a single sub-cube,
and we index the sequence axis with an `int`, an NDCube is returned::

  >>> roi_from_single_subcube = my_sequence[1, 0, 1:3, 1:4]
  >>> roi_from_single_subcube.dimensions
  <Quantity [2., 3.] pix>
  >>> roi_from_single_subcube.array_axis_physical_types
  [('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'), ('em.wl',)]

However, as with numpy slicing, we can induce the slicing operation to return
an `~ndcube.NDCubeSequence` by supplying a length-1 `slice` to the sequence
axis, rather than an `int`. This sequence will still represent the same region
of interest from the same single sub-cube, but the sequence axis will have a
length of 1, rather than removed.::

  >>> roi_length1_sequence = my_sequence[0:1, 0, 1:3, 1:4]
  >>> roi_length1_sequence.dimensions
  (<Quantity 1. pix>, <Quantity 2. pix>, <Quantity 3. pix>)
  >>> roi_length1_sequence.array_axis_physical_types
  [('meta.obs.sequence',), ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'), ('em.wl',)]


If a common axis has been defined for the `~ndcube.NDCubeSequence` one
can think of it as a contiguous data set with different sections along
the common axis described by different WCS translations.  Therefore it
would be useful to be able to index the sequence as though it were one
single cube.  This can be achieved with the
`ndcube.NDCubeSequence.index_as_cube` property.  In our above
example, ``my_sequence`` has a shape of ``(<Quantity 3. pix>,
<Quantity 3.0 pix>, <Quantity 4.0 pix>, <Quantity 5.0 pix>)`` and a
common axis of ``0``.  Therefore we can think of ``my_sequence``
as a having an effective cube-like shape of ``(<Quantity 9.0 pix>,
<Quantity 4.0 pix>, <Quantity 5.0 pix>)`` where the first sub-cube
extends along the 0th cube-like axis from 0 to 3, the second from 3 to
6 and the third from 6 to 9.  Say we want to extract the same region
of interest as above, i.e. ``my_sequence[1, 0, 1:3, 1:4]``.  Then
this can be acheived by entering::

  >>> roi_from_single_subcube = my_sequence.index_as_cube[3, 1:3, 1:4]
  >>> roi_from_single_subcube.dimensions
  <Quantity [2., 3.] pix>
  >>> roi_from_single_subcube.array_axis_physical_types
  [('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'), ('em.wl',)]


This returns the same `~ndcube.NDCube` as above.  However, also as above,
we can induce the return type to be an `~ndcube.NDCubeSequence` by supplying
a length-1 `slice`.  As before, the same region of interest from the same
sub-cube is represeted, just with sequence and common axes of length 1.::

  >>> roi_length1_sequence = my_sequence.index_as_cube[3:4, 1:3, 1:4]
  >>> roi_length1_sequence.dimensions
  (<Quantity 1. pix>, <Quantity 1. pix>, <Quantity 2. pix>, <Quantity 3. pix>)
  >>> roi_length1_sequence.array_axis_physical_types
  [('meta.obs.sequence',), ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'), ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'), ('em.wl',)]

In the case the entire region came from a single sub-cube.  However,
`~ndcube.NDCubeSequence.index_as_cube` also works when the region of
interest spans multiple sub-cubes in the sequence.  Say we want the
same region of interest in the 2nd and 3rd cube dimensions, but this
time from the final slice along the 0th cube axis of the 0th sub-cube
the whole 1st sub-cube and the 0th slice of the 2nd sub-cube.
In cube-like indexing this corresponds to slices 2 to 7 along to the
0th cube axis::

  >>> roi_across_subcubes = my_sequence.index_as_cube[2:7, 1:3, 1:4]
  >>> roi_across_subcubes.dimensions
  (<Quantity 3. pix>, <Quantity [1., 3., 1.] pix>, <Quantity 2. pix>, <Quantity 3. pix>)
  >>> roi_across_subcubes.array_axis_physical_types
  [('meta.obs.sequence',), ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'), ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'), ('em.wl',)]

Notice that since the sub-cubes are now of different lengths along the
common axis, the corresponding `~astropy.units.Quantity` gives the
lengths of each cube individually.  See section on :ref:`dimensions`
for more detail.

Cube-like Dimensions
====================

To help with handling an `~ndcube.NDCubeSequence` with a common axis
as if it were a single cube, there exist cube-like equivalents of the
`~ndcube.NDCubeSequence.dimensions`  and
`~ndcube.NDCubeSequence.array_axis_physical_types` methods.  They are
intuitively named `~ndcube.NDCubeSequence.cube_like_dimensions`  and
`~ndcube.NDCubeSequence.cube_like_array_axis_physical_types`.  These
give the lengths and physical types of the axes as if the data were
stored in a single `~ndcube.NDCube`.  So in the case of
``my_sequence``, with three sub-cubes, each with a length of 3 along
the common axis, we get::

  >>> my_sequence.cube_like_dimensions
  <Quantity [9., 4., 5.] pix>
  >>> my_sequence.cube_like_array_axis_physical_types
  [('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'), ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'), ('em.wl',)]

Note that `~ndcube.NDCubeSequence.cube_like_dimensions` returns a
single `~astropy.units.Quantity` in pixel units, as if it were
`ndcube.NDCube.dimensions`.  This is in contrast to
`ndcube.NDCubeSequence.dimensions` that returns a `tuple` of
`~astropy.units.Quantity`.

Common Axis Extra Coordinates
=============================

If a common axis is defined, it may be useful to view the extra
coordinates along that common axis defined by each of the sub-cube
`~ndcube.NDCube.extra_coords` as if the `~ndcube.NDCubeSequence` were
one contiguous Cube.  This can be done using the
``common_axis_extra_coords`` property::

  >>> my_sequence.common_axis_extra_coords
  {'time': array([datetime.datetime(2000, 1, 1, 0, 0),
        datetime.datetime(2000, 1, 1, 0, 1),
        datetime.datetime(2000, 1, 1, 0, 2),
        datetime.datetime(2000, 1, 1, 0, 3),
        datetime.datetime(2000, 1, 1, 0, 4),
        datetime.datetime(2000, 1, 1, 0, 5),
        datetime.datetime(2000, 1, 1, 0, 6),
        datetime.datetime(2000, 1, 1, 0, 7),
        datetime.datetime(2000, 1, 1, 0, 8)], dtype=object)}

This returns a dictionary where each key gives the name of a
coordinate.  The value of each key is the values of that coordinate
at each pixel along the common axis.  Since all these coordinates must
be along the common axis, it is not necessary to supply axis
information as it is with `ndcube.NDCube.extra_coords` making
`ndcube.NDCubeSequence.common_axis_extra_coords` simpler.  Because
this property has a functional form and calculates the dictionary
each time from the constituent sub-cubes' `ndcube.NDCube.extra_coords`
attributes, `ndcube.NDCubeSequence.common_axis_extra_coords` is
effectively sliced when the `~ndcube.NDCubeSequence` is sliced, e.g.::

  >>> my_sequence[1:3].common_axis_extra_coords
  {'time': array([datetime.datetime(2000, 1, 1, 0, 3),
        datetime.datetime(2000, 1, 1, 0, 4),
        datetime.datetime(2000, 1, 1, 0, 5),
        datetime.datetime(2000, 1, 1, 0, 6),
        datetime.datetime(2000, 1, 1, 0, 7),
        datetime.datetime(2000, 1, 1, 0, 8)], dtype=object)}

Sequence Axis Extra Coordinates
===============================
Analgous to `~ndcube.NDCubeSequence.common_axis_extra_coords`, it is
also possible to access the extra coordinates that are not assigned to any
`~ndcube.NDCube` data axis via the
`ndcube.NDCubeSequence.sequence_axis_extra_coords` property.  Whereas
`~ndcube.NDCubeSequence.common_axis_extra_coords` returns all the
extra coords with an ``'axis'`` value equal to the common axis,
`~ndcube.NDCubeSequence.sequence_axis_extra_coords` returns all extra
coords with an ``'axis'`` value of ``None``.  Another way of thinking
about this when there is no common axis set, is that they are
assigned to the sequence axis.  Hence the property's name.::

  >>> my_sequence.sequence_axis_extra_coords
  {'label': array(['hello', 'world', '!'], dtype=object)}

Explode Along Axis
==================

During analysis of some data - say of a stack of images - it may be
necessary to make some different fine-pointing adjustments to each
image that isn't accounted for the in the original WCS translations,
e.g. due to satellite wobble.  If these changes are not describable
with a single WCS object, it may be desirable to break up the N-D
sub-cubes of an `~ndcube.NDCubeSequence` into an sequence of sub-cubes
with dimension N-1. This would enable a separate WCS object to be
associated with each image and hence allow individual pointing
adjustments.

Rather than manually dividing the datacubes up and deriving the
corresponding WCS object for each exposure, `~ndcube.NDCubeSequence`
provides a useful method,
`~ndcube.NDCubeSequence.explode_along_axis`. To call it, simply provide
the number of the data cube axis along which you wish to break up the sub-cubes::

  >>> exploded_sequence = my_sequence.explode_along_axis(0)

Assuming we are using the same ``my_sequence`` as above, with
dimensions.shape ``(<Quantity 3.0 pix>, <Quantity 3.0 pix>, <Quantity
4.0 pix>, <Quantity 5.0 pix>)``, the ``exploded_sequence`` will be an
`~ndcube.NDCubeSequence` of nine 2-D NDCubes each with shape
``(<Quantity 4.0 pix>, <Quantity 5.0 pix>)``.::

  >>> # Check old and new shapes of the squence
  >>> my_sequence.dimensions
  (<Quantity 3. pix>, <Quantity 3. pix>, <Quantity 4. pix>, <Quantity 5. pix>)
  >>> exploded_sequence.dimensions
  (<Quantity 9. pix>, <Quantity 4. pix>, <Quantity 5. pix>)

Note that any cube axis can be input.  A common axis need not be
defined.

Plotting
========

Since ndcube 2.0, `~ndcube.NDCubeSequence` does not provide plotting methods.
The rationale is explained in `Issue #315 <https://github.com/sunpy/ndcube/issues/315>`
in our GitHub repo.
If you feel that `~ndcube.NDCubeSequence` should support plotting again,
please read and comment on that issue telling us about your use case.
Better still, let us know that you would like to work on the necessary tools to enable
sequence plotting, let us know with a comment on that issue.

Despite this, you can still visualize the data in `~ndcube.NDCubeSequence` in a number of ways.
You can slice out a single `~ndcube.NDCube` and use its `~ndcube.NDCube.plot` method.
You can extract the data and use the myriad of plotting packages available in
the Python ecosystem.
Finally, if you want to be advanced enough, you can write your own mixin class to define
the plotting methods.
Below, we will outline these latter two options in a little more detail.

Extracting and Plotting NDCubeSequence Data with Matplotlib
-----------------------------------------------------------

In order to produce plots (or perform other analysis) outside of the ``ndcube`` framework,
it may be useful to extract the data from the `~ndcube.NDCubeSequence` into single
`~numpy.ndarray` instances.
In the above examples we defined the `my_sequence` `~ndcube.NDCubeSequence` object.::

    >>> # Print dimensions of my_sequence as a reminder
    >>> print(my_sequence.dimensions)
    (<Quantity 3. pix>, <Quantity 3. pix>, <Quantity 4. pix>, <Quantity 5. pix>)

To make a 4D array out of the data arrays within the `~ndcube.NDCubes` of `my_sequence`.::

    >>> # Make a single 4D array of data in sequence with the sequence axis as the 0th.
    >>> data4d = np.stack([cube.data for cube in my_sequence.data], axis=0)
    >>> print(data.shape)
    (3, 3, 4, 5)

The same applies to other array-like data in the `~ndcube.NDCubeSequence`, like
``uncertainty`` and ``mask``.
If instead, we want to define a 3D array where every `~ndcube.NDCube` in the
`~ndcube.NDCubeSequence` is appended along the ``common_axis``,
we can use `numpy.concatenate` function::

    >>> # Make a 3D array
    >>> data3d = np.concatenate([cube.data for cube in my_sequence.data],
                                axis=my_sequence._common_axis)
    >>> print(data.shape)
    (9, 4, 5)

Having extracted the data, we can now use matplotlib to visualize it.
Let's say we want to produce a timeseries of how intensity changes in a
given pixel at a given wavelength.  We stored time in ``my_sequence.global_coords``
and associated it with the ``common_axis``.  Therefore, we could do::

    >>> import matplotlib.pyplot as plt
    >>> # Get intensity at pixel 0, 0, 0 in each cube.
    >>> intensity = np.array([cube.data[0, 0, 0] for cube in my_sequence])
    >>> times = my_sequence.common_axis_coords["time"]
    >>> plt.plot(times, intensity)
    >>> plt.show()

Alternatively, we could produce a 2D dynamic spectrum showing how the spectrum
in a given pixel changes over time.::

    >>> import matplotlib as mpl
    >>> import matplotlib.pyplot as plt
    >>> from astropy.time import Time
    >>> # Combine spectrum over time for pixel 0, 0.
    >>> spectrum_sequence = my_sequence[0, 0]
    >>> intensity = np.stack([cube.data for cube in spectrum_sequence[0, 0], axis=0)
    >>> times = Time(spectrum_sequence.sequence_axis_coords["time"])
    >>> # Assume that the wavelength in each pixel doesn't change as we move through the sequence.
    >>> wavelength = spectrum_sequence[0].axis_world_coords("em.wl")
    >>> # As the times may not be uniform, we can use NonUniformImage
    >>> # to show non-uniform pixel sizes.
    >>> fig, ax = plt.subplots(1, 1)
    >>> im = mpl.image.NonUniformImage(
    ...     ax, extent=(times[0], times[-1], wavelength[0], wavelength[-1]))
    >>> im.set_data(times, wavelength, intensity)
    >>> ax.add_image(im)
    >>> ax.set_xlim(times[0], times[-1])
    >>> ax.set_ylim(wavelength[0], wavelength[-1])
    >>> plt.show()

Now let's say we want to animate our data, for example, show how the intensity
changes over wavelength and time.
For this we can use `~ndcube.visualization.animator.ImageAnimator`.
This class is not well suited to displaying the complex relationship between coordinates
that we are used to with `~astropy.visualization.wcsaxes.WCSAxes`.
For example, non-linear coordinates non-independent coordinates.
The difficulty and complexity in correctly representing this in a generalized way
when dealing with a sequence of WCS objects is one reason plotting is currently
no longer supported by `~ndcube.NDCubeSequence`.
Nontheless, `~ndcube.visualization.animator.ImageAnimator` can still give us an idea
of how the data is changing.
In ``my_sequence``, the sequence axis represents time, the 0th and 1st cube axes
represent latittude and longitude, while the final axis represents wavelength.
Therefore, we could do the following::

    >>> from ndcube.visualization import ImageAnimator
    >>> data = np.stack([cube.data for cube in my_sequence.data], axis=0)
    >>> time_range = [my_sequence[0, 0].global_coords.get_coord("time"),
                      my_sequence[-1, 0].global_coords.get_coord("time")]
    >>> # Assume that the field of view or wavelength grid is not changing over time.
    >>> # Also assume the coordinates are independent and linear with the pixel grid.
    >>> lon, lat, wavelength = my_sequence[0].axis_world_coords_values(wcs=my_sequence[0].wcs)
    >>> lon_range = [lon[0], lon[-1]]
    >>> lat_range = [lat[0], lat[-1]]
    >>> wave_range = [wavelength[0], wavelength[-1]]
    >>> animation = ImageAnimator(data, image_axes=[2, 1],
                                  axis_ranges=[time_range, lon_range, lat_range, wave_range])
    >>> plt.show()

Alternatively we can animate how the one 1-D spectrum changes by using
`~ndcube.visualization.animator.LineAnimator`::

    >>> from ndcube.visualization import ImageAnimator
    >>> data = np.stack([cube.data for cube in my_sequence.data], axis=0)
    >>> time_range = [my_sequence[0, 0].global_coords.get_coord("time"),
                      my_sequence[-1, 0].global_coords.get_coord("time")]
    >>> # Assume that the field of view or wavelength grid is not changing over time.
    >>> # Also assume the coordinates are independent and linear with the pixel grid.
    >>> lon, lat, wavelength = my_sequence[0].axis_world_coords_values()
    >>> lon_range = [lon[0], lon[-1]]
    >>> lat_range = [lat[0], lat[-1]]
    >>> wave_range = [wavelength[0], wavelength[-1]]
    >>> animation = LineAnimator(data, plot_axis_index=-1,
                                 axis_ranges=[time_range, lon_range, lat_range, wave_range])
    >>> plt.show()

Writing Your Own NDCubeSequence Plot Mixin
------------------------------------------

Just because ndcube no longer provides plotting support doesn't mean you can't write your own
plotting functionality for `~ndcube.NDCubeSequence`.
In many cases, this might be simpler as you may be able to make some assumptions about the
data you will be analyzing and therefore won't have to write as generalized a tool.
The best way to do this is to write your own mixin class defining the plot methods, e.g.

.. code-block:: python

   class MySequencePlotMixin:
       def plot(self, **kwargs):
           pass  # Write code to plot data here.

       def plot_as_cube(self, **kwargs):
           pass  # Write code to plot data concatenated along common axis here.

Then you can create your own ``NDCubeSequence`` by combining your mixin with
`~ndcube.NDCubeSequenceBase` which holds all the non-plotting functionality of the
`~ndcube.NDCubeSequence`.

.. code-block:: python

    class MySequence(NDCubeSequenceBase, MySequencePlotMixin):

This will create a new class, ``MySequence``, which contains all the functionality of
`~ndcube.NDCubeSequence` plus the plot methods you've defined in ``MySequencePlotMixin``.

There are many other ways you could visualize the data in your `~ndcube.NDCubeSequence`
and many other visualization packages in the Python ecosystem that you could use.
These examples show just a few simple ways.  But hopefully this has shown you that
it's still possible to visualize the data in your `~ndcube.NDCubeSequence`,
whether by creating your own mixin, following the above examples, or by using
some other infrastructure.
