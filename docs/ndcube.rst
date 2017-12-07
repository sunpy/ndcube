======
NDCube
======

`~ndcube.NDCube` is the fundamental class of the ndcube package and is designed
to handle data contained in a single N-D array described by a single
WCS transformation.  `~ndcube.NDCube` is subclassed from `astropy.nddata.NDData`
and so inherits the same attributes for data, wcs, uncertainty, mask,
meta, and unit.  The WCS object contained in the ``.wcs`` attribute is
subclassed from `astropy.wcs.WCS` and contains a few additional
attributes to enable to keep track of its relationship to the data.

Initialization
--------------

To initialize the most basic `~ndcube.NDCube` object, all you need is a
`numpy.ndarray` containing the data, and an `astropy.wcs.WCS` object
describing the transformation from array-element space to real world
coordinates.  Let's create a 3-D array of data with shape (3, 4, 5)
where every value is 1::

  >>> import numpy as np
  >>> data = np.ones((3, 4, 5))

Now let's create an `astropy.wcs.WCS` object describing the
translation from the array element coordinates to real world
coordinates.  Let's the first data axis be helioprojective longitude,
the second by helioprojective latitude, and the third be wavelength.
Note that due to (confusing) convention, the order of the axes in the
WCS object is reversed relative to the data array.

  >>> import astropy.wcs
  >>> wcs_input_dict = {
  ... 'CTYPE1': 'WAVE    ', 'CUNIT1': 'Angstrom', 'CDELT1': 0.2, 'CRPIX1': 0, 'CRVAL1': 10, 'NAXIS1': 5,
  ... 'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5, 'NAXIS2': 4,
  ... 'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 3}
  >>> input_wcs = astropy.wcs.WCS(wcs_input_dict)

Now that we have a data array and a corresponding WCS object, we can
create an `~ndcube.NDCube` instance by doing::

  >>> from ndcube import NDCube
  >>> my_cube = NDCube(data, input_wcs)

The data array is stored in the ``mycube.data`` attribute while the
WCS object is stored in the ``my_cube.wcs`` attribute.  However, when
manipulating/slicing the data is it better to slice the object as a
whole.  (See section on :ref:`ndcube_slicing`.)  So the ``.data`` attribute
should only be used to access a specific value(s) in the data.
Another thing to note is that as part of the initialization, the WCS
object is converted from an `astropy.wcs.WCS` to an
`ndcube.wcs_util.WCS` object which has some additional features for
tracking "missing axes", etc. (See section on :ref:`missing_axes`.)

Thanks to the fact that `~ndcube.NDCube` is subclassed from
`astropy.nddata.NDData`, you can also supply additional data to the
`~ndcube.NDCube` instance.  These include: metadata (`dict` or
dict-like) located at `NDCube.metadata`; a data mask
(boolean `numpy.ndarray`) located at`NDCube.mask` highlighting, for
example, reliable and unreliable pixels; an uncertainty array
(`numpy.ndarray`) located at `NDCube.uncertainty` describing the
uncertainty of each data array value;  and a unit
(`astropy.units.Unit` or unit `str`). For example::

  >>> mask = np.empty((3, 4, 5), dtype=object)
  >>> mask[:, :, :] = False
  >>> meta = {"Description": "This is example NDCube metadata."}
  >>> my_cube = NDCube(data, input_wcs, uncertainty=np.sqrt(data),
  ...                         mask=mask, meta=meta, unit=None)
  INFO: uncertainty should have attribute uncertainty_type. [astropy.nddata.nddata]

N.B. The above warning is due to the fact that
`astropy.nddata.uncertainty` is recommended to have an
``uncertainty_type`` attribute giving a string describing the type of
uncertainty.  However, this is not required.

Dimensions
----------

NDCube has a useful property for inspecting its data shape and
axis types, `~ndcube.NDCube.dimensions`::

  >>> my_cube.dimensions
  DimensionPair(shape=<Quantity [ 3., 4., 5.] pix>, axis_types=['HPLN-TAN', 'HPLT-TAN', 'WAVE'])

This returns a named tuple with a ``shape`` and ``axis_types`` attribute.
``shape`` is an `~astropy.units.Quantity` of pixel units giving the
length of each dimension in the `~ndcube.NDCube` while
``axis_types`` is `list` of `str` giving the WCS transformation type for
each axis. Here the shape and axis types are given in data order, not
WCS order.

As the dimensions property returns a named tuple, the ``shape`` and
``axis_types`` can be accessed directly::

  >>> my_cube.dimensions.shape
  <Quantity [ 3., 4., 5.] pix>
  >>> my_cube.dimensions.axis_types
  ['HPLN-TAN', 'HPLT-TAN', 'WAVE']

.. _ndcube_slicing:

Slicing
-------

Arguably NDCube's most powerful capability is its slicing.  Slicing an
`~ndcube.NDCube` object using the standard slicing notation allows
users to access sub-regions of their data while simultaneously slicing
not only the other array attributes (e.g. uncertainty, mask, etc.) but
also the WCS object.  This ensures that even though the data array has
changed size and shape, each array element will still corresponding to
the same real world coordinates as they did before.  An example of how
to slice a 3-D `~ndcube.NDCube` object is::

  >>> my_cube_roi = my_cube[3:5, 10:100, 30:37]

Slicing can also reduce the dimension of an `~ndcube.NDCube`, e.g.::

  >>> my_2d_cube = my_cube[0, 10:100, 30:37]

In addition to slicing by index, `~ndcube.NDCube` supports a basic
version of slicing/index by real world coordinate via the
`~ndcube.NDCube.crop_by_coords` method.  This takes a list of
`astropy.units.Quantity` representing the real world coordinates of
the lower left corner of the region of interest.  The order of the
coordinates must be the same as the order of the data axes.  A second
iterable of `~astropy.units.Quantity` must also be provided which gives
the widths of the region of interest in each data axis::

  >>> from astropy.units import Quantity
  >>> my_cube_roi = my_cube.crop_by_coords(
  ... [Quantity(0.7, unit="deg"), Quantity(1.3e-5, unit="deg"), Quantity(1.04e-9, unit="m")],
  ... [Quantity(0.6, unit="deg"), Quantity(1., unit="deg"), Quantity(0.08e-9, unit="m")])

This method does not rebin or interpolate the data if the region of interest
defined does not perfectly map onto the array's "pixel" grid.  Instead
it translates from real world to pixel coordinates and rounds to the
nearest integer before indexing/slicing the `~ndcube.NDCube` object.
Therefore it should be noted that slightly different inputs to this
method can result in the same output.

.. _missing_axes:

Missing Axes
------------

Some WCS axis types are coupled.  For example, the helioprojective
latitude and longitude of the Sun as viewed by a camera on a satellite
orbiting Earth do not map independently to the pixel grid.  Instead,
the longitude changes as we move vertically along the same x-position
if that single x-position is aligned anywhere other than perfectly
north-south along the Sun's central meridian.  The analagous is true
of the latitude for any y-pixel position not perfectly aligned with
the Sun's equator. Therefore, knowledge of both the latitude and
longitude must be known to derive the pixel position along a single
spatial axis and vice versa.

However, there are occasions where a data array may only contain one
spatial axis, e.g. in data from a slit-spectrograph instrument.  In
this case, simply extracting the corresponding latitude or longitude
axis from the WCS object would cause the translations to break.

To deal with this scenario, `~ndcube.NDCube` supports "missing" WCS axes.  An
additional attribute is added to the object (NDCube.wcs.missing_axis) which
is a list of `bool` type indicating which WCS axes do not have a
corresponding data axis.  This allows translation information on
coupled axes to persist even if the data axes do not.  This feature
makes in possible for `~ndcube.NDCube` to seamlessly reduce the data
dimensionality via slicing and also handle data types with only one
spatial dimension, like those from a slit-spectrograph instrument
which would have otherwise been impossible.  In the majority of cases
a user will not need to worry about this feature.  But it is useful to
be aware of as many of the coordinate transformation functionalities
of `~ndcube.NDCube` are only made possible by the missing axis feature.

Extra Coordinates
-----------------

In the case of some datasets, there may be additional translations
between the array elements and real world coordinates that are 
not included in the WCS.  Consider a 3-D data cube from a rastering
slit-spectrograph instrument.  The first axis corresponds to the
x-position of the slit as it steps across a region of interest in a
given pattern.  The second corresponds to latitude along the slit.  And
the third axis corresponds to wavelength.  However, the first axis also
corresponds to time, as it takes time for the slit to move and then
take another exposure which results in a new spectrogram (y-position
vs. wavelength). It would be very useful to have the measurement times
associated along the x-axis associated.  However, the WCS can only
handle one translation per axis.

Fortunately, `~ndcube.NDCube` has a solution to this.  Values at
integer (pixel) steps along an axis can be stored within the object
and accessed via the `~ndcube.NDCube.extra_coords` property. To
attach extra coordinates to an `~ndcube.NDCube` instance, provide a
iterable of tuples of the form (`str`, `int`, `~astropy.units.Quantity`
or `list`) where the 0th entry gives the name of the coordinate, the
1st entry gives the data axis to which the extra coordinate
corresponds, and the 2nd entry gives the value of that coordinate at
each pixel along the axis.  So to add timestamps along the 0th axis of
``my_cube`` we do:: 

  >>> from datetime import datetime, timedelta
  >>> # Define our timestamps.  Must be same length as data axis.
  >>> axis_length = int(my_cube.dimensions.shape[0].value)
  >>> timestamps = [datetime(2000, 1, 1)+timedelta(minutes=i)
  ...               for i in range(axis_length)]
  >>> extra_coords_input = [("time", 0, timestamps)]
  >>> # Generate NDCube as above, except now set extra_coords kwarg.
  >>> my_cube = NDCube(data, input_wcs, uncertainty=np.sqrt(data),
  ...                  mask=mask, meta=meta, unit=None,
  ...                  extra_coords=extra_coords_input)
  INFO: uncertainty should have attribute uncertainty_type. [astropy.nddata.nddata]

The `~ndcube.NDCube.extra_coords` property returns a dictionary where each key
is a coordinate name entered by the user.  The value of each key is
itself another dictionary with keys ``'axis'`` and ``'value'`` giving the
corresponding data axis number and coordinate value at each pixel as
supplied by the user::

  >>> my_cube.extra_coords
  {'time': {'axis': 0,
    'value': [datetime.datetime(2000, 1, 1, 0, 0),
     datetime.datetime(2000, 1, 1, 0, 1),
     datetime.datetime(2000, 1, 1, 0, 2)]}}

Just like the data array and the WCS object, the extra coordinates are
sliced automatically when the `~ndcube.NDCube` object is sliced.  So
if we take the first slice of ``my_cube`` in the 0th axis, the extra
time coordinate will only contain the value from that slice.::

  >>> my_cube[0].extra_coords
  {'time': {'axis': None, 'value': datetime.datetime(2000, 1, 1, 0, 0)}}

Note that the ``axis`` value is now ``None`` because the dimensionality of the
`~ndcube.NDCube` has been reduced via the slicing::

  >>> my_cube[0].dimensions.shape
  <Quantity [ 4., 5.] pix>

and so the ``time`` extra coordinate no longer corresponds to a data
axis.  This would not have been the case if we had done the slicing
so the length of the 0th axis was >1::

  >>> my_cube[0:2].dimensions.shape
  <Quantity [ 2., 4., 5.] pix>
  >>> my_cube[0:2].extra_coords
  {'time': {'axis': 0,
    'value': [datetime.datetime(2000, 1, 1, 0, 0),
     datetime.datetime(2000, 1, 1, 0, 1)]}}

Plotting
--------

To quickly and easily visualize N-D data, `~ndcube.NDCube` provides a
simple-to-use, yet powerful plotting method, `~ndcube.NDCube.plot`,
which produces a sensible visualization based on the dimensionality of
the data.  It is intended to be a useful quicklook tool and not a
replacement for high quality plots or animations, e.g. for
publications.  The plot method can be called very simply, like so::

  >>> my_cube.plot() # doctest: +SKIP

The type of visualization returned depends on the dimensionality of
the data within the `~ndcube.NDCube` object.  For 1-D data a line plot
is produced, similar to `matplotlib.pyplot.plot`.  For 2-D data, an
image is produced similar to that of `matplotlib.pyplot.imshow`.
While for a >2-D data, a
`sunpy.visualization.imageanimator.ImageAnimatorWCS` object is
returned.  This displays a 2-D image with sliders for each additional
dimension which allow the user to animate through the different values
of each dimension and see the effect in the 2-D image.

No args are required.  The necessary information to generate the plot
is derived from the data and metadata in the `~ndcube.NDCube`
itself. Setting the x and y ranges of the plot can be done simply by
indexing the `~ndcube.NDCube` object itself to the desired region of
interest and then calling the plot method, e.g.::

  >>> my_cube[0, 10:100, :].plot() # doctest: +SKIP

In addition to this, some optional kwargs can be used to customize the
plot.  The ``axis_ranges`` kwarg can be used to set the axes ticklabels.  See the
`~sunpy.visualization.imageanimator.ImageAnimatorWCS` documentation for
more detail.  However, if this is not set, the axis ticklabels are
automatically derived in real world coordination from the WCS obect
within the `~ndcube.NDCube`.

By default the final two data dimensions are used for the plot
axes in 2-D or greater visualizations, but this can be set by the user
using the ``images_axes`` kwarg::

  >>> my_cube.plot(image_axes=[0,1]) # doctest: +SKIP

where the first entry in the list gives the index of the data index to
go on the x-axis, and the second entry gives the index of the data
axis to go on the y-axis.

In addition, the units of the axes or the data can be set by the
``unit_x_axis``, ``unit_y_axis``, unit kwargs.  However, if not set,
these are derived from the `~ndcube.NDCube` wcs and unit attributes.

Coordinate Transformations
--------------------------

The fundamental point the WCS system is the ability to easily
translate between pixel and real world coordinates.  For this purpose, 
`~ndcube.NDCube` provides convenience wrappers for the better known
astropy functions, `astropy.wcs.WCS.all_pix2world` and
`astropy.wcs.WCS.all_world2pix`. These are
`~ndcube.NDCube.pixel_to_world` and `~ndcube.NDCube.world_to_pixel`.
It is highly recommended that when using `~ndcube.NDCube` these
convenience wrappers are used rather than the original astropy
functions for a few reasons. For example, they can track house-keeping
data, are aware of "missing" WCS axis, are unit-aware, etc.

To use `~ndcube.NDCube.pixel_to_world`, simply input a list of
`~astropy.units.Quantity` objects with pixel units. Each
`~astropy.units.Quantity` corresponds to an axis so the number of
`~astropy.units.Quantity` objects should equal the number of data
axes.  Also, the order of the quantities should correspond to the
data axes' order, not the WCS order.  The nth element of each
`~astropy.units.Quantity` describes the pixel coordinate in each axis
of the nth pixel to be transformed. For example, if we wanted to
transform the pixel coordinates of the pixel (2, 3, 4) in ``my_cube``
we would do::

  >>> import astropy.units as u
  >>> real_world_coords = my_cube.pixel_to_world(
  ... [Quantity([2], unit=u.pix), Quantity([3], unit=u.pix), Quantity([4], unit=u.pix)])

To convert two pixels with pixel coordinates (2, 3, 4) and (5, 6, 7),
we would call pixel_to_world like so::

  >>> real_world_coords = my_cube.pixel_to_world(
  ... [Quantity([2, 5], unit=u.pix), Quantity([3, 6], unit=u.pix), Quantity([4, 7], unit=u.pix)])

As can be seen, since each `~astropy.units.Quantity` describes a
different pixel coordinate of the same number of pixels, the lengths
of each `~astropy.units.Quantity` must be the same.

`~ndcube.NDCube.pixel_to_world` returns a similar list of Quantities
as to those that were input, except that they are now in real world
coordinates::

  >>> real_world_coords
  [<Quantity [ 1.40006967, 2.6002542 ] deg>,
   <Quantity [ 1.49986193, 2.99724799] deg>,
   <Quantity [  1.10000000e-09,  1.16000000e-09] m>]

The exact units used are defined within the `~ndcube.NDCube`
instance's `~ndcube.wcs_util.WCS` object.  Once again, the coordinates
of the nth pixel is given by the nth element of each of the
`~astropy.units.Quantity` objects returned.

Using `~ndcube.NDCube.world_to_pixel` to convert real world
coordinates to pixel coordinates is exactly the same, but in reverse.
This time the input `~astropy.units.Quantity` objects must be in real
world coordinates compatible with those defined in the
`~ndcube.NDCube` instance's `~ndcube.wcs_util.WCS` object.  The output
is a list of `~astropy.units.Quantity` objects in pixel units is
returned::

  >>> pixel_coords = my_cube.world_to_pixel(
  ... [Quantity(1.40006967, unit="deg"), Quantity(1.49986193, unit="deg"),
  ...  Quantity(1.10000000e-09,  unit="m")])
  >>> pixel_coords
  [<Quantity 2.0000000101029034 pix>,
   <Quantity 2.9999999961693913 pix>,
   <Quantity 3.999999999999993 pix>]

Both `~ndcube.NDCube.pixel_to_world` and
`~ndcube.NDCube.world_to_pixel` have an additional optional kwarg,
``origin``, whose default is 0.  This is the same as the ``origin`` arg in
`~astropy.wcs.WCS.all_pix2world` and `~astropy.wcs.WCS.all_world2pix`
and defines whether the WCS translation is 0-based (C) or 1-based
(FORTRAN).  Changing this kwarg will result in the pixel coordinates
being offset by 1.  In most cases, the approriate setting will be
``origin=0``, but 1-based may be required for writing the WCS
translations to a FITS header.
