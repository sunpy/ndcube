.. _ndcube:

======
NDCube
======

`~ndcube.NDCube` is the fundamental class of the ndcube package and is designed
to handle data contained in a single N-D array described by a single
set of WCS transformations.  `~ndcube.NDCube` is subclassed from
`astropy.nddata.NDData` and so inherits the same attributes for data,
wcs, uncertainty, mask, meta, and unit.  The WCS object contained in
the ``.wcs`` attribute is subclassed from `astropy.wcs.WCS` and
contains a few additional attributes to enable to keep track of its
relationship to the data.

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
coordinates.  Let the first data axis be helioprojective longitude,
the second be helioprojective latitude, and the third be wavelength.
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
`ndcube.utils.wcs.WCS` object which has some additional features for
tracking "missing axes", etc. (See section on :ref:`missing_axes`.)

Thanks to the fact that `~ndcube.NDCube` is subclassed from
`astropy.nddata.NDData`, you can also supply additional data to the
`~ndcube.NDCube` instance.  These include: metadata (`dict` or
dict-like) located at `NDCube.meta`; a data mask
(boolean `numpy.ndarray`) located at `NDCube.mask` marking, for
example, reliable and unreliable pixels; an uncertainty array
(`numpy.ndarray`) located at `NDCube.uncertainty` describing the
uncertainty of each data array value;  and a unit
(`astropy.units.Unit` or unit `str`). For example::

  >>> mask = np.zeros_like(my_cube.data, dtype=bool)
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

`~ndcube.NDCube` has useful properties for inspecting its data shape and
axis types, `~ndcube.NDCube.dimensions` and
`~ndcube.NDCube.world_axis_physical_types`::

  >>> my_cube.dimensions
  <Quantity [3., 4., 5.] pix>
  >>> my_cube.world_axis_physical_types
  ('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat', 'em.wl')

`~ndcube.NDCube.dimensions` returns an `~astropy.units.Quantity` of
pixel units giving the length of each dimension in the
`~ndcube.NDCube` while `~ndcube.NDCube.world_axis_physical_types`
returns an iterable of strings denoting the type of physical property
represented by each axis.  The axis names are in accordance with the
International Virtual Observatory Alliance (IVOA)
`UCD1+ controlled vocabulary <http://www.ivoa.net/documents/REC/UCD/UCDlist-20070402.html>`_.
Here the shape and axis types are given in data order, not WCS order.

.. _ndcube_slicing:

Slicing
-------

Arguably NDCube's most powerful capability is its slicing.  Slicing an
`~ndcube.NDCube` instance using the standard slicing notation allows
users to access sub-regions of their data while simultaneously slicing
not only the other array attributes (e.g. uncertainty, mask, etc.) but
also the WCS object.  This ensures that even though the data array has
changed size and shape, each array element will still correspond to
the same real world coordinates as they did before.  An example of how
to slice a 3-D `~ndcube.NDCube` object is::

  >>> my_cube_roi = my_cube[3:5, 10:100, 30:37]

Slicing can also reduce the dimension of an `~ndcube.NDCube`, e.g.::

  >>> my_2d_cube = my_cube[0, 10:100, 30:37]

In addition to slicing by index, `~ndcube.NDCube` supports a basic
version of slicing/indexing by real world coordinates via the
`~ndcube.NDCube.crop_by_coords` method.  This takes a list of
`astropy.units.Quantity` instances representing the minimum real world
coordinates of the region of interest in each dimension.  The
order of the coordinates must be the same as the order of the data
axes.  A second iterable of `~astropy.units.Quantity` must also be
provided which gives the widths of the region of interest in each data
axis::

  >>> import astropy.units as u
  >>> my_cube_roi = my_cube.crop_by_coords([0.7*u.deg, 1.3e-5*u.deg, 1.04e-9*u.m],
  ...                                     [0.6*u.deg, 1.*u.deg, 0.08e-9*u.m])

This method does not rebin or interpolate the data if the region of interest
does not perfectly map onto the array's "pixel" grid.  Instead
it translates from real world to pixel coordinates and rounds to the
nearest integer pixel before indexing/slicing the `~ndcube.NDCube`
instance. Therefore it should be noted that slightly different inputs to
this method can result in the same output.

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

However, there are occasions when a data array may only contain one
spatial axis, e.g. data from a slit-spectrograph.  In this case,
simply extracting the corresponding latitude or longitude axis from
the WCS object would cause the translations to break.

To deal with this scenario, `~ndcube.NDCube` supports "missing" WCS
axes.  An additional attribute is added to the WCS object
(`NDCube.wcs.missing_axes`) which  is a list of `bool` type indicating
which WCS axes do not have a corresponding data axis.  This allows
translation information on coupled axes to persist even if the data
axes do not.  This feature also makes it possible for `~ndcube.NDCube`
to seamlessly reduce the data dimensionality via slicing.  In the
majority of cases a user will not need to worry about this feature.
But it is useful to be aware of as many of the coordinate
transformation functionalities of `~ndcube.NDCube` are only made
possible by the missing axis feature.

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
take another exposure. It would be very useful to have the measurement
times also associated with the x-axis.  However, the WCS may only
handle one translation per axis.

Fortunately, `~ndcube.NDCube` has a solution to this.  Values at
integer (pixel) steps along an axis can be stored within the object
and accessed via the `~ndcube.NDCube.extra_coords` property. To
attach extra coordinates to an `~ndcube.NDCube` instance, provide an
iterable of tuples of the form (`str`, `int`,
`~astropy.units.Quantity` or array-like) during instantiation.  The 0th
entry gives the name of the coordinate, the 1st entry gives the data
axis to which the extra coordinate corresponds, and the 2nd entry
gives the value of that coordinate at each pixel along the axis.  So
to add timestamps along the 0th axis of ``my_cube`` we do::

  >>> from datetime import datetime, timedelta
  >>> # Define our timestamps.  Must be same length as data axis.
  >>> axis_length = int(my_cube.dimensions[0].value)
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

  >>> my_cube.extra_coords # doctest: +SKIP
  {'time': {'axis': 0, 'value': [datetime.datetime(2000, 1, 1, 0, 0), datetime.datetime(2000, 1, 1, 0, 1), datetime.datetime(2000, 1, 1, 0, 2)]}}

Just like the data array and the WCS object, the extra coordinates are
sliced automatically when the `~ndcube.NDCube` instance is sliced.  So
if we take the first slice of ``my_cube`` in the 0th axis, the extra
time coordinate will only contain the value from that slice.::

  >>> my_cube[0].extra_coords # doctest: +SKIP
  {'time': {'axis': None, 'value': datetime.datetime(2000, 1, 1, 0, 0)}}

Note that the ``axis`` value is now ``None`` because the dimensionality of the
`~ndcube.NDCube` has been reduced via the slicing::

  >>> my_cube[0].dimensions
  <Quantity [4., 5.] pix>

and so the ``time`` extra coordinate no longer corresponds to a data
axis.  This would not have been the case if we had done the slicing
so the length of the 0th axis was >1::

  >>> my_cube[0:2].dimensions
  <Quantity [2., 4., 5.] pix>
  >>> my_cube[0:2].extra_coords # doctest: +SKIP
  {'time': {'value': [datetime.datetime(2000, 1, 1, 0, 0), datetime.datetime(2000, 1, 1, 0, 1)], 'axis': 0}}

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

In addition, some optional kwargs can be used to customize the
plot.  The ``axis_ranges`` kwarg can be used to set the axes ticklabels.  See the
`~sunpy.visualization.imageanimator.ImageAnimatorWCS` documentation for
more detail.  However, if this is not set, the axis ticklabels are
automatically derived in real world coordinates from the WCS object
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
`~ndcube.NDCube.pixel_to_world`, `~ndcube.NDCube.world_to_pixel`, and
`~ndcube.NDCube.axis_world_coords`. It is highly recommended that when
using `~ndcube.NDCube` these convenience wrappers are used rather than
the original astropy functions for a few reasons. For example, they
can track house-keeping data, are aware of "missing" WCS axis, are
unit-aware, etc.

To use `~ndcube.NDCube.pixel_to_world`, simply input
`~astropy.units.Quantity` objects with pixel units. Each
`~astropy.units.Quantity` corresponds to an axis so the number of
`~astropy.units.Quantity` objects should equal the number of data
axes.  Also, the order of the quantities should correspond to the
data axes' order, not the WCS order.  The nth element of each
`~astropy.units.Quantity` describes the pixel coordinate in that
axis. For example, if we wanted to transform the pixel coordinates of
the pixel (2, 3, 4) in ``my_cube`` we would do::

  >>> import astropy.units as u
  >>> real_world_coords = my_cube.pixel_to_world(2*u.pix, 3*u.pix, 4*u.pix)

To convert two pixels with pixel coordinates (2, 3, 4) and (5, 6, 7),
we would call pixel_to_world like so::

  >>> real_world_coords = my_cube.pixel_to_world([2, 5]*u.pix, [3, 6]*u.pix, [4, 7]*u.pix)

As can be seen, since each `~astropy.units.Quantity` describes a
different pixel coordinate of the same number of pixels, the lengths
of each `~astropy.units.Quantity` must be the same.

`~ndcube.NDCube.pixel_to_world` returns a similar list of Quantities
to those that were input, except that they are now in real world
coordinates::

  >>> real_world_coords
  [<Quantity [1.40006967, 2.6002542 ] deg>, <Quantity [1.49986193, 2.99724799] deg>, <Quantity [1.10e-09, 1.16e-09] m>]

The exact units used are defined within the `~ndcube.NDCube`
instance's `~ndcube.utils.wcs.WCS` object.  Once again, the coordinates
of the nth pixel is given by the nth element of each of the
`~astropy.units.Quantity` objects returned.

Using `~ndcube.NDCube.world_to_pixel` to convert real world
coordinates to pixel coordinates is exactly the same, but in reverse.
This time the input `~astropy.units.Quantity` objects must be in real
world coordinates compatible with those defined in the
`~ndcube.NDCube` instance's `~ndcube.utils.wcs.WCS` object.  The output
is a list of `~astropy.units.Quantity` objects in pixel unit.::

  >>> pixel_coords = my_cube.world_to_pixel(
  ... 1.400069678 * u.deg, 1.49986193 * u.deg, 1.10000000e-09 * u.m)
  >>> pixel_coords
  [<Quantity 2.00000003 pix>, <Quantity 3. pix>, <Quantity 4. pix>]

Note that both `~ndcube.NDCube.pixel_to_pixel` and
`~ndcube.NDCube.world_to_pixel` can handle non-integer pixels.
Moreover, they can also handle pixel beyond the bounds of the
`~ndcube.NDCube` and even negative pixels.  This is because the WCS
translations should be valid anywhere in space, and not just within
the field of view of the `~ndcube.NDCube`.  This capability has many
useful applications, for example, in comparing observations from
different instruments with overlapping fields of view.

There are times however, when you only want to know the real world
coordinates of the `~ndcube.NDCube` field of view.  To make this easy,
`~ndcube.NDCube` has a another coordinate transformation method
`~ndcube.NDCube.axis_world_coords`.  This method returns the real world
coordinates for each pixel along a given data axis.  So in the case of
``my_cube``, if we wanted the wavelength axis we could call::

  >>> my_cube.axis_world_coords(2)
  <Quantity [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09, 1.10e-09] m>

Note we set ``axes`` to ``2`` since ``axes`` is defined in data axis
order.  We can also define the axis using any unique substring
from the axis names defined in
`ndcube.NDCube.world_axis_physical_types`::

  >>> my_cube.world_axis_physical_types
  ('custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat', 'em.wl')
  >>> # Since 'wl' is unique to the wavelength axis name, let's use that.
  >>> my_cube.axis_world_coords('wl')
  <Quantity [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09, 1.10e-09] m>

Notice how this returns the same result as when we set ``axes`` to
the corresponding data axis number.

As discussed above, some WCS axes
are not independent.  For those axes,
`~ndcube.NDCube.axis_world_coords` returns a
`~astropy.units.Quantity` with the same number of dimensions as
dependent axes.  For example, helioprojective longitude and latitude
are dependent.  Therefore if we ask for longitude, we will get back a
2D `~astropy.units.Quantity` with the same shape as the longitude x
latitude axes lengths.  For example::

  >>> longitude = my_cube.axis_world_coords('lon')
  >>> my_cube.dimensions
  <Quantity [3., 4., 5.] pix>
  >>> longitude.shape
  (3, 4)
  >>> longitude
  <Quantity [[0.60002173, 0.59999127, 0.5999608 , 0.59993033],
             [1.        , 1.        , 1.        , 1.        ],
             [1.39997827, 1.40000873, 1.4000392 , 1.40006967]] deg>

It is also possible to request more than one axis's world coordinates
by setting ``axes`` to an iterable of data axis number and/or axis
type strings.::

  >>> my_cube.axis_world_coords(2, 'lon')
  (<Quantity [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09, 1.10e-09] m>,
   <Quantity [[0.60002173, 0.59999127, 0.5999608 , 0.59993033],
              [1.        , 1.        , 1.        , 1.        ],
              [1.39997827, 1.40000873, 1.4000392 , 1.40006967]] deg>)

Notice that the axes' coordinates have been returned in the same order
in which they were requested.

Finally, if the user wants the world
coordinates for all the axes, ``axes`` can be set to ``None``, which
is in fact the default.::

  >>> my_cube.axis_world_coords()
  (<Quantity [[0.60002173, 0.59999127, 0.5999608 , 0.59993033],
            [1.        , 1.        , 1.        , 1.        ],
            [1.39997827, 1.40000873, 1.4000392 , 1.40006967]] deg>,
   <Quantity [[1.26915033e-05, 4.99987815e-01, 9.99962939e-01,
               1.49986193e+00],
            [1.26918126e-05, 5.00000000e-01, 9.99987308e-01,
             1.49989848e+00],
            [1.26915033e-05, 4.99987815e-01, 9.99962939e-01,
             1.49986193e+00]] deg>,
   <Quantity [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09, 1.10e-09] m>)

By default `~ndcube.NDCube.axis_world_coords` returns the coordinates at the
center of each pixel. However, the pixel edges can be obtained by setting
the ``edges`` kwarg to True.

For example,
  >>> my_cube.axis_world_coords(edges=True)
  (<Quantity [[0.40006761, 0.40002193, 0.39997624, 0.39993054, 0.39988484],
            [0.80001604, 0.80000081, 0.79998558, 0.79997035, 0.79995511],
            [1.19998396, 1.19999919, 1.20001442, 1.20002965, 1.20004489],
            [1.59993239, 1.59997807, 1.60002376, 1.60006946, 1.60011516]] deg>,
   <Quantity [[-0.24994347,  0.24998788,  0.74995729,  1.24988864,
              1.74970582],
            [-0.24995565,  0.25000006,  0.74999384,  1.24994955,
              1.74979108],
            [-0.24995565,  0.25000006,  0.74999384,  1.24994955,
              1.74979108],
            [-0.24994347,  0.24998788,  0.74995729,  1.24988864,
              1.74970582]] deg>,
   <Quantity [1.01e-09, 1.03e-09, 1.05e-09, 1.07e-09, 1.09e-09, 1.11e-09] m>)

As stated previously, `~ndcube.NDCube` is only written
to handle single arrays described by single WCS instances.  For cases
where data is made up of multiple arrays, each described by different
WCS translations, `ndcube` has another class,
`~ndcube.NDCubeSequence`, which will discuss in the next section.
