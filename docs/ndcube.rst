======
NDCube
======

`NDCube` is the fundamental class of the ndcube package and is designed
to handle array contained in a single N-D array described by a single
WCS transformation.  `NDCube` is subclassed from `astropy.nddata.NDData`
and so inherits the same attributes for data, wcs, uncertainty, mask,
meta, and unit.  The WCS object contained in the .wcs attribute is
subclassed from `astropy.wcs.WCS` and contains a few additional
attributes to enable to keep track of its relationship to the data.

Initialization
----------------

To initialize the most basic `NDCube` object, all you need is a
`numpy.ndarray` containing the data, and an `astropy.wcs.WCS` object
describing the transformation from array-element space to real world
coordinates.  Let data be the array and wcs be the WCS object.  Then
you can create an `NDCube` by doing::

  import ndcube
  my_cube = ndcube.NDCube(data, wcs)

The data array is stored in the mycube.data attribute while the WCS
object is stored in the my_cube.wcs attribute.  However, when
manipulating/slicing the data is it better to slice the object as a
whole.  (See section on Slicing.)  So the .data attribute should only
be accessed to access a specific value(s) in the data.  Another thing
to note is that as part of the initialization, the wcs object is
converted from an `astropy.wcs.WCS` to an `ndcube.wcs_util.WCS` object
which has some additional features for tracking "missing axes", etc.
(See section on Missing Axes.)

Thanks to the fact that `NDCube` is subclassed from
`astropy.nddata.NDData`, you can also supply additional data to the
`NDCube` instance.  These include: metadata (`dict` or
dict-like); a data mask (boolean `numpy.ndarray`) highlighting, for
example, reliable and unreliable pixels; an uncertainty array
(`numpy.ndarray`) describing the uncertainty of each data array value;
and a unit (`astropy.units.Unit` or unit `str`).
For example::

  my_cube = ndcube.NDCube(data, wcs, uncertainty=uncertainty,
  mask=mask, meta=meta, unit=None)

N.B. Following the unfortunately confusing convention, the order of
the axes in the WCS object are reversed compared to the data.

Slicing
--------

Arguably NDCube's most powerful capability is its slicing.  Slicing an
NDCube object using the standard slicing notation allows users to
access sub-regions of their data while simultaneously slicing not only
the other array attributes (e.g. uncertainty, mask, etc.) but also the
WCS object.  This ensures that even though the data array has changed
size and shape, each array element will still corresponding to the
same real world coordinates as they did before.  An example of how to
slice a 3-D `NDCube` object is::

  my_cube_roi = my_cube[3:5, 10:100, 30:37]

Slicing can also reduce the dimension of an `NDCube`, e.g.::

  my_2d_cube = my_cube[0, 10:100, 30:37]

In addition to slicing by index, `NDCube` supports a basic version of
slicing/index by real world coordinate via the `NDCube.crop_by_coords`
method.  This takes a list of `astropy.units.Quantity` representing
the real world coordinates of the lower left corner of the region of
interest.  The order of the coordinates must be the same as the order
of the data axes.  A second iterable of `astropy.units.Quantity` must
also be provided which gives the widths of the region of interest in
each data axis::

  from astropy.units import Quantity
  my_cube_roi = my_cube.crop_by_coords(
      [Quantity(30, unit="arcsec"), Quantity(10, unit="arcsec"),
      Quantity(1e-7, unit="m")],
      [Quantity(20, unit="arcsec"), Quantity(20, unit="arcsec"),
      Quantity(8e-7, unit="m")])

This method does not rebin or interpolate the data if the region of interest
defined does not perfectly map onto the array's "pixel" grid.  Instead
it translates from real world to pixel coordinates and rounds to the
nearest integer before indexing/slicing the `NDCube` object.
Therefore it should be noted that slightly different inputs to this
method can result in the same output.

Dimensions
-----------

NDCube has a useful property for inspecting its data shape and
axis types, `NDCube.dimensions`::

  my_cube.dimensions

This returns a named tuple with a "shape" and "axis_types" attribute.
"shape" is an `astropy.units.Quantity` of pixel units giving the
length of each dimension in the `NDCube`.  Meanwhile, "axis_types" is
`list` of `str` giving the WCS transformation type for each axis.
Here the shape and axis types are given in data order, not WCS order.

As the dimensions property returns a named tuple, the shape and axis
types can be accessed directly::

  my_cube.dimensions.shape
  my_cube.dimensions.axis_types


Missing Axes
-------------

Some WCS axis types are coupled.  For example, the helioprojected
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

To deal with this scenario, `NDCube` supports "missing" WCS axes.  An
additional attribute is added to the object (NDCube.wcs.missing_axis) which
is a list of `bool` type indicating which WCS axes do not have a
corresponding data axis.  This allows translation information on
coupled axes to persist even if the data axes do not.  This feature
makes in possible for `NDCube` to seamlessly reduce the data
dimensionality via slicing and also handle data types with only one
spatial dimension, like those from a slit-spectrograph instrument
which would have otherwise been impossible.  In the majority of cases
a user will not need to worry about this feature.  But it is useful to
be aware of as many of the coordinate transformation functionalities
of `NDCube` are only made possible by the missing axis feature.

Extra Coordinates
-------------------

In the case of some datasets, there may be additional translations
that between the array elements and real world coordinates that are
not included in the WCS.  Consider a 3-D data cube from a rastering
slit-spectrograph instrument.  The first axis corresponds to the
x-position of the slit as it steps across a region of interest in a
set pattern.  The second corresponds to latitude along the slit.  And
the third axis corresponds to wavelength.  However, first axis also
corresponds to time, as it takes time for the slit to move and then
take another exposure which results in a new spectrogram (y-position
vs. wavelength). It would be very useful to have the time of each
position in the x-axis associated with the time at which the exposure
was taken, but the WCS can only handle one translation per axis.

Fortunately, `NDCube` has a solution to this.  Values at integer steps
along an axis can be stored within the object and accessed via the
`NDCube._extra_coords()` property.  This property is currently
"private" but can be made public in any subclass of NDCube.  The
_extra_coords() property returns a dictionary of dictionaries.  Each
sub-dictionary corresponds to an extra coordinate, e.g. time, and
gives the value of number of the data axis to which it corresponds as
well as the value of that coordinate at each data array element::

  my_cube._extra_coords

Just like the data array and the WCS object, the extra coordinates are
sliced automatically when the `NDCube` object is sliced.

To attach extra coordinates to an `NDCube` instance, use the
extra_coords kwarg during initialization::

  my_cube = ndcube.NDCube(data, wcs, extra_coords=extra_coords_input)

where extra_coords_input is an iterable of tuple of types (`str`, `int`,
`astropy.units.Quantity`).  Each tuple corresponds to an extra
coordinate and gives the name, data axis, and values of the
coordinate.  The third element of the tuple must be of the same length
as data axis to which it is assigned.

Plotting
---------

To quickly and easily visualize N-D data, `NDCube` provides a
simple-to-use, yet powerful plotting method, `NDCube.plot`, which
produces a sensible visualization based on the dimensionality of the
data within the `NDCube` object.  It is intended to be a useful
quicklook tool and not a replacement for high quality plots or
animations, e.g. for publications.  The plot method can be called very
simply, like so::

  my_cube.plot()

The type of visualization returned depends on the dimensionality of
the dat within the `NDCube` object.  For 1-D data a line plot is
produced, similar to `matplotlib.pyplot.plot`.  For 2-D data, an image
is produced similar to that of `matplotlib.pyplot.imshow`.  While for
a >2-D data, a `sunpy.visualization.imageanimator.ImageAnimatorWCS`
object is returned.  This displays a 2-D imaged with sliders for each
additional dimension which allow the user to animate through the
different values of each dimension and see the effect in the 2-D
image.

No args are required.  The necessary information to generate the plot
are derived from the data and metadata in the `NDCube` itself.
Setting the x and y ranges of the plot can be done simply by indexing
the `NDCube` object itself to the desired region of interest and then
calling the plot method, e.g.::

  my_cube[0, 10:100, :].plot()

In addition to this, some optional kwargs can be used to customize the
plot.  The axis_ranges kwarg can be used to set the axes ticklabels.  See the
`sunpy.visualization.imageanimator.ImageAnimatorWCS` documentation for
more detail.  However, if this is not set, the axis ticklabels are
automatically derived in real world coordination from the WCS obect
within the `NDCube`.

By default the final two data dimensions are used for the plot
axes in 2-D or greater visualizations, but this can be set by the user
using the images_axes kwarg::

  my_cube.plot(image_axes=[0,1])

where the first entry in the list gives the index of the data index to
go on the x-axis, and the second entry gives the index of the data
index to go on the y-axis.

In addition, the units of the axes or the data can be set by the
unit_x_axis, unit_y_axis, unit kwargs.  However, if not set, these are
derived from the `NDCube` wcs and unit attributes.

Coordinate Transformations
----------------------------

The fundamental point the WCS system is the ability to easily
translate between pixel and real world coordinates.  For this purpose, 
`NDCube` provides convenience wrappers for the better known astropy
functions, `astropy.wcs.WCS.all_pix2world` and
`astropy.wcs.WCS.all_world2pix`. These are `NDCube.pixel_to_world` and
`NDCube.world_to_pixel`.  It is highly recommended
that when using `NDCube` these convenience wrappers are used rather
than the original astropy functions for a few reasons.  For example,
they can track house-keeping data, are aware of "missing" WCS axis,
are unit-aware, etc.

To use the pixel_to_world method, simply input a list of
`astropy.units.Quantity` objects with pixel units. Each quantity
corresponds to an axis so the number of Quantity objects should equal
the number of data axes.  Also, the order of the quantities should 
correspond to the data axes' order, not the WCS order.  The nth
element of each Quantity describes the pixel coordinate in each axis
of the nth pixel to be transformed.  For example, in a 3-D data set,
if we wanted to transform the pixel coordinates of the pixel (2, 3, 4),
We would enter a list call pixel_to_world in the following way::

   import astropy.units as u
   real_world_coords = my_cube.pixel_to_world(
       [u.Quantity([2], unit=u.pix), u.Quantity([3], unit=u.pix),
       u.Quantity([4], unit=u.pix)])

To convert two pixels with pixel coordinates (2, 3, 4) and (5, 6, 7),
we would call pixel_to_world like so::

  real_world_coords = my_cube.pixel_to_world(
       [u.Quantity([2, 5], unit=u.pix), u.Quantity([3, 6], unit=u.pix),
       u.Quantity([4, 7], unit=u.pix)])

As can be seen, since each Quantity describes a different pixel
coordinate of the same number of pixels, the lengths of each Quantity
must be the same.

pixel_to_world returns a similar list of Quantities as to those that
were input, except that they are now in real world coordinates.  The
exact units used are defined within the `NDCube` instance's WCS
object.  Once again, the coordinates of the nth pixel is given by the
nth elements from each of the Quantities returned.

Using world_to_pixel to convert real world coordinates to pixel
coordinates is exactly the same, but in reverse.  This time the input
Quantities must be in real world coordinates compatible with those
defined in the `NDCube` instance's WCS object and a list of Quantities
in pixel units is returned.

Both `NDCube.pixel_to_world` and `NDCube.world_to_pixel` have an
additional optional kwarg, origin, whose default is 0.  This is the
same as the origin arg in `astropy.wcs.WCS.all_pix2world` and
`astropy.wcs.WCS.all_world2pix` and defines whether the WCS
translation is 0-based (C) or 1-based (FORTRAN).  Changing this kwarg
will result in the pixel coordinates being offset by 1.  In most
cases, the approriate setting will be origin=0, but 1-based may be
required for writing the WCS translations to a FITS header.
