.. doctest-skip-all::
    all

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
`~ndcube.NDCube.array_axis_physical_types`::

  >>> my_cube.dimensions
  <Quantity [3., 4., 5.] pix>
  >>> my_cube.array_axis_physical_types
  [('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
   ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
   ('em.wl',)]

`~ndcube.NDCube.dimensions` returns an `~astropy.units.Quantity` of
pixel units giving the length of each dimension in the
`~ndcube.NDCube` while `~ndcube.NDCube.array_axis_physical_types`
returns tuples of strings denoting the types of physical properties
represented by each axis.  As more than one physical type can be associated
with an axis, the length of each tuple can be greater than 1.
This is the case of the 0th and 1st array axes which are associated with
the coupled physical axes of latitude and longitude. The axis names are
in accordance with the International Virtual Observatory Alliance (IVOA)
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

  >>> my_cube_roi = my_cube[0:2, 1:4, 1:4]

Slicing can also reduce the dimension of an `~ndcube.NDCube`, e.g.::

  >>> my_2d_cube = my_cube[0, 1:4, 1:4]

In addition to slicing by index, `~ndcube.NDCube` supports a basic
version of slicing/indexing by real world coordinates via the
`~ndcube.NDCube.crop` method.  This takes a list of high level astropy objects,
e.g. `~astropy.time.Time`, `~astropy.coordinates.SkyCoord`,
`~astropy.coordinates.SpectralCoord`, `~astropy,units.Quantity`, etc., which depend
the physical types of the axes in the cube.  Each high level object
represents the minimum and maximum real world coordinates of the region of interest
in each dimension.  The order of the coordinates must be the same as that expected by
`astropy.wcs.WCS.world_to_array_index`.::

  >>> import astropy.units as u
  >>> from astropy.coordinates import SkyCoord, SpectralCoord
  >>> from sunpy.coordinates.frames import Helioprojective
  >>> wave_range = SpectralCoord([1.04e-9, 1.08e-9], unit=u.m)
  >>> sky_range = SkyCoord(Tx=[1, 1.5], Ty=[0.5, 1.5], unit=u.deg, frame=Helioprojective)
  >>> lower_corner = [wave_range[0], sky_range[0]]
  >>> upper_corner = [wave_range[-1], sky_range[-1]]
  >>> my_cube_roi = my_cube.crop(lower_corner, upper_corner)

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
`~ndcube.NDCube` provides a convenience function for returning the real
world coordinates of each pixel/array element of the data cube,
`~ndcube.NDCube.axis_world_coords`.  So in the case of
``my_cube``, if we wanted the wavelength axis we could call::

  >>> my_cube.axis_world_coords(2) # doctest: +SKIP
  <Quantity [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09, 1.10e-09] m>

Note we set ``axes`` to ``2`` since ``axes`` is defined in data axis
order.  We can also define the axis using any unique substring

`ndcube.NDCube.wcs.world_axis_physical_types`::

  >>> my_cube.wcs.world_axis_physical_types
  ['em.wl', 'custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon']
  >>> # Since 'wl' is unique to the wavelength axis name, let's use that.
  >>> my_cube.axis_world_coords('wl') # doctest: +SKIP
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

  >>> longitude = my_cube.axis_world_coords('lon') # doctest: +SKIP
  >>> my_cube.dimensions
  <Quantity [3., 4., 5.] pix>
  >>> longitude.shape # doctest: +SKIP
  (3, 4)
  >>> longitude # doctest: +SKIP
  <SkyCoord (Helioprojective: obstime=None, rsun=695700.0 km, observer=earth): (Tx, Ty) in arcsec
        [[(2160.07821927, 4.56894119e-02), (2159.96856373, 1.79995614e+03),
          (2159.85889149, 3.59986658e+03), (2159.74920255, 5.39950295e+03)],
         [(3600.        , 4.56905253e-02), (3600.        , 1.80000000e+03),
          (3600.        , 3.59995431e+03), (3600.        , 5.39963453e+03)],
         [(5039.92178073, 4.56894119e-02), (5040.03143627, 1.79995614e+03),
          (5040.14110851, 3.59986658e+03), (5040.25079745, 5.39950295e+03)]]>

It is also possible to request more than one axis's world coordinates
by setting ``axes`` to an iterable of data axis number and/or axis
type strings.::

  >>> my_cube.axis_world_coords(2, 'lon') # doctest: +SKIP
  (<SkyCoord (Helioprojective: obstime=None, rsun=695700.0 km, observer=earth): (Tx, Ty) in arcsec
        [[(2160.07821927, 4.56894119e-02), (2159.96856373, 1.79995614e+03),
          (2159.85889149, 3.59986658e+03), (2159.74920255, 5.39950295e+03)],
         [(3600.        , 4.56905253e-02), (3600.        , 1.80000000e+03),
          (3600.        , 3.59995431e+03), (3600.        , 5.39963453e+03)],
         [(5039.92178073, 4.56894119e-02), (5040.03143627, 1.79995614e+03),
          (5040.14110851, 3.59986658e+03), (5040.25079745, 5.39950295e+03)]]>,
    <Quantity [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09, 1.10e-09] m>)


Notice that the axes' coordinates have been returned in the same order
in which they were requested.

Finally, if the user wants the world
coordinates for all the axes, ``axes`` can be set to ``None``, which
is in fact the default.::

  >>> my_cube.axis_world_coords() # doctest: +SKIP
  (<SkyCoord (Helioprojective: obstime=None, rsun=695700.0 km, observer=earth): (Tx, Ty) in arcsec
        [[(2160.07821927, 4.56894119e-02), (2159.96856373, 1.79995614e+03),
          (2159.85889149, 3.59986658e+03), (2159.74920255, 5.39950295e+03)],
         [(3600.        , 4.56905253e-02), (3600.        , 1.80000000e+03),
          (3600.        , 3.59995431e+03), (3600.        , 5.39963453e+03)],
         [(5039.92178073, 4.56894119e-02), (5040.03143627, 1.79995614e+03),
          (5040.14110851, 3.59986658e+03), (5040.25079745, 5.39950295e+03)]]>,
    <Quantity [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09, 1.10e-09] m>)


By default `~ndcube.NDCube.axis_world_coords` returns the coordinates at the
center of each pixel. However, the pixel edges can be obtained by setting
the ``edges`` kwarg to True.

For example::

  >>> my_cube.axis_world_coords(edges=True) # doctest: +SKIP
  (<SkyCoord (Helioprojective: obstime=None, rsun=695700.0 km, observer=earth): (Tx, Ty) in arcsec
        [[(1440.24341188, -899.79647591), (1440.07895112,  899.95636786),
          (1439.91446531, 2699.84625127), (1439.74995445, 4499.59909505),
          (1439.58541853, 6298.94094507)],
         [(2880.05774973, -899.84032206), (2880.00292413,  900.00022848),
          (2879.94809018, 2699.97783871), (2879.89324788, 4499.81838925),
          (2879.83839723, 6299.24788597)],
         [(4319.94225027, -899.84032206), (4319.99707587,  900.00022848),
          (4320.05190982, 2699.97783871), (4320.10675212, 4499.81838925),
          (4320.16160277, 6299.24788597)],
         [(5759.75658812, -899.79647591), (5759.92104888,  899.95636786),
          (5760.08553469, 2699.84625127), (5760.25004555, 4499.59909505),
          (5760.41458147, 6298.94094507)]]>,
    <Quantity [1.01e-09, 1.03e-09, 1.05e-09, 1.07e-09, 1.09e-09, 1.11e-09] m>)


As stated previously, `~ndcube.NDCube` is only written
to handle single arrays described by single WCS instances.  For cases
where data is made up of multiple arrays, each described by different
WCS translations, `ndcube` has another class,
`~ndcube.NDCubeSequence`, which will discuss in the next section.
