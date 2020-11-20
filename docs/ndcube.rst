.. _ndcube:

======
NDCube
======

`~ndcube.NDCube` is the fundamental class of the ndcube package.  It is designed
to handle data contained in a single N-D array described by a single
set of WCS transformations.  `~ndcube.NDCube` is subclassed from
`astropy.nddata.NDData` and so inherits the same attributes for data,
wcs, uncertainty, mask, meta, and unit. Since v2.0, the ``wcs`` object must
adhere to astropy's APE 14 WCS API and ndcube leverages the WCS slicing
functionality provided by astropy, much of which was upstreamed from ndcube 1.0.

Initialization
==============

To initialize a basic `~ndcube.NDCube` object, all you need is a
`numpy.ndarray`-like array containing the data and an APE-14-compliant WCS object
(e.g. `astropy.wcs.WCS`) describing the coordinate transformation to and from
array-elements. Let's create a 3-D array of data with shape (3, 4, 5)
where every value is 1::

  >>> import numpy as np
  >>> data = np.ones((3, 4, 5))

Now let's create an `astropy.wcs.WCS` object describing the
translation from the array element coordinates to real world
coordinates.  Let the first data axis be helioprojective longitude,
the second be helioprojective latitude, and the third be wavelength.
Note that due to convention, the order of the axes in the
WCS object is reversed relative to the data array.::

  >>> import astropy.wcs
  >>> wcs_input_dict = {
  ... 'CTYPE1': 'WAVE    ', 'CUNIT1': 'Angstrom', 'CDELT1': 0.2, 'CRPIX1': 0, 'CRVAL1': 10, 'NAXIS1': 5,
  ... 'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5, 'NAXIS2': 4,
  ... 'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 3}
  >>> input_wcs = astropy.wcs.WCS(wcs_input_dict)

Now we can create an `~ndcube.NDCube`.::

  >>> from ndcube import NDCube
  >>> my_cube = NDCube(data, input_wcs)

The data array is stored in ``mycube.data`` while the
WCS object is stored in ``my_cube.wcs``.  However, when
manipulating/slicing the data is it better to slice the object as a
whole.  (See section on :ref:`ndcube_slicing`.)  So the ``.data`` attribute
should only be used to access specific raw data values.

Thanks to `~ndcube.NDCube`'s inheritance from `astropy.nddata.NDData`,
you can also supply additional data to the
`~ndcube.NDCube` instance.  These include: metadata located at `NDCube.meta`;
a mask (boolean array), located at `NDCube.mask`, marking reliable and unreliable pixels;
an uncertainty array located at `NDCube.uncertainty` (subclass of `astropy.nddata.NDUncertainty`) describing
the uncertainty of each data array value;
and a unit (`astropy.units.Unit` or unit `str`).::

  >>> from astropy.nddata import StdDevUncertainty
  >>> uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))
  >>> # Like numpy masked arrays, False means the data is unmasked.
  >>> mask = np.zeros_like(my_cube.data, dtype=bool)
  >>> meta = {"Description": "This is example NDCube metadata."}
  >>> my_cube = NDCube(data, input_wcs, uncertainty=uncertainty,
  ...                         mask=mask, meta=meta, unit=u.ct)

Dimensions
----------

`~ndcube.NDCube` has useful properties for inspecting its data shape and
axis types, `~ndcube.NDCube.dimensions` and `~ndcube.NDCube.array_axis_physical_types`::

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
represented by each array axis.  As more than one physical type can be associated
with an axis, the length of each tuple can be greater than 1.
This is the case for the 1st and 2nd array axes which are associated with
the coupled physical axes of latitude and longitude. The axis names are
in accordance with the International Virtual Observatory Alliance (IVOA)
`UCD1+ controlled vocabulary <http://www.ivoa.net/documents/REC/UCD/UCDlist-20070402.html>`_.

.. _ndcube_slicing:

Slicing
-------

Arguably NDCube's most powerful capability is its slicing.  Slicing an
`~ndcube.NDCube` instance using the standard slicing notation allows
users to access sub-regions of their data while simultaneously slicing
not only the other array attributes (e.g. uncertainty, mask, etc.) but
also the WCS object.  This ensures that even though the data array has
changed size and shape, each array element will still corresponds to
the same real world coordinates as they did before.
In addition to the WCS, `~ndcube.NDCube` slicing also alters the `~ndcube.ExtraCoords` object located at `ndcube.NDCube.my_extra_coords`.  See the :ref:`extra_coords` section.
An example of how to slice a 3-D `~ndcube.NDCube` object is::

  >>> my_cube_roi = my_cube[0:2, 1:4, 1:4]

Slicing can also reduce the dimensionality of an `~ndcube.NDCube`, e.g.::

  >>> my_2d_cube = my_cube[0, 1:4, 1:4]

In addition to slicing by index, `~ndcube.NDCube` supports a basic
version of slicing/indexing by real world coordinates via the
`~ndcube.NDCube.crop` method.  This takes two iterables of high level astropy objects
-- e.g. `~astropy.time.Time`, `~astropy.coordinates.SkyCoord`, `~astropy.coordinates.SpectralCoord`, `~astropy,units.Quantity` etc. --
which depend the physical types of the axes in the cube.  Each iterable describes a single
location in the data array in real world coordinates.  The first iterable
describes the lower corner of the region of interest and thus contains the lower limit
of all the real world coordinates.  The second iterable represents the upper corner
of the region of interest and thus contains the upper limit of all the real world coordinates.
The crop method indentifies the smallest rectangular region in the data array
that contains both the lower and upper limits in all the real world coordinates,
and crops the `~ndcube.NDCube` to that region. It does not rebin or interpolate the data.
The order of the high level coordinate objects in each iterable must be the same as
that expected by `astropy.wcs.WCS.world_to_array_index`, namely in world order.::

  >>> import astropy.units as u
  >>> from astropy.coordinates import SkyCoord, SpectralCoord
  >>> from sunpy.coordinates.frames import Helioprojective
  >>> wave_range = SpectralCoord([1.04e-9, 1.08e-9], unit=u.m)
  >>> sky_range = SkyCoord(Tx=[1, 1.5], Ty=[0.5, 1.5], unit=u.deg, frame=Helioprojective)
  >>> lower_corner = [wave_range[0], sky_range[0]]
  >>> upper_corner = [wave_range[-1], sky_range[-1]]
  >>> my_cube_roi = my_cube.crop(lower_corner, upper_corner)

And Much More
=============
`~ndcube.NDCube` provides many more helpful features, specifically regarding coordinate transformations and visualization.  See the :ref:plotting and :ref:coordinates sections.

.. _coordinates:

===========
Coordinates
===========
In the :ref:`ndcube` section we showed how `~ndcube.NDCube`'s slicing ensures the coordinate transformations remain consistent with the data as it is sliced.  In this section we will discuss the many other ways in which the ndcube classes support the integration of data and its coordinates.

.. _extra_coords:

ExtraCoords
===========
In the :ref:`ndcube` section we saw that the WCS object stored at `nducbe.NDCube.wcs` contains the primary set of coordinate transformations that describe the data.  However, what if you have alternative or additional coordinates that are not represented by the WCS?  The `ndcube.ExtraCoords` class provides users with a mechanism of attaching such coordinates to their `~ndcube.NDCube` instances.

Let's consider data from a slit spectrograph.  The first axis represents the position of the slit, the 2nd axis represents position along the slit.  Together these represent positions on the sky represented by coupled coordinates like latitude and longitude.  The final axis represents the spectral dimension.  The WCS nicely describes the relationship between the array axes and these 3 world coordinates.  However, if the spectrograph is in rastering mode -- i.e. moving the slit sequentially to build up and 2-D image of a region -- then the first axis is also associated with time.  This is because it takes time for the slit to be moved, a measurement taken and then the slit moved again.  If the time dimension is not captured by the WCS, then users can represent it as an extra coordinate.  Coordinates can be compiled as lookup tables, gathered in an `~ndcube.ExtraCoords` object and attached to an `~ndcube.NDCube` during instantiation.::

  >>> from astropy.time import Time, TimeDelta
  >>> from ndcube import ExtraCoords
  >>> # Define Time object giving the times along the 1st axis.
  >>> time_axis = 0
  >>> time_axis_length = int(my_cube.dimensions[axis].value)
  >>> base_time = Time('2000-01-01', format='fits', scale='utc')
  >>> timestamps = Time(base_time + TimeDelta(60 * i, format='sec') for i in range(time_axis_length))
  >>> # Construct an empty ExtraCoords object hen add the time coordinate.
  >>> my_extra_coords = ExtraCoords()
  >>> # To add a coordinate, supply its name, array axes, and values at each arrya element.
  >>> my_extra_coords.add_coordinate('time', (0,), timestamps)
  >>> my_cube = NDCube(data, input_wcs, uncertainty=uncertainty, mask=mask,
  ...                  meta=meta, unit=u.ct, extra_coords=my_extra_coords)
  
Combined WCS
------------
Extra coordinates can be be combined with the primary WCS via `ndcube.NDCube.combined_wcs`.  As this is a fully valid WCS describing the data, it can be used just like `ndcube.NDCube.wcs` in coordinate transformations and visualizations.

.. _global_coords:

GlobalCoords
============

Sometimes coordinate are not associated with any axis.  Take the case of a 2-D `~ndcube.NDCube` representing a single image.  The time at which that image was taken is important piece of coordinate information.  But because it is not associated with either the x and y axes, it cannot be stored in the WCS or `~ndcube.ExtraCoords` objects.  Storing such coordinates in the role of the `ndcube.GlobalCoords` class.  Let' assume that ``my_cube`` was taken by a pixelated spectroscopic detector that measures position and wavelength simultaneously.  ``my_cube`` thus represents a single 3-D frame measurement taken at a given time.  Now let's attach the frame time using `~ndcube.GlobalCoords`.::

  >>> from astropy.time import Time
  >>> from ndcube import GlobalCoords
  >>> # Generate a fresh version of my_cube.
  >>> my_cube = NDCube(data, input_wcs, uncertainty=uncertainty, mask=mask,
  ...                  meta=meta, unit=u.ct)
  >>> # To add a global coord, provide a name, physical type, and value.
  >>> my_cube.global_coords.add('time', 'time', Time('2000-01-01', format='fits', scale='utc'))
  
`~ndcube.GlobalCoords` allows multiple coordinates of the same physical type.  Therefore when adding a global coordinate, you must provide a unique coordinate name, its physical time and the coordinate value.  The value of the coordinate can be accessed by indexing the `~ndcube.GlobalCoords` instance with the coordinate name::

  >>> my_cube.global_coords['time']
  <Time format='fits', scale='utc', value='2000-01-01T00:00:00.000'>

The coordinate's physical type can be accessed via the `~ndcube.GlobalCoords.physical_types` `dict` property::

  >>> my_cube.global_coords.physical_types['time']
  
Because `~ndcube.GlobalCoords` inherits from `Mapping`, it contains a number of mixin methods similar to those of `dict`.::

  >>> list(my_cube.global_coords.keys())  # Returns a list of global coordinate names
  ['time']
  >>> list(my_cube.global_coords.values()  # Returns a list of coordinate values
  [<Time format='fits', scale='utc', value='2000-01-01T00:00:00.000'>]
  >>> list(my_cube.global_coords.items())  # Returns a list of (name, value) pairs
  [('time', <Time format='fits', scale='utc', value='2000-01-01T00:00:00.000'>)]
  
One of the most common use cases for `~ndcube.GlobalCoords` is slicing.  In addition to tracking and updating the `~ndcube.NDCube.wcs` and `~ndcube.NDCube.extra_coords` objects, `~ndcube.NDCube`'s slicing infrastucture also identifies when an axis has been dropped and remembers the value of any independent coordinates at the location along that axis at which the cube was sliced.  Let's demonstrate this by slicing away the wavelength axis of ``my_cube``.::

  >>> my_2d_cube = my_cube[:, :, 0]
  >>> my_2d_cube.array_axis_physical_types  # Note the wavelength axis is now gone.
  [('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
   ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon')]
  >>> # The wavelength value at the slicing location is now in the GLobalCoords object.
  >>> list(my_2d_cube.global_coords.keys())
  ['time', 'em.wl']
  >>> my_2d_cube.global_coords.physical_types['em.wl']
  'em.wl'
  >>> my_2d_cube.global_coords['em.wl']
  <SpectralCoord 1e-9 m>

.. _coordinate_transformations:

Coordinate Transformations
==========================

Axis World Coordinates
----------------------

WCS objects are a powerful and concise way of storing complex functional coordinate transformations.  However, their API be cumbersome when the coordinates along a whole axis is desired.  Making this process easy and intuitive is the purpose of `ndcube.NDCube.axis_world_coords`.  Using the information on the data dimensions and optional inputs from the user, this method returns high level coordinate objects -- e.g. `~astropy.coordinates.SKyCoord`, `~astropy.time.Time`, `~astropy.coordinates.SpectralCoord`, `~astropy.units.Quantity` -- containing the coordinates at each array element.  Let's say we wanted the wavelength values along the spectral axis of ``my_cube``.  We can do this in a couple ways.  First we can provide `~ndcube.NDCube.axis_world_coords` with the array axis number of the spectral axis.

  >>> my_cube.axis_world_coords(2)
  <SpectralCoord [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09, 1.10e-09] m>

Alternatively we can provide a unique substring of the physical type of the coordinate, stored in `ndcube.NDCube.wcs.world_axis_physical_types`::

  >>> my_cube.wcs.world_axis_physical_types
  ['em.wl', 'custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon']
  >>> # Since 'wl' is unique to the wavelength axis name, let's use that.
  >>> my_cube.axis_world_coords('wl')
  <SpectralCoord [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09, 1.10e-09] m>

As discussed above, some WCS axes are not independent.  For those axes, `~ndcube.NDCube.axis_world_coords` returns objects with the same number of dimensions as dependent axes.  For example, helioprojective longitude and latitude are dependent.  Therefore if we ask for longitude, we will get back a `~astropy.coordinates.SkyCoord` with containing 2-D latitude and longitude arrays with the same shape as the array axes to which they correspond.  For example::

  >>> celestial = my_cube.axis_world_coords('lon')
  >>> my_cube.dimensions
  <Quantity [3., 4., 5.] pix>
  >>> celestial.shape
  (3, 4)
  >>> celestial
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

  >>> my_cube.axis_world_coords(2, 'lon')
  (<SpectralCoord [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09, 1.10e-09] m>,
   <SkyCoord (Helioprojective: obstime=None, rsun=695700.0 km, observer=earth): (Tx, Ty) in arcsec
        [[(2160.07821927, 4.56894119e-02), (2159.96856373, 1.79995614e+03),
          (2159.85889149, 3.59986658e+03), (2159.74920255, 5.39950295e+03)],
         [(3600.        , 4.56905253e-02), (3600.        , 1.80000000e+03),
          (3600.        , 3.59995431e+03), (3600.        , 5.39963453e+03)],
         [(5039.92178073, 4.56894119e-02), (5040.03143627, 1.79995614e+03),
          (5040.14110851, 3.59986658e+03), (5040.25079745, 5.39950295e+03)]]>
  )


The coordinate objects are returned in world axis order in accordance with APE 14.

Finally, if the user wants the world coordinates for all the axes, ``axes`` can be set to ``None``, which
is in fact the default.::

  >>> my_cube.axis_world_coords()
  (<SpectralCoord [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09, 1.10e-09] m>,
   <SkyCoord (Helioprojective: obstime=None, rsun=695700.0 km, observer=earth): (Tx, Ty) in arcsec
        [[(2160.07821927, 4.56894119e-02), (2159.96856373, 1.79995614e+03),
          (2159.85889149, 3.59986658e+03), (2159.74920255, 5.39950295e+03)],
         [(3600.        , 4.56905253e-02), (3600.        , 1.80000000e+03),
          (3600.        , 3.59995431e+03), (3600.        , 5.39963453e+03)],
         [(5039.92178073, 4.56894119e-02), (5040.03143627, 1.79995614e+03),
          (5040.14110851, 3.59986658e+03), (5040.25079745, 5.39950295e+03)]]>
  )


By default `~ndcube.NDCube.axis_world_coords` returns the coordinates at the
center of each pixel. However, the pixel edges can be obtained by setting
the ``edges`` kwarg to ``True``. For example::

  >>> my_cube.axis_world_coords(edges=True)
  (<SpectralCoord [1.01e-09, 1.03e-09, 1.05e-09, 1.07e-09, 1.09e-09, 1.11e-09] m>,
   <SkyCoord (Helioprojective: obstime=None, rsun=695700.0 km, observer=earth): (Tx, Ty) in arcsec
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
          (5760.41458147, 6298.94094507)]]>
  )

Working with Raw Coordinates
............................

As shown, `~ndcube.NDCube.axis_world_coords` returns high level coordinate objects.  However, users and developers also have the option to use `ndcube.NDCube.axis_world_coords_values`.  The API for this method is exactly the same as `~ndcube.NDCube.axis_world_coords`.  However, it returns `~astropy.units.Quantity` objects for each physical type.  These are returned in the same order as `~astropy.wcs.WCS.world_axis_physical_types`.  These objects do not contain important contextual information, such as reference frame, which is needed to fully interpret the coordinate values.  However for some use cases this level of completeness is not needed.::

  >>> my_cube.axis_world_coords_values()

.. _plotting:

========
Plotting
========

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
