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

NDCube Coordinate Transformations
=================================
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

NDCubeSequence Coordinate Transformations
=========================================

Sequence Axis Coordinates
-------------------------
As described in the :ref:`ndcubesequence` section, the sequence axis can be thought of as an additional array axis perpendicular to those in the `~ndcube.NDCubes` within and `~ndcube.NDCubeSequence`.  In that model, the `~ndcube.GlobalCoords` on each `~ndcube.NDCube` represent coordinate values along the sequence axis.  The `ndcube.NDCubeSequence.sequence_axis_coords` property collates a list for each global coordinate with each array element giving the coordinate value from the corresponding `~ndcube.NDCube`.  The lists are returned in a `dict` with keys giving my the global coordinate names.::

  >>> my_sequence.sequence_axis_coords
  ???
  
As with any `dict`, the coordinate names can be seen via the ``.keys()`` method, while the values of a coordinate can be retrieved by indexing with the coordinate name.::

  >>> my_sequence.sequence_axis_coords.keys()
  ['time']
  >>> my_sequence.sequence_axis_coords['time']
  <Time ???>

Common Axis Coordinates
-----------------------
The :ref:`ndcubesequence` section also explains how a common axis can be defined for a `~ndcube.NDCubeSequence`, signifying that the sequence axis is in fact parallel to one of the `~ndcube.NDCube` array axes and that the cubes can be thought of as arranged sequentially along that axis.  In this model, coordinates for the last index along the common axis in one cube should be followed by the coordinate value of the first array index in the next cube.  The `ndcube.NDCubeSequence.common_axis_coords` property finds the physical types associated with the common axis in each cube and concatenates them.::

  >>> my_sequence.common_axis_coords
  ???
