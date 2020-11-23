===========
Coordinates
===========
In the :ref:`ndcube` section we showed how `~ndcube.NDCube`'s slicing ensures the coordinate transformations remain consistent with the data as it is sliced.  In this section we will discuss the many other ways in which the ndcube classes support the integration of data and its coordinates.

But first let's recreate the data and WCS components of an `~ndcube.NDCube` for use in our demonstration.

.. code-block:: python

  >>> import numpy as np
  >>> import astropy.wcs
  >>> data = np.ones((3, 4, 5))
  >>> wcs_input_dict = {
  ... 'CTYPE1': 'WAVE    ', 'CUNIT1': 'Angstrom', 'CDELT1': 0.2, 'CRPIX1': 0, 'CRVAL1': 10, 'NAXIS1': 5,
  ... 'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5, 'NAXIS2': 4,
  ... 'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 3}
  >>> input_wcs = astropy.wcs.WCS(wcs_input_dict)

.. _extra_coords:

ExtraCoords
===========
In the :ref:`ndcube` section we saw that the WCS object stored at `nducbe.NDCube.wcs` contains the primary set of coordinate transformations that describe the data.  However, what if you have alternative or additional coordinates that are not represented by the WCS?  The `ndcube.ExtraCoords` class provides users with a mechanism of attaching such coordinates to their `~ndcube.NDCube` instances.  Let's start by creating a `~astropy.time.Time` object representing the times corresponding to each position along the 1st axis of the data array defined above.

.. code-block:: python

  >>> from astropy.time import Time, TimeDelta
  >>> base_time = Time('2000-01-01', format='fits', scale='utc')
  >>> timestamps = Time([base_time + TimeDelta(60 * i, format='sec') for i in range(data.shape[0])])

Now let's create an `~ndcube.ExtraCoords` instance and add our time extra coordinate to it.  To do this we need to supply the physical type of the coordinate, the array axis to which is corresponds, and the values of the coordinate.  The number of values should equal the length of the axis.

.. code-block:: python

  >>> from ndcube import ExtraCoords
  >>> my_extra_coords = ExtraCoords()
  >>> my_extra_coords.add_coordinate('time', (2,), timestamps)  # TO DO: Change the mapping to 0 when bug fixed.

An indefinite number of coordinates can be added in this way.  Alternatively, we can generate an `~ndcube.ExtraCoords` object from a WCS.  The names of the coordinates can be accessed via the `~ndcube.ExtraCoords.keys` method.

.. code-block:: python

  >>> my_extra_coords.keys()
  ('time',)

The values of the coordinates can be accessed via `~ndcube.ExtraCoords.wcs`.  This property generates a WCS object based on the coordinates stored in the `~ndcube.ExtraCoords` object.  It should be noted that because `~ndcube.ExtraCoords` is a stand alone object, its WCS does not necessarily match the data array of an `~ndcube.NDCube`.  To convert the extra coordinates into a WCS directly applicable to an `~ndcube.NDCube`, see the :ref:`combined_wcs` section below on `~ndcube.NDCube.combined_wcs`.

Finally, the `~ndcube.ExtraCoords` object can be attached to an `~ndcube.NDCube`  during instantiation and access via the `~ndcube.NDCube.extra_coords` property.

.. code-block:: python

  >>> from ndcube import NDCube
  >>> my_cube = NDCube(data, input_wcs, extra_coords=my_extra_coords)

If extra coordinates are present, their physical types are revealed by `~ndcube.NDCube.array_axis_physical_types`.

.. code-block:: python

  >>> my_cube.array_axis_physical_types
  [('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon', 'time'), ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'), ('em.wl',)]

.. _combined_wcs::

Combined WCS
------------
The `~ndcube.NDCube.combined_wcs` generates a WCS that combines the extra coords with those  stored in the primary WCS.  Unlike `ndcube.ExtraCoords.wcs`, `~ndcube.NDCube.combined_wcs` is a valid WCS for describing the `~ndcube.NDCube` data array and so can be used with the `~ndcube.NDCube` coordinate transformation and plotting features.

.. _global_coords:

GlobalCoords
============
Sometimes coordinates are not associated with any axis.  Take the case of a 2-D `~ndcube.NDCube` representing a single image.  The time at which that image was taken is important piece of coordinate information.  But because the data does not have a 3rd dimension, it cannot be stored in the WCS or `~ndcube.ExtraCoords` objects.  Storing such coordinates is the role of the `ndcube.GlobalCoords` class.  `~ndcube.NDCube` is instatiated with an empty `~ndcube.GlobalCoords` object already attached at `ndcube.NDCube.global_coords`.  Let's attach a scalar global coordinate to ``my_cube`` representing some kind of distance.  We do by supplying the coordinate's name, physical type and value via the `~ndcube.GlobalCoords.add` method.

.. code-block:: python

  >>> import astropy.units as u
  >>> my_cube.global_coords.add('distance', 'pos.distance', 1 * u.m)

`~ndcube.GlobalCoords` allows multiple coordinates of the same physical type.  Therefore when adding a global coordinate, you must provide a unique coordinate name, its physical time and the coordinate value.  The value of the coordinate can be accessed by indexing the `~ndcube.GlobalCoords` instance with the coordinate name.

.. code-block:: python

  >>> my_cube.global_coords['distance']
  <Quantity 1. m>

The coordinate's physical type can be accessed via the `~ndcube.GlobalCoords.physical_types` `dict` property.

.. code-block:: python

  >>> my_cube.global_coords.physical_types['distance']
  'pos.distance'

Because `~ndcube.GlobalCoords` inherits from `Mapping`, it contains a number of mixin methods similar to those of `dict`.

.. code-block:: python

  >>> list(my_cube.global_coords.keys())  # Returns a list of global coordinate names
  ['distance']
  >>> list(my_cube.global_coords.values())  # Returns a list of coordinate values
  [<Quantity 1. m>]
  >>> list(my_cube.global_coords.items())  # Returns a list of (name, value) pairs
  [('distance', <Quantity 1. m>)]

One of the most common use cases for `~ndcube.GlobalCoords` is associated with slicing.  In addition to tracking and updating the `~ndcube.NDCube.wcs` and `~ndcube.NDCube.extra_coords` objects, `~ndcube.NDCube`'s slicing infrastucture also identifies when an axis has been dropped.  It determines the value of any independent coordinates at the location along the dropped axis axis at which the cube was sliced and enters it into the `~ndcube.GlobalCoords` object.

.. code-block:: python

  >>> my_2d_cube = my_cube[:, :, 0]
  >>> my_2d_cube.array_axis_physical_types  # Note the wavelength axis is now gone.
  [('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
   ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon', 'time')]

  >>> # The wavelength value at the slicing location is now in the GLobalCoords object.
  >>> list(my_2d_cube.global_coords.keys())
  ['distance', 'em.wl']
  >>> my_2d_cube.global_coords.physical_types['em.wl']
  'em.wl'
  >>> my_2d_cube.global_coords['em.wl']
  <SpectralCoord 1e-9 m>

.. _cube_coordinates:

NDCube Coordinate Transformations
=================================
WCS objects are a powerful and concise way of storing complex functional coordinate transformations.  However, their API be cumbersome when the coordinates along a whole axis is desired.  Making this process easy and intuitive is the purpose of `ndcube.NDCube.axis_world_coords`.  Using the information on the data dimensions and optional inputs from the user, this method returns high level coordinate objects - e.g. `~astropy.coordinates.SkyCoord`, `~astropy.time.Time`, `~astropy.coordinates.SpectralCoord`, `~astropy.units.Quantity` - containing the coordinates at each array element.  Let's say we wanted the wavelength values along the spectral axis of ``my_cube``.  We can do this in a couple ways.  First we can provide `~ndcube.NDCube.axis_world_coords` with the array axis number of the spectral axis.

.. code-block:: python

  >>> my_cube.axis_world_coords(2)
  WARNING: target cannot be converted to ICRS, so will not be set on SpectralCoord [astropy.wcs.wcsapi.fitswcs]
  (<SpectralCoord [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09, 1.10e-09] m>,)

Alternatively we can provide a unique substring of the physical type of the coordinate, stored in `ndcube.NDCube.wcs.world_axis_physical_types`:

.. code-block:: python

  >>> my_cube.wcs.world_axis_physical_types
  ['em.wl', 'custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon']
  >>> # Since 'wl' is unique to the wavelength axis name, let's use that.
  >>> my_cube.axis_world_coords('wl')
  WARNING: target cannot be converted to ICRS, so will not be set on SpectralCoord [astropy.wcs.wcsapi.fitswcs]
  (<SpectralCoord [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09, 1.10e-09] m>,)

As discussed above, some WCS axes are not independent.  For those axes, `~ndcube.NDCube.axis_world_coords` returns objects with the same number of dimensions as dependent axes.  For example, helioprojective longitude and latitude are dependent.  Therefore if we ask for longitude, we will get back a `~astropy.coordinates.SkyCoord` with containing 2-D latitude and longitude arrays with the same shape as the array axes to which they correspond.  For example:

.. code-block:: python

  >>> celestial = my_cube.axis_world_coords('lon')[0]  # Must extract object from returned tuple with [0]
  WARNING: target cannot be converted to ICRS, so will not be set on SpectralCoord [astropy.wcs.wcsapi.fitswcs]
  >>> my_cube.dimensions
  <Quantity [3., 4., 5.] pix>
  >>> celestial.shape
  (3, 4)
  >>> celestial
  <SkyCoord (Helioprojective: obstime=None, rsun=695700.0 km, observer=None): (Tx, Ty) in arcsec
    [[(2160.07821927, 4.56894119e-02), (2159.96856373, 1.79995614e+03),
      (2159.85889149, 3.59986658e+03), (2159.74920255, 5.39950295e+03)],
     [(3600.        , 4.56905253e-02), (3600.        , 1.80000000e+03),
      (3600.        , 3.59995431e+03), (3600.        , 5.39963453e+03)],
     [(5039.92178073, 4.56894119e-02), (5040.03143627, 1.79995614e+03),
      (5040.14110851, 3.59986658e+03), (5040.25079745, 5.39950295e+03)]]>

It is also possible to request more than one axis's world coordinates by setting ``axes`` to an iterable of data axis number and/or axis type strings.  The coordinate objects are returned in world axis order in accordance with APE 14.

.. code-block:: python

  >>> my_cube.axis_world_coords(2, 'lon')
  (<SpectralCoord [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09, 1.10e-09] m>,
   <SkyCoord (Helioprojective: obstime=None, rsun=695700.0 km, observer=None): (Tx, Ty) in arcsec
       [[(2160.07821927, 4.56894119e-02), (2159.96856373, 1.79995614e+03),
         (2159.85889149, 3.59986658e+03), (2159.74920255, 5.39950295e+03)],
        [(3600.        , 4.56905253e-02), (3600.        , 1.80000000e+03),
         (3600.        , 3.59995431e+03), (3600.        , 5.39963453e+03)],
        [(5039.92178073, 4.56894119e-02), (5040.03143627, 1.79995614e+03),
         (5040.14110851, 3.59986658e+03), (5040.25079745, 5.39950295e+03)]]>)

If the user wants the world coordinates for all the axes, ``axes`` can be set to ``None``, which
is in fact the default.

.. code-block:: python

  >>> my_cube.axis_world_coords()
  (<SpectralCoord [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09, 1.10e-09] m>,
   <SkyCoord (Helioprojective: obstime=None, rsun=695700.0 km, observer=None): (Tx, Ty) in arcsec
       [[(2160.07821927, 4.56894119e-02), (2159.96856373, 1.79995614e+03),
         (2159.85889149, 3.59986658e+03), (2159.74920255, 5.39950295e+03)],
        [(3600.        , 4.56905253e-02), (3600.        , 1.80000000e+03),
         (3600.        , 3.59995431e+03), (3600.        , 5.39963453e+03)],
        [(5039.92178073, 4.56894119e-02), (5040.03143627, 1.79995614e+03),
         (5040.14110851, 3.59986658e+03), (5040.25079745, 5.39950295e+03)]]>)

By default `~ndcube.NDCube.axis_world_coords` returns the coordinates at the
center of each pixel. However, the pixel edges can be obtained by setting
the ``edges`` kwarg to ``True``. For example:

.. code-block:: python

  >>> my_cube.axis_world_coords(edges=True)
  (<SpectralCoord [1.01e-09, 1.03e-09, 1.05e-09, 1.07e-09, 1.09e-09, 1.11e-09] m>,
   <SkyCoord (Helioprojective: obstime=None, rsun=695700.0 km, observer=None): (Tx, Ty) in arcsec
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
         (5760.41458147, 6298.94094507)]]>)

`~ndcube.NDCube.axis_world_coords` also allows the user to pick which WCS object should be used, `ndcube.NDCube.wcs` or `ndcube.NDCube.combined_wcs` by setting the ``wcs=`` keyword.  This means that extra_coords can be retrieved, or not, as the user wishes.

.. code-block:: python

  >>> combined_ccords = my_cube.axis_world_coords(wcs=my_cube.combined_wcs)

Working with Raw Coordinates
............................

If users would prefer not to deal with high level coordinate objects, they can elect to use `ndcube.NDCube.axis_world_coords_values`.  The API for this method is the same as `~ndcube.NDCube.axis_world_coords`.  The only difference is that `~astropy.units.Quantity` objects are returned, one for each physical type requested.  In the above case this means that there would be seperate `~astropy.units.Quantity` objects for latitude and longitude, but they would both have the same 2-D shape.  The `~astropy.units.Quantity` objects are returned in world order and correspond to the physical types in the `~astropy.wcs.WCS.world_axis_physical_types`.  The `~astropy.units.Quantity` objects do not contain important contextual information, such as reference frame, which is needed to fully interpret the coordinate values.  However for some use cases this level of completeness is not needed.

.. code-block:: python

  >>> coord_values = my_cube.axis_world_coords_values()

.. _sequence_coordinates::

NDCubeSequence Coordinate Transformations
=========================================

Sequence Axis Coordinates
-------------------------
As described in the :ref:`ndcubesequence` section, the sequence axis can be thought of as an additional array axis perpendicular to those of the cubes within an `~ndcube.NDCubeSequence`.  In that model, the `~ndcube.GlobalCoords` on each `~ndcube.NDCube` represent coordinate values along the sequence axis.  The `ndcube.NDCubeSequence.sequence_axis_coords` property collates a list for each global coordinate with each element giving the coordinate value from the corresponding `~ndcube.NDCube`.  These lists are returned as a `dict` with the keys being the coordinate names.  To demonstrate this, whose cubes have `~ndcube.GlobalCoords`.

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

  >>> import astropy.units as u
  >>> from ndcube import NDCube, NDCubeSequence
  >>> my_cube0 = NDCube(data0, input_wcs)
  >>> my_cube0.global_coords.add('distance', 'pos.distance', 1 * u.m)
  >>> my_cube1 = NDCube(data1, input_wcs)
  >>> my_cube1.global_coords.add('distance', 'pos.distance', 2 * u.m)
  >>> my_cube2 = NDCube(data2, input_wcs)
  >>> my_cube2.global_coords.add('distance', 'pos.distance', 3 * u.m)
  >>> my_sequence = NDCubeSequence([my_cube0, my_cube1, my_cube2])

Now call `ndcube.NDCubeSequence.sequence_axis_coords`.

.. code-block:: python

  >>> my_sequence.sequence_axis_coords
  {'distance': [<Quantity 1. m>, <Quantity 2. m>, <Quantity 3. m>]}

As with any `dict`, the coordinate names can be seen via the ``.keys()`` method, while the values of a coordinate can be retrieved by indexing with the coordinate name.

.. code-block:: python

  >>> my_sequence.sequence_axis_coords.keys()
  dict_keys(['distance'])
  >>> my_sequence.sequence_axis_coords['distance']
  [<Quantity 1. m>, <Quantity 2. m>, <Quantity 3. m>]

Common Axis Coordinates
-----------------------
The :ref:`ndcubesequence` section also explains how a common axis can be defined for a `~ndcube.NDCubeSequence`, signifying that the sequence axis is in fact parallel to one of the `~ndcube.NDCube` array axes and that the cubes can be thought of as arranged sequentially along that axis.  In this model, coordinates along the common_axis can be concatenated because they are ordered.  The `ndcube.NDCubeSequence.common_axis_coords` property finds the physical types associated with the common axis in each cube and concatenates them.

.. code-block:: python

  >>> my_sequence = NDCubeSequence([my_cube0, my_cube1, my_cube2], common_axis=2)
  >>> my_sequence.common_axis_coords
  [[<SpectralCoord 1.02e-09 m>,
    <SpectralCoord 1.04e-09 m>,
    <SpectralCoord 1.06e-09 m>,
    <SpectralCoord 1.08e-09 m>,
    <SpectralCoord 1.1e-09 m>,
    <SpectralCoord 1.02e-09 m>,
    <SpectralCoord 1.04e-09 m>,
    <SpectralCoord 1.06e-09 m>,
    <SpectralCoord 1.08e-09 m>,
    <SpectralCoord 1.1e-09 m>,
    <SpectralCoord 1.02e-09 m>,
    <SpectralCoord 1.04e-09 m>,
    <SpectralCoord 1.06e-09 m>,
    <SpectralCoord 1.08e-09 m>,
    <SpectralCoord 1.1e-09 m>]]
