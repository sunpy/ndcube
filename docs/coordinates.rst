.. _coordinates:

==========================
Coordinate Transformations
==========================

Introduction to WCS
===================

To describe the mapping between array elements/pixels and real world coordinates, ndcube heavily leverages the World Coordinate System (WCS) framework, specifically the tools written by Astropy that implement this framework in Python.
WCS allows a wide variety of projections, rotations and transformations be stored and executed.
Because it allows coordinates transformations to be stored functionally, rather than in memory-heavy lookup tables, and because it caters for both astronomy-specific coordinate systems (e.g. RA & Dec.) as well as simpler, more common ones (e.g. wavelength), WCS has become the most common coordinate transformation framework in astronomy.

The most commonly used WCS implementation in Python is the `astropy.wcs.WCS` object, which stores critical information describing the coordinate transformations as required by the FITS data model (e.g. the reference pixel and its corresponding coordinate values, ``CRPIX`` and ``CRVAL``, and the projection type, ``CTYPE`` etc.).
It also executes these transformations via methods like `~astropy.wcs.WCS.world_to_pixel` and `~astropy.wcs.WCS.pixel_to_world` which convert between pixel indices and world coordinate values.
However, these methods are independent of the data array and the `~astropy.wcs.WCS` object carries little or no information about the data itself.
That is why the ndcube package is needed.

Nonetheless, astropy's WCS implementation is a crucial pillar of ndcube, as is the more generalized offshoot, `gWCS <https://gwcs.readthedocs.io/en/stable/>`__, which provides greater generalization outside of the FITS data model.
Crucially though for ndcube, both implementations adhere to the `Astropy WCS API <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`__.
A familiarity with WCS and the Astropy and gWCS Python implementations will be helpful (although hopefully not essential) in understanding this guide.
We therefore encourage users to read `Astropy's WCS guide <https://docs.astropy.org/en/stable/wcs/>`__ and the `gWCS documentation <https://gwcs.readthedocs.io/en/stable/>`__ to learn more.

In this section we will discuss the features ndcube has built upon these implementations to support the integration of data and coordinates.

NDCube Coordinates
==================

Although WCS objects are a powerful and concise way of storing complex functional coordinate transformations, their API can be cumbersome when the coordinates along a whole axis are desired.
Making this process easy and intuitive is the purpose of the `ndcube.NDCube.axis_world_coords` method.
Using the attached WCS object, information on the data dimensions, and optional inputs from the user, this method returns high level coordinate objects --- e.g. `~astropy.coordinates.SkyCoord`, `~astropy.time.Time`, `~astropy.coordinates.SpectralCoord`, `~astropy.units.Quantity` --- containing the coordinates for each array element.
Say we have a 3-D `~ndcube.NDCube` with a shape of ``(4, 4, 5)`` and physical types of space, space, wavelength.
Now let's say we want the wavelength values along the spectral axis.
We can do this in a couple ways.
First we can provide `~ndcube.NDCube.axis_world_coords` with the array axis number of the spectral axis.

.. expanding-code-block:: python
  :summary: Click to reveal/hide instantiation of the NDCube.

  >>> import astropy.wcs
  >>> import numpy as np

  >>> from ndcube import NDCube

  >>> # Define data array.
  >>> data = np.random.rand(4, 4, 5)

  >>> # Define WCS transformations in an astropy WCS object.
  >>> wcs = astropy.wcs.WCS(naxis=3)
  >>> wcs.wcs.ctype = 'WAVE', 'HPLT-TAN', 'HPLN-TAN'
  >>> wcs.wcs.cunit = 'Angstrom', 'deg', 'deg'
  >>> wcs.wcs.cdelt = 0.2, 0.5, 0.4
  >>> wcs.wcs.crpix = 0, 2, 2
  >>> wcs.wcs.crval = 10, 0.5, 1

  >>> # Now instantiate the NDCube
  >>> my_cube = NDCube(data, wcs=wcs)

.. code-block:: python

  >>> my_cube.axis_world_coords(2)
  (<SpectralCoord [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09, 1.10e-09] m>,)

Alternatively we can provide a unique substring of the physical type of the coordinate, stored in `ndcube.NDCube.wcs.world_axis_physical_types`:

.. code-block:: python

  >>> my_cube.wcs.world_axis_physical_types
  ['em.wl', 'custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon']
  >>> # Since 'wl' is unique to the wavelength axis name, let's use that.
  >>> my_cube.axis_world_coords('wl')
  (<SpectralCoord [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09, 1.10e-09] m>,)

As discussed above, some WCS axes are not independent.
For those axes, `~ndcube.NDCube.axis_world_coords` returns objects with the same number of dimensions as dependent axes.
For example, helioprojective longitude and latitude are dependent.
Therefore if we ask for longitude, we will get back a `~astropy.coordinates.SkyCoord` containing 2-D latitude and longitude arrays with the same shape as the array axes to which they correspond.
For example:

.. code-block:: python

  >>> celestial = my_cube.axis_world_coords('lon')[0]  # Must extract object from returned tuple with [0]
  >>> my_cube.dimensions
  <Quantity [4., 4., 5.] pix>
  >>> celestial.shape
  (4, 4)
  >>> celestial
  <SkyCoord (Helioprojective: obstime=None, rsun=695700.0 km, observer=None): (Tx, Ty) in arcsec
    [[(2160.07821927, 4.56894119e-02), (2159.96856373, 1.79995614e+03),
      (2159.85889149, 3.59986658e+03), (2159.74920255, 5.39950295e+03)],
     [(3600.        , 4.56905253e-02), (3600.        , 1.80000000e+03),
      (3600.        , 3.59995431e+03), (3600.        , 5.39963453e+03)],
     [(5039.92178073, 4.56894119e-02), (5040.03143627, 1.79995614e+03),
      (5040.14110851, 3.59986658e+03), (5040.25079745, 5.39950295e+03)],
     [(6479.70323031, 4.56860725e-02), (6479.92250932, 1.79982456e+03),
      (6480.14182173, 3.59960344e+03), (6480.36116753, 5.39910830e+03)]]>

It is also possible to request more than one axis's world coordinates by setting ``axes`` to an iterable of data axis number and/or axis type strings.
The coordinate objects are returned in world axis order.

.. code-block:: python

  >>> my_cube.axis_world_coords(2, 'lon')
  (<SpectralCoord [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09, 1.10e-09] m>, <SkyCoord (Helioprojective: obstime=None, rsun=695700.0 km, observer=None): (Tx, Ty) in arcsec
      [[(2160.07821927, 4.56894119e-02), (2159.96856373, 1.79995614e+03),
        (2159.85889149, 3.59986658e+03), (2159.74920255, 5.39950295e+03)],
       [(3600.        , 4.56905253e-02), (3600.        , 1.80000000e+03),
        (3600.        , 3.59995431e+03), (3600.        , 5.39963453e+03)],
       [(5039.92178073, 4.56894119e-02), (5040.03143627, 1.79995614e+03),
        (5040.14110851, 3.59986658e+03), (5040.25079745, 5.39950295e+03)],
       [(6479.70323031, 4.56860725e-02), (6479.92250932, 1.79982456e+03),
        (6480.14182173, 3.59960344e+03), (6480.36116753, 5.39910830e+03)]]>)

If the user wants the world coordinates for all the axes, the ``axes`` arg can set to ``None`` or simply omitted.

.. code-block:: python

  >>> my_cube.axis_world_coords()
  (<SpectralCoord [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09, 1.10e-09] m>, <SkyCoord (Helioprojective: obstime=None, rsun=695700.0 km, observer=None): (Tx, Ty) in arcsec
      [[(2160.07821927, 4.56894119e-02), (2159.96856373, 1.79995614e+03),
        (2159.85889149, 3.59986658e+03), (2159.74920255, 5.39950295e+03)],
       [(3600.        , 4.56905253e-02), (3600.        , 1.80000000e+03),
        (3600.        , 3.59995431e+03), (3600.        , 5.39963453e+03)],
       [(5039.92178073, 4.56894119e-02), (5040.03143627, 1.79995614e+03),
        (5040.14110851, 3.59986658e+03), (5040.25079745, 5.39950295e+03)],
       [(6479.70323031, 4.56860725e-02), (6479.92250932, 1.79982456e+03),
        (6480.14182173, 3.59960344e+03), (6480.36116753, 5.39910830e+03)]]>)

By default `~ndcube.NDCube.axis_world_coords` returns the coordinates at the center of each pixel.
However, the coordinates at the edges of each pixel can be obtained by setting the ``pixel_corners`` kwarg to ``True``.
For example:

.. code-block:: python

  >>> my_cube.axis_world_coords(pixel_corners=True)
  (<SpectralCoord [1.01e-09, 1.03e-09, 1.05e-09, 1.07e-09, 1.09e-09, 1.11e-09] m>, <SkyCoord (Helioprojective: obstime=None, rsun=695700.0 km, observer=None): (Tx, Ty) in arcsec
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
        (5760.41458147, 6298.94094507)],
       [(7199.36047891, -899.70880283), (7199.63452676,  899.86866585),
        (7199.90861634, 2699.58313412), (7200.18274766, 4499.1606028 ),
        (7200.45692072, 6298.32719784)]]>)

Working with Raw Coordinates
----------------------------

If users would prefer not to deal with high level coordinate objects, they can elect to use `ndcube.NDCube.axis_world_coords_values`.
The API for this method is the same as `~ndcube.NDCube.axis_world_coords`.
The only difference is that a `~collections.namedtuple` of `~astropy.units.Quantity` objects are returned, one for each physical type requested.
In the above case this means that there would be separate `~astropy.units.Quantity` objects for latitude and longitude, but they would both have the same 2-D shape.
The `~astropy.units.Quantity` objects are returned in world order and correspond to the physical types in the `astropy.wcs.WCS.world_axis_physical_types`.
The `~astropy.units.Quantity` objects do not contain important contextual information, such as reference frame, which is needed to fully interpret the coordinate values.
However for some use cases this level of completeness is not needed.

.. code-block:: python

  >>> my_cube.axis_world_coords_values()
  CoordValues(custom_pos_helioprojective_lon=<Quantity [[0.60002173, 0.59999127, 0.5999608 , 0.59993033],
               [1.        , 1.        , 1.        , 1.        ],
               [1.39997827, 1.40000873, 1.4000392 , 1.40006967],
               [1.79991756, 1.79997847, 1.80003939, 1.80010032]] deg>, custom_pos_helioprojective_lat=<Quantity [[1.26915033e-05, 4.99987815e-01, 9.99962939e-01,
                1.49986193e+00],
               [1.26918126e-05, 5.00000000e-01, 9.99987308e-01,
                1.49989848e+00],
               [1.26915033e-05, 4.99987815e-01, 9.99962939e-01,
                1.49986193e+00],
               [1.26905757e-05, 4.99951267e-01, 9.99889844e-01,
                1.49975231e+00]] deg>, em_wl=<Quantity [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09, 1.10e-09] m>)

.. _extra_coords:

ExtraCoords
===========

So far we have seen how `~ndcube.NDCube` uses its WCS object (``NDCube.wcs``) to store and perform coordinates transformations.
But what if we have alternative or additional coordinates that are not represented by the WCS?
For example, say we have a raster scan from a scanning slit spectrograph whose x-axis is folded in with time.
This occurs because the x-axis is built up over sequential exposures taken at different slit positions.

Our ``NDCube.wcs`` might describe latitude and longitude, but omit time.
So how can we represent time without having to construct a whole new custom WCS object?
One way is to use the `ndcube.ExtraCoords` class located at ``NDCube.extra_coords``.
It provides a mechanism of attaching coordinates to `~ndcube.NDCube` instances in addition to those in the primary WCS object.
This may be desired because, as above, the primary WCS omits a physical type.
Or it may be that the users have an alternative set of coordinates to the primary set at ``.wcs``.
To demonstrate how to use `~ndcube.ExtraCoords`, let's start by creating a `~astropy.time.Time` object representing the time at each location along the first axis of ``my_cube``.

.. code-block:: python

  >>> from astropy.time import Time, TimeDelta
  >>> base_time = Time('2000-01-01', format='fits', scale='utc')
  >>> timestamps = Time([base_time + TimeDelta(60 * i, format='sec') for i in range(data.shape[0])])

By default an `~ndcube.NDCube` is instantiated with an empty `~ndcube.ExtraCoords` object.
So let's add a time coordinate to the `~ndcube.ExtraCoords` instance at ``my_cube.extra_coords``.
To do this we need to supply the physical type of the coordinate, the array axis to which is corresponds, and the values of the coordinate.
The number of values should equal the axis's length (or shape if it corresponds to more than one axis) and the physical type must be a valid `IVOA UCD1+ controlled words <http://www.ivoa.net/documents/REC/UCD/UCDlist-20070402.html>`__ word.
If one does not exist for your coordinate, prepend the type with ``custom:``.

.. code-block:: python

  >>> my_cube.extra_coords.add('time', (0,), timestamps)

An indefinite number of coordinates can be added in this way.
The names of the coordinates can be accessed via the `~ndcube.ExtraCoords.keys` method.

.. code-block:: python

  >>> my_cube.extra_coords.keys()
  ('time',)

The physical types of extra coordinates are also returned by `~ndcube.NDCube.array_axis_physical_types`.

.. code-block:: python

  >>> my_cube.array_axis_physical_types
  [('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon', 'time'), ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'), ('em.wl',)]

The values of the extra coordinates at each array index can be retrieved using and combination of `ndcube.NDCube.axis_world_coords` and `ndcube.NDCube.combined_wcs`.
See :ref:`combined_wcs` below.

.. _combined_wcs:

Combined WCS
------------

The `~ndcube.NDCube.combined_wcs` generates a WCS that combines the extra coords with those stored in the primary WCS.
Unlike `ndcube.ExtraCoords.wcs`, `~ndcube.NDCube.combined_wcs` is a valid WCS for describing the `~ndcube.NDCube` data array and so can be used with the `~ndcube.NDCube` coordinate transformation and plotting features, e.g:

.. code-block:: python

  >>> my_cube.axis_world_coords(wcs=my_cube.combined_wcs)
  (<SpectralCoord [1.02e-09, 1.04e-09, 1.06e-09, 1.08e-09, 1.10e-09] m>, <SkyCoord (Helioprojective: obstime=None, rsun=695700.0 km, observer=None): (Tx, Ty) in arcsec
        [[(2160.07821927, 4.56894119e-02), (2159.96856373, 1.79995614e+03),
          (2159.85889149, 3.59986658e+03), (2159.74920255, 5.39950295e+03)],
         [(3600.        , 4.56905253e-02), (3600.        , 1.80000000e+03),
          (3600.        , 3.59995431e+03), (3600.        , 5.39963453e+03)],
         [(5039.92178073, 4.56894119e-02), (5040.03143627, 1.79995614e+03),
          (5040.14110851, 3.59986658e+03), (5040.25079745, 5.39950295e+03)],
         [(6479.70323031, 4.56860725e-02), (6479.92250932, 1.79982456e+03),
          (6480.14182173, 3.59960344e+03), (6480.36116753, 5.39910830e+03)]]>, <Time object: scale='utc' format='fits' value=['2000-01-01T00:00:00.000' '2000-01-01T00:01:00.000'
     '2000-01-01T00:02:00.000' '2000-01-01T00:03:00.000']>)

Note that the extra coordinate of time is now also returned.

.. _global_coords:

GlobalCoords
============

Sometimes coordinates are not associated with any axis.
Take the case of a 2-D `~ndcube.NDCube` representing a single image.
The time at which that image was taken is important piece of coordinate information.
But because the data does not have a 3rd dimension, it cannot be stored in the WCS or `~ndcube.ExtraCoords` objects.

Storing such coordinates is the role of the `ndcube.GlobalCoords` class.
`~ndcube.NDCube` is instantiated with an empty `~ndcube.GlobalCoords` object already attached at `ndcube.NDCube.global_coords`.
Coordinates can be added to this object if and when the user sees fit.
Let's attach a scalar global coordinate to ``my_cube`` representing some kind of distance.
We do this by supplying the coordinate's name, physical type and value via the `~ndcube.GlobalCoords.add` method.

.. code-block:: python

  >>> import astropy.units as u
  >>> my_cube.global_coords.add('distance', 'pos.distance', 1 * u.m)

Because `~ndcube.GlobalCoords` allows multiple coordinates of the same physical type, a unique coordinate name must be provided.
Furthermore the physical type must be a valid `IVOA UCD1+ controlled words <http://www.ivoa.net/documents/REC/UCD/UCDlist-20070402.html>`__ word.
If one does not exist for your coordinate, prepend the type with ``custom:``.

The value of the coordinate can be accessed by indexing the `~ndcube.GlobalCoords` instance with the coordinate name.

.. code-block:: python

  >>> my_cube.global_coords['distance']
  <Quantity 1. m>

The coordinate's physical type can be accessed via the `~ndcube.GlobalCoords.physical_types` `dict` property.

.. code-block:: python

  >>> my_cube.global_coords.physical_types['distance']
  'pos.distance'

Because `~ndcube.GlobalCoords` inherits from `~collections.abc.Mapping`, it contains a number of mixin methods similar to those of `dict`.

.. code-block:: python

  >>> list(my_cube.global_coords.keys())  # Returns a list of global coordinate names
  ['distance']
  >>> list(my_cube.global_coords.values())  # Returns a list of coordinate values
  [<Quantity 1. m>]
  >>> list(my_cube.global_coords.items())  # Returns a list of (name, value) pairs
  [('distance', <Quantity 1. m>)]

A common use case for `~ndcube.GlobalCoords` is associated with slicing (:ref:`cube_slicing`).
In addition to tracking and updating the `~ndcube.NDCube.wcs` and `~ndcube.NDCube.extra_coords` objects, `~ndcube.NDCube`'s slicing infrastucture also identifies when the array axes to which a coordinate corresponds are dropped.
The values of dropped coordinates at the position where the `~ndcube.NDCube` was sliced are stored in the `astropy.wcs.WCS` instance from where `~ndcube.GlobalCoords` can access and return them.

.. code-block:: python

  >>> my_2d_cube = my_cube[:, :, 0]
  >>> my_2d_cube.array_axis_physical_types  # Note the wavelength axis is now gone.
  [('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon', 'time'),
   ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon')]

  >>> # The wavelength value at the slicing location is now in the GLobalCoords object.
  >>> list(my_2d_cube.global_coords.keys())  # doctest: +SKIP
  ['distance', 'em.wl']  # doctest: +SKIP
  >>> my_2d_cube.global_coords.physical_types['em.wl']  # doctest: +SKIP
  'em.wl'  # doctest: +SKIP
  >>> my_2d_cube.global_coords['em.wl']  # doctest: +SKIP
  <SpectralCoord 1e-9 m>

.. _cube_coordinates:


.. _sequence_coordinates:

NDCubeSequence Coordinates
==========================

Sequence Axis Coordinates
-------------------------

As described in the :ref:`ndcubesequence` section, the sequence axis can be thought of as an additional array axis perpendicular to those of the cubes within an `~ndcube.NDCubeSequence`.
In that model, the `~ndcube.GlobalCoords` on each `~ndcube.NDCube` represent coordinate values along the sequence axis.
The `ndcube.NDCubeSequence.sequence_axis_coords` property collates a list for each global coordinate with each element giving the coordinate value from the corresponding `~ndcube.NDCube`.
These lists are returned as a `dict` with the keys being the coordinate names.
To demonstrate this, let's call `ndcube.NDCube.sequence_axis_coords` on an `~ndcube.NDCubeSequence` whose cubes have `~ndcube.GlobalCoords`.
(Click the "Instantiating NDCubeSequence" link below to reveal the code used to create the `~ndcube.NDCubeSequence`.)

.. expanding-code-block:: python
  :summary: Instantiating NDCubeSequence

  >>> import astropy.units as u
  >>> import astropy.wcs
  >>> import numpy as np
  >>> from ndcube import NDCube, NDCubeSequence

  >>> # Define data arrays.
  >>> shape = (4, 4, 5)
  >>> data0 = np.random.rand(*shape)
  >>> data1 = np.random.rand(*shape)
  >>> data2 = np.random.rand(*shape)
  >>> data3 = np.random.rand(*shape)

  >>> # Define WCS transformations.
  >>> wcs = astropy.wcs.WCS(naxis=3)
  >>> wcs.wcs.ctype = 'WAVE', 'HPLT-TAN', 'HPLN-TAN'
  >>> wcs.wcs.cunit = 'Angstrom', 'deg', 'deg'
  >>> wcs.wcs.cdelt = 0.2, 0.5, 0.4
  >>> wcs.wcs.crpix = 0, 2, 2
  >>> wcs.wcs.crval = 10, 0.5, 1

  >>> # Instantiate NDCubes.
  >>> cube0 = NDCube(data0, wcs=wcs)
  >>> cube0.global_coords.add('distance', 'pos.distance', 1*u.m)
  >>> cube1 = NDCube(data1, wcs=wcs)
  >>> cube1.global_coords.add('distance', 'pos.distance', 2*u.m)
  >>> cube2 = NDCube(data2, wcs=wcs)
  >>> cube2.global_coords.add('distance', 'pos.distance', 3*u.m)
  >>> cube3 = NDCube(data3, wcs=wcs)
  >>> cube3.global_coords.add('distance', 'pos.distance', 4*u.m)

  >>> my_sequence = NDCubeSequence([cube0, cube1, cube2, cube3])

.. code-block:: python

  >>> my_sequence.sequence_axis_coords
  {'distance': [<Quantity 1. m>, <Quantity 2. m>, <Quantity 3. m>, <Quantity 4. m>]}

As with any `dict`, the coordinate names can be seen via the ``.keys()`` method, while the values of a coordinate can be retrieved by indexing with the coordinate name.

.. code-block:: python

  >>> my_sequence.sequence_axis_coords.keys()
  dict_keys(['distance'])
  >>> my_sequence.sequence_axis_coords['distance']
  [<Quantity 1. m>, <Quantity 2. m>, <Quantity 3. m>, <Quantity 4. m>]

Common Axis Coordinates
-----------------------

The :ref:`ndcubesequence` section also explains how a common axis can be defined for a `~ndcube.NDCubeSequence`, signifying that the sequence axis is parallel to one of the `~ndcube.NDCube` array axes.
Take the example of an `~ndcube.NDCubeSequence` of four 3-D NDCubes with axes of space-space-wavelength.
Suppose that each cube represents a different interval in the spectral dimension and that the cubes are arranged in ascending wavelength order within the `~ndcube.NDCubeSequence`, i.e. ``common_axis=2``.
If each NDCube has a shape of ``(4, 4, 5)``, then there are 20 positions along the common axis (5 array elements x 4 NDCubes).

The purpose of `ndcube.NDCubeSequence.common_axis_coords` is to make it easy to get the value of a coordinate at any point along the common axis, irrespective of the cube to which it corresponds.
It determines which coordinates within the NDCubes' WCS and `~ndcube.ExtraCoords` objects correspond to the common axis and are present in all cubes.
For each of these coordinates, a list is produced with the same length as the common axis.
Each entry gives the coordinate value(s) at that position along the common axis.
The coordinates are returned in world axis order.

.. expanding-code-block:: python
  :summary: Click to see instantiation of NDCubeSequence

  >>> from copy import deepcopy

  >>> import astropy.units as u
  >>> import astropy.wcs
  >>> import numpy as np

  >>> from ndcube import NDCube, NDCubeSequence

  >>> # Define data arrays.
  >>> shape = (4, 4, 5)
  >>> data0 = np.random.rand(*shape)
  >>> data1 = np.random.rand(*shape)
  >>> data2 = np.random.rand(*shape)
  >>> data3 = np.random.rand(*shape)

  >>> # Define WCS transformations.
  >>> wcs0 = astropy.wcs.WCS(naxis=3)
  >>> wcs0.wcs.ctype = 'WAVE', 'HPLT-TAN', 'HPLN-TAN'
  >>> wcs0.wcs.cunit = 'm', 'deg', 'deg'
  >>> wcs0.wcs.cdelt = 2e-11, 0.5, 0.4
  >>> wcs0.wcs.crpix = 0, 2, 2
  >>> wcs0.wcs.crval = 1e-9, 0.5, 1
  >>> wcs1 = deepcopy(wcs0)
  >>> wcs1.wcs.crval[0] = 1.1e-9
  >>> wcs2 = deepcopy(wcs0)
  >>> wcs2.wcs.crval[0] = 1.2e-9
  >>> wcs3 = deepcopy(wcs0)
  >>> wcs3.wcs.crval[0] = 1.3e-9

  >>> # Instantiate NDCubes.
  >>> cube0 = NDCube(data0, wcs=wcs0)
  >>> cube1 = NDCube(data1, wcs=wcs1)
  >>> cube2 = NDCube(data2, wcs=wcs2)
  >>> cube3 = NDCube(data3, wcs=wcs3)

  # Instantiate NDCubeSequence.
  >>> my_sequence = NDCubeSequence([cube0, cube1, cube2, cube3], common_axis=2)

.. code-block:: python

  >>> my_sequence.common_axis_coords
  [[<SpectralCoord 1.02e-09 m>,
    <SpectralCoord 1.04e-09 m>,
    <SpectralCoord 1.06e-09 m>,
    <SpectralCoord 1.08e-09 m>,
    <SpectralCoord 1.1e-09 m>,
    <SpectralCoord 1.12e-09 m>,
    <SpectralCoord 1.14e-09 m>,
    <SpectralCoord 1.16e-09 m>,
    <SpectralCoord 1.18e-09 m>,
    <SpectralCoord 1.2e-09 m>,
    <SpectralCoord 1.22e-09 m>,
    <SpectralCoord 1.24e-09 m>,
    <SpectralCoord 1.26e-09 m>,
    <SpectralCoord 1.28e-09 m>,
    <SpectralCoord 1.3e-09 m>,
    <SpectralCoord 1.32e-09 m>,
    <SpectralCoord 1.34e-09 m>,
    <SpectralCoord 1.36e-09 m>,
    <SpectralCoord 1.38e-09 m>,
    <SpectralCoord 1.4e-09 m>]]
