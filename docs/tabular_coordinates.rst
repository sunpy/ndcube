.. _tabular_coordinates:

===================
Tabular Coordinates
===================

The documentation so far has assumed that you are constructing an NDCube object from a WCS object.
Either one that you have read from a file or explicitly created.
In this section we will discuss the functionality ndcube provides for constructing WCSes from tables of coordinate values.

Tabular coordinates are useful when there is no mathematical description of the axis, or when it's a natural fit for the data you have.
It's worth considering that tabular coordinates are generally not as polished as a functional transform in a WCS, if you can build a functional WCS for your coordinate system then that is highly recommended.


Tabular Coordinates and WCSes
=============================

All coordinate information in ndcube is represented as a WCS.
Even the `.ExtraCoords` class which allows the user to add tabular data to axes generates a WCS which is then used by the coordinate functions and plotting in ndcube.
This is done through the use of the `gwcs` library.

The FITS WCS standard also supports tabular axes with the ``-TAB`` CTYPE.
Support for reading files using this convention has (reasonably) recently been added to Astropy, so if you have a FITS file using this convention you should be able to load it into a `~astropy.wcs.WCS` object.
If you wish to be able to serialise your NDCube object to FITS files you will need to manually construct a WCS object using the ``-TAB`` convention.

The functionality provided by ndcube makes it easy to construct a `gwcs.WCS` object backed by lookup tables.
At the time of writing there are some known issues with the support for generic lookup tables in gwcs.


Constructing a WCS from Lookup Tables
=====================================

ndcube supports constructing lookup tables from `~astropy.coordinates.SkyCoord`,  `~astropy.time.Time` and `~astropy.units.Quantity` objects.
These objects are wrapped in `.BaseTableCoordinate` objects which can be composed together into a multi-dimensional WCS.

A simple example of constructing a WCS from a lookup table is the following temporal axis::

  >>> from astropy.time import Time
  >>> import astropy.units as u
  >>> from ndcube.extra_coords.lookup_table_coord import TimeTableCoordinate

  >>> time_axis = Time("2021-01-01T00:00:00") + (list(range(10)) * u.hour)
  >>> time_axis
  <Time object: scale='utc' format='isot' value=['2021-01-01T00:00:00.000' '2021-01-01T01:00:00.000'
   '2021-01-01T02:00:00.000' '2021-01-01T03:00:00.000'
   '2021-01-01T04:00:00.000' '2021-01-01T05:00:00.000'
   '2021-01-01T06:00:00.000' '2021-01-01T07:00:00.000'
   '2021-01-01T08:00:00.000' '2021-01-01T09:00:00.000']>

  >>> gwcs = TimeTableCoordinate(time_axis).wcs
  >>> gwcs
  <WCS(output_frame=TemporalFrame, input_frame=PixelFrame, forward_transform=Model: Tabular1D
  N_inputs: 1
  N_outputs: 1
  Parameters:
    points: (<Quantity [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.] pix>,)
    lookup_table: [    0.  3600.  7200. 10800. 14400. 18000. 21600. 25200. 28800. 32400.] s
    method: linear
    fill_value: nan
    bounds_error: False)>

This `gwcs.WCS` object can then be passed to the constructor of `.NDCube` alongside your array and other parameters.


Combining Two Coordinates into a Single WCS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can extend this example to be a space-space-time cube::

  >>> from astropy.coordinates import SkyCoord
  >>> from ndcube.extra_coords.lookup_table_coord import SkyCoordTableCoordinate

  >>> icrs_table = SkyCoord(range(10)*u.deg, range(10)*u.deg)
  >>> icrs_table
  <SkyCoord (ICRS): (ra, dec) in deg
      [(0., 0.), (1., 1.), (2., 2.), (3., 3.), (4., 4.), (5., 5.), (6., 6.),
       (7., 7.), (8., 8.), (9., 9.)]>

  >>> gwcs = (TimeTableCoordinate(time_axis) & SkyCoordTableCoordinate(icrs_table, mesh=True)).wcs
  >>> gwcs
  <WCS(output_frame=CompositeFrame, input_frame=PixelFrame, forward_transform=Model: CompoundModel
      Inputs: ('x', 'x0', 'x1')
      Outputs: ('y', 'y0', 'y1')
      Model set size: 1
      Expression: [0] & ([1] | [2] & [3])
      Components:
          [0]: <Tabular1D(points=(<Quantity [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.] pix>,), lookup_table=[    0.  3600.  7200. 10800. 14400. 18000. 21600. 25200. 28800. 32400.] s)>
      <BLANKLINE>
          [1]: <Mapping([0, 1, 0, 1])>
      <BLANKLINE>
          [2]: <Tabular2D(points=[<Quantity [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.] pix>, <Quantity [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.] pix>], lookup_table=[[0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
           [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
           [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
           [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
           [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
           [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
           [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
           [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
           [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
           [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]] deg)>
      <BLANKLINE>
          [3]: <Tabular2D(points=[<Quantity [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.] pix>, <Quantity [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.] pix>], lookup_table=[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
           [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
           [2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]
           [3. 3. 3. 3. 3. 3. 3. 3. 3. 3.]
           [4. 4. 4. 4. 4. 4. 4. 4. 4. 4.]
           [5. 5. 5. 5. 5. 5. 5. 5. 5. 5.]
           [6. 6. 6. 6. 6. 6. 6. 6. 6. 6.]
           [7. 7. 7. 7. 7. 7. 7. 7. 7. 7.]
           [8. 8. 8. 8. 8. 8. 8. 8. 8. 8.]
           [9. 9. 9. 9. 9. 9. 9. 9. 9. 9.]] deg)>
      Parameters:)>
