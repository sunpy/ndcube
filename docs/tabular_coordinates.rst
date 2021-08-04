.. _tabular_coordinates:

===================
Tabular Coordinates
===================

So far we have assumed that we are constructing an NDCube using a WCS that has been read from a file or explicitly created.
However, in some cases we may have tables giving the coordinate values at each pixel and converting these into a WCS manually can be tedious.
Therefore ndcube provides tools for constructing WCSes from such tables.

Tabular coordinates are useful when there is no mathematical description of the axis, or when it's a natural fit for the data you have.
It's worth considering that tabular coordinates are generally not as polished as a functional transform in a WCS.
Therefore, if possible, building a functional WCS for your coordinate system is highly recommended.

Tabular Coordinates and WCSes
=============================

All coordinate information in ndcube is represented as a WCS.
Even the `.ExtraCoords` class, which allows the user to add tabular data to axes, uses the
`gwcs <https://gwcs.readthedocs.io/en/stable/>`_ library to store this information as a WCS.
This enables ndcube's coordinate transformation and plotting functions to leverage the same infrastructure, irrespective of whether the coordinates are functional or tabular.

The FITS WCS standard also supports tabular axes with the ``-TAB`` CTYPE.
Support for reading files using this convention has (reasonably) recently been added to Astropy, so if you have a FITS file using this convention you should be able to load it into a `~astropy.wcs.WCS` object.
If you wish to be able to serialise your NDCube object to FITS files you will need to manually construct a WCS object using the ``-TAB`` convention.

The functionality provided by ndcube makes it easy to construct a `gwcs.wcs.WCS` object backed by lookup tables.
At the time of writing there are some known issues with the support for generic lookup tables in gwcs.

Constructing a WCS from Lookup Tables
=====================================

ndcube supports constructing lookup tables from `~astropy.coordinates.SkyCoord`,  `~astropy.time.Time` and `~astropy.units.Quantity` objects.
These objects are wrapped in `BaseTableCoordinate <.table_coord>` objects which can be composed together into a multi-dimensional WCS.

.. note::

   Only one dimensional tables are currently supported. It is possible to construct higher dimensional lookup tables by "meshing" the inputs, which is described below.

A simple example of constructing a WCS from a lookup table in a `.TimeTableCoordinate`
is the following temporal axis::

  >>> from astropy.time import Time
  >>> import astropy.units as u
  >>> from ndcube.extra_coords import TimeTableCoordinate

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

We can extend this example to be a space-space-time cube.
In this example we are going to utilize the ``mesh=`` keyword argument for the first time.
This keyword argument interprets the input to `.SkyCoordTableCoordinate` in a similar way to `numpy.meshgrid`.

.. code-block::

  >>> from astropy.coordinates import SkyCoord
  >>> from ndcube.extra_coords import SkyCoordTableCoordinate

  >>> icrs_table = SkyCoord(range(10)*u.deg, range(10, 20)*u.deg)
  >>> icrs_table
  <SkyCoord (ICRS): (ra, dec) in deg
      [(0., 10.), (1., 11.), (2., 12.), (3., 13.), (4., 14.), (5., 15.),
       (6., 16.), (7., 17.), (8., 18.), (9., 19.)]>

  >>> gwcs = (TimeTableCoordinate(time_axis) & SkyCoordTableCoordinate(icrs_table, mesh=True)).wcs
  >>> gwcs
  <WCS(output_frame=CompositeFrame, input_frame=PixelFrame, forward_transform=Model: CompoundModel
  Inputs: ('x', 'x0', 'x1')
  Outputs: ('y', 'y0', 'y1')
  Model set size: 1
  Expression: [0] & [1] & [2]
  Components:
      [0]: <Tabular1D(points=(<Quantity [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.] pix>,), lookup_table=[    0.  3600.  7200. 10800. 14400. 18000. 21600. 25200. 28800. 32400.] s)>
  <BLANKLINE>
      [1]: <Tabular1D(points=(<Quantity [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.] pix>,), lookup_table=[0. 1. 2. 3. 4. 5. 6. 7. 8. 9.] deg)>
  <BLANKLINE>
      [2]: <Tabular1D(points=(<Quantity [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.] pix>,), lookup_table=[10. 11. 12. 13. 14. 15. 16. 17. 18. 19.] deg)>
  Parameters:)>

As you can see the coordinate information is stored in memory efficient one dimensional tables, and then converted to a two dimensional coordinate when needed.

.. note::

   Due to `a limitation <https://github.com/spacetelescope/gwcs/issues/120>`__ in gwcs only unit spherical (two dimensional) SkyCoords are supported at this time.
