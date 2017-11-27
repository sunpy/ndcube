=================
NDCube
=================

`NDCube` is the fundamental class of the ndcube package and is designed
to handle array contained in a single N-D array described by a single
WCS transformation.  `NDCube` is subclassed from `astropy.nddata.NDData`
and so inherits the same attributes for data, wcs, uncertainty, mask,
meta, and unit.  The WCS object contained in the .wcs attribute is
subclassed from `astropy.wcs.WCS` and contains a few additional
attributes to enable to keep track of its relationship to the data.

Initialization
----------

To initialize the most basic `NDCube` object, all you need is a
`numpy.ndarray` containing the data, and an `astropy.wcs.WCS` object
describing the transformation from array-element space to real world
coordinates.  Let data be the array and wcs be the WCS object.  Then
you can create an `NDCube` by doing::

  import ndcube
  my_cube = ndcube.NDCube(data, wcs)

Thanks to the fact that `NDCube` is subclassed from
`astropy.nddata.NDData`, you can also supply metadata (`dict` or
dict-like), a data mask (boolean `numpy.ndarray`), an
uncertainty array (`numpy.ndarray`) describing the uncertainty of each data array value,
and a unit (`astropy.units.Unit` or unit `str`).  For example::

  my_cube = ndcube.NDCube(data, wcs, uncertainty=uncertainty,
  mask=mask, meta=meta, unit=None)

N.B. Following the unfortunately confusing convention, the order of
the axes in the WCS object are reversed compared to the data.

Dimensions
---------

NDCube has a useful property for inspecting its data shape and
axis types, `NDCube.dimensions`::

  my_cube.dimensions

This returns a named tuple with a "shape" and "axis_types" attribute.
"shape" is an `astropy.units.Quantity` of pixel units giving the
length of each dimension in the `NDCube`.  Meanwhile, "axis_types" is
`list` of `str` giving the WCS transformation type for each axis.
Here the shape and axis types are given in data order, not WCS order.

Slicing
-----

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

Missing Axes
----------

Some WCS axis types are coupled.  For example, latitude and longitude
