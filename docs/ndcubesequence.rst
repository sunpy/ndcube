===========
NDCubeSequence
===========

`NDCubeSequence` is a class for handling multiple `NDCube` objects as
though they were one contiguous data set.  Another way of thinking
about it is that `NDCubeSequence` provides the ability to manipulate a
data set described by multiple separate WCS transformations.

Regarding implementation, and `NDCubeSequence` instance is effectively
a list of `NDCube` instances with some helper methods attached.

Initialization
----------

To initialize the most basic `NDCube` object, all you need is a list
of `NDCube` instances::

  from ndcube import NDCubeSequence
  my_sequence = NDCubeSequence([my_cube0, my_cube1])

where my_cube0 and my_cube1 are `NDCube` instances.  While, each
`NDCube` can have its own meta, it is also possible to supply
additional metadata upon initialization of the `NDCubeSequence`.  This
metadata may be common to all sub-cubes or is specific to the sequence
rather than the sub-cubes.  This metadata is input as a dictionary::

  my_sequence = NDCubeSequence([my_cube0, my_cube1],
                                                         meta=my_sequence_metadata)
  
and stored in the meta attribute::

  my_sequence.meta

Meanwhile, the `NDCube` instances are stored in the .data attribute::

  my_sequence.data

However, analgously to `NDCube`, it is strongly advised that the data
is manipulated by slicing the `NDCubeSequence` rather than more
manually delving into the .data attribute.  For more explanation, see
the section on Slicing.

Common Axis
-----------

It is possible (although not required) to set a common axis of the
`NDCubeSequence`.  A common axis is defined as the axis of the
sub-cubes parallel to the axis of the sequence.

For example, assume the 0th axis of the sub-cubes, my_cube0 and
my_cube1, in an `NDCubeSequence`, my_sequence, represent time.  Let's
say my_cube0 represents observations taken from a period before
my_cube1. If mycube0 and mycube1 are ordered chronologically in the
sequence.  Then moving  along the 0th axis of one sub-cube and moving
along the sequence axis from one cube to the next both represent
movement in time.  The difference is simply size of the steps.   In
other words, the 0th slice of my_cube1 along the 0th axis directly
follows in time from the last slice of my_cube0 along the 0th axis.
Therefore it can be said that axis of the sub-cubes common to the
sequence is 0.

To define a common axis, set the kwarg during intialization of
the `NDCubeSequence` to the desired data axis number::

  my_sequence = NDCubeSequence([my_cube0, my_cube1], common_axis=0)

The common axis is then stored in the private _common_axis attribute::

  my_sequence ._common_axis

Defining a common axis enables to full range of the `NDCubeSequence`
features to be utilized including `NDCubeSequence.plot`,
`NDCubeSequence._common_axis_extra_coords`, and
`NDCubeSequence.index_as_cube`. See following sections for more
details on these features.

Dimensions
---------

`NDCubeSequence` has a useful dimensions property for inspecting the
size and shape of an `NDCubeSequence` instance::

  my_sequence.dimensions

This is analagous to the `NDCube.dimensions` property.  Like NDCube it
returns a named tuple with a "shape" and "axis_types" where the
values of the 0th sub-cube are returned as `astropy.units.Quantity`
and `str` objects, respectively.  In addition however, another
dimension is return at the start of the named tuple.  Its value is an
`int` (not a Quantity) giving the number of sub-cubes in the
sequence.  Since this is not a WCS axis, its axis type is given the
label "Sequence Axis".

Say we have three 3-D NDCubes in an NDCubeSequence.  Then the output of
the dimensions property might look something like::

  my_sequence.dimensions
      SequenceDimensionPair(shape=(3, <Quantity 10.0 pix>, <Quantity
      20.0 pix>, <Quantity 30.0 pix>), axis_types=('Sequence Axis',
      'HPLN-TAN', 'HPLT-TAN', 'WAVE')) 

As the dimensions property returns a named tuple, the shape and axis
types can be accessed directly::

  my_sequence.dimensions.shape
  my_sequence.dimensions.axis_types


Slicing
-----
As with `NDCube`, slicing an `NDCubeSequence` using the standard
slicing API simulataneously slices the data arrays, WCS objects, masks,
uncertainty arrays, etc. in each relevant sub-cube.  For example, say
we have three NDCubes in an `NDCubeSequence`, each of shape (10, 20,
30). So `NDCubeSequence dimensions` would give us something like::

  my_sequence.dimensions.shape
      (3, <Quantity 10.0 pix>, <Quantity 20.0 pix>, <Quantity 30.0
      pix>)

Say we want to obtain a region of interest between the 10th and 15th
pixels in the 2nd dimension and 5th and 25th pixels in the 3rd
dimension of the 3rd slice along the 0th axis in each of 1st and 2nd
sub-cubes in the sequence.  This would a cumbersome slicing operation
if treating the sub-cube independently, probably involving a for
loop.  (This would be made even worse without the power of `NDCube`
where the data arrays, WCS objects, masks, uncertainty arrays,
etc. would all have to be sliced independently!)  However, with
`NDCubeSequence` this becomes as simple as indexing a single array::

  regions_of_interest_in_sequence = my_sequence[1:3, 3, 10:15, 5:25]

This will return a new `NDCubeSequence` with 2 2-D NDCubes, one for
each region of interest from the 3rd slice along the 0th axis in each
original sub-cube.  If our regions of interest only came from a single
sub-cube -- say the 3rd and 4th slices along the 0th axis in the 1st
sub-cube, an NDCube is returned::

  roi_from_single_subcube = my_sequence[1, 3:5, 10:15, 5:25]

If a common axis has been defined for the `NDCubeSequence` one can
think of it as a contiguous data set with different sections along the
common axis described by different WCS translations.  Therefore it
would be useful to be able to index the sequence as though it were one
single cube.  This can be achieved with the
`NDCubeSequence.index_as_cube` property.  In our above example,
my_sequence has a shape of (3, <Quantity 10.0 pix>, <Quantity 20.0
pix>, <Quantity 30.0 pix>) and a common cube axis of 0.  Therefore we
can think of my_sequence as a having an effective cube-like shape of
(<Quantity 30.0 pix>, <Quantity 20.0 pix>, <Quantity 30.0 pix>) where
the first sub-cube extends along the 0th cube-like axis from 0 to 10,
the second from 10 to 20 and the third from 20 to 30.  Say we want to
extract the same region of interest as above, i.e. from the 3rd and
4th slices along the 0th cube axis in the 1st sub-cube.  Since we are
0-based counting, this corresponds to the 13th and 14th (10+3 and
10+4) slices in the cube-like indexing format.  So get the same result
as above we would type::

  roi_from_single_subcube = my_sequence.index_as_cube[13:15, 10:15, 5:25]

In this case the entire region came from a single sub-cube so an
NDCube is returned.  However, `NDCubeSequence.index_as_cube` also
works when the region of interest spans multiple sub-cubes in the
sequence.  Say we want the same region if interest in the 2nd and 3rd
cube dimensions from the final slice along the 0th cube axis of the
0th sub-cube, the whole 1st sub-cube and the 0th slice of the 2nd
sub-cube. In cube-like indexing this corresponds to slices 9 to 21
along to the 0th cube axis::

  roi_across_subcubes = my_sequence.index_as_cube[9:21, 10:15, 5:25]

As the data comes from multiple sub-cubes, a new `NDCubeSequence` is
returned.

Common Axis Extra Coordinates
--------------------------

If common axis is defined, it may be useful to view the extra
coordinates along that common axis defined by each of the sub-cube
`NDCube._extra_coords` as if the `NDCubeSequence` were one contiguous
Cube.  This can be done using the _common_axis_extra_coords property.
This property is private but can be made public in any subclass of
`NDCubeSequence`.  To call it, enter::

  my_sequence._common_axis_extra_coords

This returns a dictionary where each key gives the name of the
coordinate and the value of the key are the values of that coordinate
at each pixel along the common axis.  Since all these coordinates must
be along the common axis, it is not necessary to supply axis
information as it is with `NDCube._extra_coords` making
`NDCubeSequence._common_axis_extra_coords` simpler.
  
Plotting
------

The `NDCubeSequence.plot` method allows the sequence to be animated as
though it were one contiguous `NDCube`. It has the same API and same
kwargs as `NDCube.plot`.  See documentation for `NDCube.plot` for more
details.


Explode Along Axis
----------------

During data analysis, say of a stack of images, you may need to do
fine-pointing adjustments to each exposure, for example, due to
satellite wobble, that isn't accounted for the in the original WCS
translations in your data.  This is not possible with a single WCS
object.  Therefore it may be desirable to break up an `NDCube` of an
`NDCubeSequence` into an sequence of sub-cubes with dimension N-1.
This would enable a separate WCS object to be associated with each
exposure can hence allow manual adjustment of the pointing of each
image.

Rather than manually dividing the datacubes up and deriving the
corresponding WCS object for each exposure, `NDCubeSequence` provides
a useful method, `NDCubeSequence.explode_along_axis`.  To call it
simply provide the number of the data cube axis along which you wish
to break up the sub-cubes::

  exploded_sequence = my_sequence.explode_along_axis(0)

Assuming we are using the same my_sequence as above, with
dimensions.shape (3, <Quantity 10.0 pix>, <Quantity 20.0 pix>,
<Quantity 30.0 pix>), exploded_axis will be an `NDCubeSequence` of 2-D
NDCubes with sequence shape (30, <Quantity 20.0 pix>, <Quantity 30.0
pix>).  Note that any cube axis can be input.  A common axis need not
be defined.
