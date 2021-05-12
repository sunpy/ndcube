1.3.0 (2020-03-27)
==================

Features
--------

- Add new NDCollection class for linking and manipulating partially or non-aligned NDCubes or NDCubeSequences. (`#238 <https://github.com/sunpy/ndcube/pull/238>`__)


Bug Fixes
---------

- Fixed the files included and excluded from the tarball. (`#212 <https://github.com/sunpy/ndcube/pull/212>`__)
- Fix crashing bug when an NDCube axis after the first is sliced with a numpy.int64. (`#223 <https://github.com/sunpy/ndcube/pull/223>`__)
- Raises error if NDCube is sliced with an Ellipsis. (`#224 <https://github.com/sunpy/ndcube/pull/224>`__)
- Changes behavior of NDCubeSequence slicing. Previously, a slice item of interval
  length 1 would cause an NDCube object to be returned. Now an NDCubeSequence made
  up of 1 NDCube is returned. This is consistent with how interval length 1 slice
  items slice arrays. (`#241 <https://github.com/sunpy/ndcube/pull/241>`__)


1.2.0 (2019-09-10)
==================

Features
--------

- Changed all instances of "missing_axis" to "missing_axes" (`#157 <https://github.com/sunpy/ndcube/pull/157>`__)
- Added a feature to get the pixel_edges from `ndcube.NDCube.axis_world_coords` (`#174 <https://github.com/sunpy/ndcube/pull/174>`__)


Bug Fixes
---------

- `ndcube.NDCube.world_axis_physical_types` now sets the axis label to the WCS CTYPE if no corresponding IVOA name can be found. (`#164 <https://github.com/sunpy/ndcube/pull/164>`__)
- Fixed the bug of using `pixel_edges` instead of `pixel_values` in plotting (`#176 <https://github.com/sunpy/ndcube/pull/176>`__)
- Fix 2D plotting from crashing when both data and WCS are 2D. (`#182 <https://github.com/sunpy/ndcube/pull/182>`__)
- Fix the ability to pass a custom Axes to `ndcube.NDCube.plot` for a 2D cube. (`#204 <https://github.com/sunpy/ndcube/pull/204>`__)


Trivial/Internal Changes
------------------------

- Include more helpful error when invalid item type is used to slice an `~ndcube.NDCube`. (`#158 <https://github.com/sunpy/ndcube/pull/158>`__)


1.1
===

API-Breaking Changes
--------------------
- `~ndcube.NDCubeBase.crop_by_extra_coord` API has been broken and
  replaced.
  Old version:
  ``crop_by_extra_coord(min_coord_value, interval_width, coord_name)``.
  New version:
  ``crop_by_extra_coord(coord_name, min_coord_value,  max_coord_value)``.
  [#142]

New Features
------------
- Created a new `~ndcube.NDCubeBase` which has all the functionality
  of `~ncube.NDCube` except the plotting.  The old ``NDCubeBase``
  which outlined the ``NDCube`` API was renamed ``NDCubeABC``.
  `~ndcube.NDCube` has all the same functionality as before except is
  now simply inherits from `~ndcube.NDCubeBase` and
  `~ndcube.mixins.plotting.NDCubePlotMixin`. [#101]
- Moved NDCubSequence plotting to a new mixin class,
  NDCubSequencePlotMixin, making the plotting an optional extra.  All
  the non-plotting functionality now lives in the NDCubeSequenceBase
  class. [#98]
- Created a new `~ndcube.NDCubeBase.explode_along_axis` method that
  breaks an NDCube out into an NDCubeSequence along a chosen axis.  It
  is equivalent to
  `~ndcube.NDCubeSequenceBase.explode_along_axis`. [#118]
- NDCubeSequence plot mixin can now animate a cube as a 1-D line if a single
  axis number is supplied to plot_axis_indices kwarg.

API Changes
-----------
- Replaced API of what was previously ``utils.wcs.get_dependent_axes``,
  with two new functions, ``utils.wcs.get_dependent_data_axes`` and
  ``utils.wcs.get_dependent_wcs_axes``. This was inspired by a new
  implementation in ``glue-viz`` which is intended to be merged into
  ``astropy`` in the future.  This API change helped fix the
  ``NDCube.world_axis_physical_type`` bug listed below. [#80]
- Give users more control in plotting both for NDCubePlotMixin and
  NDCubeSequencePlotMixin.  In most cases the axes coordinates, axes
  units, and data unit can be supplied manually or via supplying the
  name of an extra coordinate if it is wanted to describe an
  axis. In the case of NDCube, the old API is currently still
  supported by will be removed in future versions. [#98 #103]

Bug Fixes
---------
- Allowed `~ndcube.NDCubeBase.axis_world_coords` to accept negative
  axis indices as arguments. [#106]
- Fixed bug in ``NDCube.crop_by_coords`` in case where real world
  coordinate system was rotated relative to pixel grid. [#113].
- `~ndcube.NDCubeBase.world_axis_physical_types` is now not
  case-sensitive to the CTYPE values in the WCS. [#109]
- `~ndcube.NDCubeBase.plot` now generates a 1-D line animation when
  image_axis is an integer.


1.0.1
=====

New Features
------------
- Added installation instructions to docs. [#77]

Bug Fixes
---------
- Fixed bugs in ``NDCubeSequence`` slicing and
  ``NDCubeSequence.dimensions`` in cases where sub-cubes contain
  scalar ``.data``. [#79]
- Fixed ``NDCube.world_axis_physical_types`` in cases where there is a
  ``missing`` WCS axis. [#80]
- Fixed bugs in converting between negative data and WCS axis
  numbers. [#91]
- Add installation instruction to docs. [#77]
- Fix function name called within NDCubeSequence.plot animation update
  plot. [#95]
