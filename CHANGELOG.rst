1.1 (Unreleased)
================

New Features
------------
- Created a new `~ndcube.NDCubeBase` which has all the functionality
  of `~ncube.NDCube` except the plotting.  The old ``NDCubeBase``
  which outlined the ``NDCube`` API was renamed ``NDCubeABC``.
  `~ndcube.NDCube` has all the same functionality as before except is
  now simply inherits from `~ndcube.NDCubeBase` and
  `~ndcube.mixins.plotting.NDCubePlotMixin`.

API Changes
-----------
- Replaced API of what was previously ``utils.wcs.get_dependent_axes``,
  with two new functions, ``utils.wcs.get_dependent_data_axes`` and
  ``utils.wcs.get_dependent_wcs_axes``. This was inspired by a new
  implementation in ``glue-viz`` which is intended to be merged into
  ``astropy`` in the future.  This API change helped fix the
  ``NDCube.world_axis_physical_type`` bug listed below. [#80]

Bug Fixes
---------
- Allowed `~ndcube.NDCubeBase.axis_world_coords` to accept negative
  axis indices as arguments. [#106]


1.0.1
==================

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
