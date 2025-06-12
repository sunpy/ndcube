2.3.2 (2025-06-12)
==================

Bug Fixes
---------

- Removed the check for compatible coordinate systems within `~ndcube.NDCube.reproject_to` as it was redundant. (`#833 <https://github.com/sunpy/ndcube/pull/833>`__)


Internal Changes
----------------

- Make `~ndcube.visualization.mpl_plotter.MatplotlibPlotter` only add data to line plots if at least one value of the data is unmasked and finite. (`#802 <https://github.com/sunpy/ndcube/pull/802>`__)


2.3.1 (2025-02-11)
==================

Documentation
-------------

- Fixed a typo in the docstring of `~ndcube.NDCube.data`.

2.3.0 (2025-01-14)
==================

Breaking Changes
----------------

- "dimensions" property on ~ndcube.NDCube` and `~ndcube.NDCubeSequence` have been deprecated and replaced by "shape" (`#684 <https://github.com/sunpy/ndcube/pull/684>`__)


Removals
--------

- The ability to create an `.NDCollection` object with numerical keys is deprecated as it leads to ambiguous behavior when slicing the collection. (`#778 <https://github.com/sunpy/ndcube/pull/778>`__)


New Features
------------

- Add Sliceable `~ndcube.meta.NDMeta` class for axis-associated metadata. (`#455 <https://github.com/sunpy/ndcube/pull/455>`__)
- Added a new `~ndcube.NDCube.squeeze` method to `ndcube.NDCube`.
  It will remove all axes of length 1. (`#669 <https://github.com/sunpy/ndcube/pull/669>`__)
- Added `~ndcube.NDCube.quantity` attribute to `~ndcube.NDCube` to return the data array with in the cube with the physical units stored in the `~ndcube.NDCube.unit` attribute. (`#677 <https://github.com/sunpy/ndcube/pull/677>`__)
- Enable `~ndcube.NDCube` to be raised to a power. (`#678 <https://github.com/sunpy/ndcube/pull/678>`__)
- Added `ndcube.NDCube.shape` as a replacement for "dimensions". (`#684 <https://github.com/sunpy/ndcube/pull/684>`__)
- Enable rtruediv on `~ndcube.NDCube` such that a user can now do 1/ `~ndcube.NDCube`. (`#685 <https://github.com/sunpy/ndcube/pull/685>`__)
- Add a ``keepdims=False`` kwarg to `~ndcube.NDCube.crop` and `~ndcube.NDCube.crop_by_values` setting to true keeps length-1 dimensions default behavior drops these dimensions. (`#732 <https://github.com/sunpy/ndcube/pull/732>`__)
- When calling :meth:`ndcube.NDCubeBase.axis_world_coords` or :meth:`ndcube.NDCubeBase.axis_world_coords_values` with a
  specific axis or axes specified, the methods now avoid doing calculations for any other uncorrelated axes, offering
  significant speedups when those other axes are large. (`#767 <https://github.com/sunpy/ndcube/pull/767>`__)
- Add support for passing ``-1`` as an element of the ``bin_shape`` argument to `.NDCube.rebin` which will not rebin along that dimension (sets the bin shape to equal the data shape). (`#777 <https://github.com/sunpy/ndcube/pull/777>`__)
- It is now possible to set the ``NDCube.data`` property of a cube with an array of the same shape and unit as the current cube. (`#795 <https://github.com/sunpy/ndcube/pull/795>`__)


Bug Fixes
---------

- :meth:`ndcube.NDCube.rebin` ``bin_shape`` argument now accepts a `astropy.units.Quantity` as input if the units are convertible to pixels. (`#744 <https://github.com/sunpy/ndcube/pull/744>`__)
- Fix support for astropy 7.0, this involved a change to ``CompoundLowLevelWCS`` so that in handles ``pixel_bounds`` if only one component WCS sets a pixel bound. (`#772 <https://github.com/sunpy/ndcube/pull/772>`__)
- Fixed a bug which caused ``NDCubeSequence.cube_like_dimensions`` to break. (`#788 <https://github.com/sunpy/ndcube/pull/788>`__)
- Added an internal code to shortcut non-correlated axes avoiding the creation of a full coordinate grid, reducing memory use in specific circumstances. (`#798 <https://github.com/sunpy/ndcube/pull/798>`__)


Documentation
-------------

- Added a gallery example  (:ref:`sphx_glr_generated_gallery_creating_even_spaced_wavelength_visualisation.py`) showcasing how to create a visualisation of unevenly spaced wavelength data cube using AIA data. (`#691 <https://github.com/sunpy/ndcube/pull/691>`__)
- Added a gallery example  (:ref:`sphx_glr_generated_gallery_creating_a_gwcs_from_quantities.py`) showcasing how to create a GWCS from quantities. (`#695 <https://github.com/sunpy/ndcube/pull/695>`__)


Internal Changes
----------------

- Move code creating a masked array for `ndcube.NDCube.rebin` to a private function which is then called by `ndcube.NDCube.rebin`. (`#705 <https://github.com/sunpy/ndcube/pull/705>`__)
- Added explicit support for numpy 2.0. (`#729 <https://github.com/sunpy/ndcube/pull/729>`__)


2.2.0 (2023-11-16)
==================

Backwards Incompatible Changes
------------------------------

- `ndcube.NDCollection.aligned_axes` attribute is now property. (`#607 <https://github.com/sunpy/ndcube/pull/607>`__)
- Dropped Python 3.8 support
- Increased minimum required for ``astropy`` to 5.0
- Increased minimum required for ``numpy`` to 1.21
- Increased minimum required for ``gwcs`` to 0.18. (`#636 <https://github.com/sunpy/ndcube/pull/636>`__)

Features
--------

- Provides `~ndcube.wcs.tools.unwrap_wcs_to_fitswcs`, a function to create a `astropy.wcs.WCS` instance equivalent to a sliced and/or resampled WCS instance.
  Only valid if the underlying implementation of the wrapped WCS instance is also an `astropy.wcs.WCS` instance. (`#649 <https://github.com/sunpy/ndcube/pull/649>`__)

Bug Fixes
---------

- Fix docstring causing doc build problems for some users. (`#600 <https://github.com/sunpy/ndcube/pull/600>`__)
- Fixes a bug where passing a string representation of a unit to `ndcube.NDCube.to` raised a `TypeError`. (`#605 <https://github.com/sunpy/ndcube/pull/605>`__)
- Keywords for ``sequence_axis_coords`` and ``sequence_axis_unit`` were not passed to the ``SequenceAnimator`` constructor.
  This is now fixed. (`#607 <https://github.com/sunpy/ndcube/pull/607>`__)
- Adds ``None`` check on ``self.aligned_axes`` to ``NDCollection.copy()`` and ``NDCollection.pop()`` which raised a ``AttributeError`` (`#646 <https://github.com/sunpy/ndcube/pull/646>`__)

Trivial/Internal Changes
------------------------

- Updated the error messages when missing either ``matplotlib`` or ``mpl_animators`` when creating a plot. (`#651 <https://github.com/sunpy/ndcube/pull/651>`__)

v2.1.0 (2023-03-01)
===================

Backwards Incompatible Changes
------------------------------

- To support compatibility with reproject 0.9, the API of `.NDCube.reproject_to`
  has been tweaked so that any keyword argument to the underlying reprojection
  function can be passed through. This has the effect of being a breaking change
  if you were specifying any arguments after ``shape_out=`` as positional rather
  than keyword arguments. (Note that in a future release we will probably change
  to require keyword arguments to ``reproject_to``. (`#552 <https://github.com/sunpy/ndcube/pull/552>`__)


Features
--------

- Implement new property, `ndcube.ExtraCoords.is_empty` that returns ``True`` if the object has got extra coords.  Otherwise return ``False``. (`#450 <https://github.com/sunpy/ndcube/pull/450>`__)
- Add `ndcube.ExtraCoords.resample` method to resample extra coordinates by a certain factor in each array dimension. (`#450 <https://github.com/sunpy/ndcube/pull/450>`__)
- Implement a new :meth:`ndcube.NDCube.rebin` method to combine integer numbers of pixels along each axis into larger pixels. (`#450 <https://github.com/sunpy/ndcube/pull/450>`__)
- Add new methods to interpolate lookup table coordinates: ``ndcube.extra_coords.table_coord.QuantityTableCoordinate.interpolate``, ``ndcube.extra_coords.table_coord.SkyCoordTableCoordinate.interpolate``, ``ndcube.extra_coords.table_coord.TimeTableCoordinate.interpolate``, `ndcube.extra_coords.table_coord.MultipleTableCoordinate.interpolate` (`#450 <https://github.com/sunpy/ndcube/pull/450>`__)
- Add `ndcube.NDCubeSequence.crop` and `ndcube.NDCubeSequence.crop_by_values` methods which crop the `~ndcube.NDCubeSequence` based on input world coordinate ranges. (`#466 <https://github.com/sunpy/ndcube/pull/466>`__)
- Add basic arithmetic methods between `~ndcube.NDCube` objects and broadcastable arrays,
  scalars, and `~astropy.units.Quantity` objects. Operations between two `~ndcube.NDCube` objects
  are not supported. (`#541 <https://github.com/sunpy/ndcube/pull/541>`__)
- Add `ndcube.NDCube.to` to convert cube to new unit. (`#586 <https://github.com/sunpy/ndcube/pull/586>`__)
- Created `~ndcube.GlobalCoordsABC` and updated `~ndcube.ndcube.NDCubeABC`, and `~ndcube.ExtraCoordsABC` to reflect official NDCube 2 API definition in SEP. (`#592 <https://github.com/sunpy/ndcube/pull/592>`__)


Bug Fixes
---------

- Fix bug #535 where `~ndcube.NDCollection` could not update when ``aligned_axes`` is `None` (`#538 <https://github.com/sunpy/ndcube/pull/538>`__)
- Fix a bug where ``aligned_axis_physical_types`` caused ``__str__``
  to error when ``aligned_axes`` was `None`. (`#539 <https://github.com/sunpy/ndcube/pull/539>`__)
- Fix a bug where ``data_unit`` was not being correctly passed through to the underlying plotting
  function when animating a cube. (`#578 <https://github.com/sunpy/ndcube/pull/578>`__)


Improved Documentation
----------------------

- Add example to example gallery of how to create an NDCube from a FITS file. (`#544 <https://github.com/sunpy/ndcube/pull/544>`__)


v2.0.3 (2022-09-23)
===================

Bug Fixes
---------

- Dynamically copy docstring and function signature from ``ndcube.NDCube.plotter.plot`` to :meth:`ndcube.NDCube.plot`. (`#534 <https://github.com/sunpy/ndcube/pull/534>`__)
- Fixed a bug where the ``plot_axes`` key was not respected when passing ``axes`` to ``plot``
  for 2D cubes. (`#551 <https://github.com/sunpy/ndcube/pull/551>`__)
- Limit maximum reproject version to 0.9 due to API changes. ndcube 2.1 will support the
  new reproject keyword arguments. (`#564 <https://github.com/sunpy/ndcube/pull/564>`__)


v2.0.2 (2022-05-10)
===================

Bug Fixes
---------

- Fix a bug in the ``NDCube._as_mpl_axes`` implementation, allowing cubes with
  compatible dimensions to be passed as the ``projection=`` keyword argument to
  certain matplotlib functions again. (`#509 <https://github.com/sunpy/ndcube/pull/509>`__)


Trivial/Internal Changes
------------------------

- Remove use of deprecated ``distutils`` module. (`#520 <https://github.com/sunpy/ndcube/pull/520>`__)


2.0.1 (2021-11-19)
==================

Bug Fixes
---------

- Enable `~ndcube.NDCollection` to accept aligned axes inputs in any integer type. (`#495 <https://github.com/sunpy/ndcube/pull/495>`__)
- Patch to convert quantity objects passed to ``crop_by_coords`` to the units given in the ``wcs.world_axis_units``. (`#497 <https://github.com/sunpy/ndcube/pull/497>`__)
- Fix a bug which prevented the ``axes_units=`` kwarg from working when using the
  matplotlib animators. (`#498 <https://github.com/sunpy/ndcube/pull/498>`__)
- Add support for length-1 lookup table coords within extra coords. (`#499 <https://github.com/sunpy/ndcube/pull/499>`__)
- Bump the minimum version of astropy to 4.2 to correctly support capturing
  dropped world dimensions into global coords when slicing the WCS. (`#500 <https://github.com/sunpy/ndcube/pull/500>`__)


2.0.0 (2021-10-29)
==================

Backwards Incompatible Changes
------------------------------

- Remove unused util functions and the ndcube WCS class.  Refactor util functions for converting between between data and WCS indices to reflect the APE14 nomenclature that distinguishes between array, pixel and world axes. (`#280 <https://github.com/sunpy/ndcube/pull/280>`__)
- NDCubeSequence animation axes can no longer be set by extra coords. (`#294 <https://github.com/sunpy/ndcube/pull/294>`__)
- ImageAnimatorNDCubeSequence, ImageAnimatorCubeLikeNDCubeSequence, LineAnimatorNDCubeSequence and LineAnimatorCubeLikeNDCubeSequence have been removed and replaced by NDCubeSequenceAnimator. (`#294 <https://github.com/sunpy/ndcube/pull/294>`__)
- Change type of output of ``ndcube.NDCollection.aligned_world_axis_physical_types`` from tuple to list. This is to be consistent with output of `astropy.wcs.WCS.world_axis_physical_types`. (`#302 <https://github.com/sunpy/ndcube/pull/302>`__)
- Change output type when common axis item is a slice that covers only one subcube. Previously this would return an NDCube. Now an NDCubeSequence is always returned unless the common axis item is an integer. Also, refactor NDCubeSequence.index_as_cube so codebase is simpler. (`#311 <https://github.com/sunpy/ndcube/pull/311>`__)
- Replace NDCube.crop_by_coords and NDCube.crop_by_extra_coords with new method, NDCube.crop (`#316 <https://github.com/sunpy/ndcube/pull/316>`__)
- Remove NDCubeSequence plotting. (`#322 <https://github.com/sunpy/ndcube/pull/322>`__)
- Update `.NDCube.array_axis_physical_types` return physical types from extra coords as well as the WCS. (`#338 <https://github.com/sunpy/ndcube/pull/338>`__)
- Rename `ndcube.ExtraCoords.add` method from previous name "add_coordinate". (`#394 <https://github.com/sunpy/ndcube/pull/394>`__)
- The `~ndcube.NDCube` object no longer inherits from `astropy.nddata.NDArithmeticMixin` as the methods were not coordinate aware. (`#457 <https://github.com/sunpy/ndcube/pull/457>`__)


Deprecations and Removals
-------------------------

- Remove ``NDCube.pixel_to_world`` and ``NDCube.world_to_pixel``. (`#300 <https://github.com/sunpy/ndcube/pull/300>`__)
- Remove ``world_axis_physical_types`` methods from `~ndcube.NDCube` and  `~ndcube.NDCubeSequence`. (`#302 <https://github.com/sunpy/ndcube/pull/302>`__)
- Remove ``NDCubeSequence.sequence_axis_extra_coords``. This is replaced by ``NDCubeSequence.sequence_axis_coords``. (`#335 <https://github.com/sunpy/ndcube/pull/335>`__)
- Remove ``ndcube.NDCubeSequence.common_axis_extra_coords``.  Will be replaced by `ndcube.NDCubeSequence.common_axis_coords`. (`#344 <https://github.com/sunpy/ndcube/pull/344>`__)
- Remove ``NDCollection.aligned_world_axis_physical_types``.  It will be replaced by `~ndcube.NDCollection.aligned_axis_physical_types`. (`#347 <https://github.com/sunpy/ndcube/pull/347>`__)


Features
--------

- Implement a new `ndcube.ExtraCoords` class which allows the specification of extra coordinates via lookup tables or WCS. This class exposes the extra coords as an APE 14 WCS object. (`#271 <https://github.com/sunpy/ndcube/pull/271>`__)
- Add new method, ``ndcube.NDCube.axis_world_coord_values``, to return world coords for all pixels for all axes in WCS as quantity objects. (`#279 <https://github.com/sunpy/ndcube/pull/279>`__)
- Added a new method `ndcube.NDCube.array_axis_physical_types` to show which physical types are associated with each array axis. (`#281 <https://github.com/sunpy/ndcube/pull/281>`__)
- Add properties to NDCubeSequence giving the world physical types for each array axis. (`#301 <https://github.com/sunpy/ndcube/pull/301>`__)
- Add as_mpl_axes method to NDCube plotting mixin so the an NDCube can be provided to astropy WCSAxes as a projection. (`#314 <https://github.com/sunpy/ndcube/pull/314>`__)
- Make pyplot colorbar work with the output on NDCube.plot when it is a 2D image. (`#314 <https://github.com/sunpy/ndcube/pull/314>`__)
- Introduce a new class, `~ndcube.global_coords.GlobalCoords`, for holding scalar coordinates that don't apply to any pixel axes. (`#323 <https://github.com/sunpy/ndcube/pull/323>`__)
- Implement `ndcube.NDCube.axis_world_coords` which returns high level coordinate
  objects for all, or a subset of, axes. (`#327 <https://github.com/sunpy/ndcube/pull/327>`__)
- New property, NDCubeSequence.sequence_axis_coords creates lists of GlobalCoords from each NDCube in the sequence.  This replaces NDCubeSequence.sequence_axis_extra_coords, but because it uses the GlobaCoords infrastructure, can handle more than just coords that began as extra coords. (`#335 <https://github.com/sunpy/ndcube/pull/335>`__)
- Implement `ndcube.NDCubeSequence.common_axis_coords` to replace ``~ndcube.NDCubeSequence.common_axis_extra_coords``. In contrast to old property, this new property collates coordinates from the wcs as well as extra_coords. (`#344 <https://github.com/sunpy/ndcube/pull/344>`__)
- New property, `ndcube.NDCollection.aligned_axis_physical_types`.  This replaces ``~ndcube.NDCollection.aligned_world_axis_physical_types`` and returns a list of tuples, where each tuple gives the physical types common between all members of the collection for a given aligned axis. (`#347 <https://github.com/sunpy/ndcube/pull/347>`__)
- Allow `ndcube.NDCubeSequence.explode_along_axis` to explode sequence along any axis, not just the common axis. (`#358 <https://github.com/sunpy/ndcube/pull/358>`__)
- Plotting functionality on `~ndcube.NDCube` has been refactored to use pluggable
  "plotter" classes. All plotting functionality can now be accessed via the
  `ndcube.NDCube.plotter` attribute, with `ndcube.NDCube.plot` becoming an alias for ``ndcube.NDCube.plotter.plot``.

  Advanced users, or package maintainers that which to customise the plotting
  functionality of an `~ndcube.NDCube` instance can set the ``.plotter`` attribute of
  a cube to be a subclass of `ndcube.visualization.BasePlotter` which then
  customises the behaviour of the :meth:`ndcube.NDCube.plot` method and provides any other
  methods implemented on the plotter. (`#401 <https://github.com/sunpy/ndcube/pull/401>`__)
- Preserve sliced-out coordinates from WCS in the GlobalCoords instance. (`#402 <https://github.com/sunpy/ndcube/pull/402>`__)
- Enable instantiating an NDCube from an existing NDCube by copying extra/global coords. (`#404 <https://github.com/sunpy/ndcube/pull/404>`__)
- Support exposing dropped dimensions when `~ndcube.ExtraCoords` is sliced. (`#411 <https://github.com/sunpy/ndcube/pull/411>`__)
- `~ndcube.ExtraCoords` is now explicitly limited to one dimensional tables because of a limitation in our use of `astropy.modeling`. (`#414 <https://github.com/sunpy/ndcube/pull/414>`__)
- Adds functionality to reproject an `~.NDCube` object to coordinates described by another WCS or FITS Header by calling the new `~.NDCube.reproject_to` method. (`#434 <https://github.com/sunpy/ndcube/pull/434>`__)
- Change the ``edges=`` keyword to ``pixel_corners=`` in
  :meth:`ndcube.NDCube.axis_world_coords` and `ndcube.NDCube.axis_world_coords_values` to make its
  meaning clearer based on SEP feedback. (`#437 <https://github.com/sunpy/ndcube/pull/437>`__)
- `~.NDCube.axis_world_coords` and `~.NDCube.axis_world_coords_values` now use a different, substantially faster and more memory efficient algorithm to generate the coordinates along all axes. (`#442 <https://github.com/sunpy/ndcube/pull/442>`__)
- Extends `ndcube.NDCube.reproject_to` functionality by supporting ``adaptive`` and ``exact`` algorithms for an `~ndcube.NDCube` with 2D celestial WCS. (`#448 <https://github.com/sunpy/ndcube/pull/448>`__)
- Introduce optional offset between old and new pixel grids in `ndcube.wcs.wrappers.resampled_wcs.ResampledLowLevelWCS`. (`#449 <https://github.com/sunpy/ndcube/pull/449>`__)
- `ndcube.ExtraCoords.from_lookup_tables` accepts (a sequence of) ``physical_types`` as kwarg to set the types of its ``lookup_tables``. (`#451 <https://github.com/sunpy/ndcube/pull/451>`__)
- Create new plotter class for animating `~ndcube.NDCubeSequence` is the 2.0 framework. This class always sets the sequence axis as a slider and leverages `ndcube.NDCube.plot`. (`#456 <https://github.com/sunpy/ndcube/pull/456>`__)
- Add ``__len__`` method to `~ndcube.NDCubeSequence` which makes ``len(sequence)`` return the number of cubes in the sequence. (`#464 <https://github.com/sunpy/ndcube/pull/464>`__)
- Add ``__iter__`` method to `~ndcube.NDCubeSequence` which iterates through the cubes within the sequence. (`#465 <https://github.com/sunpy/ndcube/pull/465>`__)
- Add property to `~ndcube.ExtraCoords` that returns a WCS of extra coords that describes all axes of associated cube. (`#472 <https://github.com/sunpy/ndcube/pull/472>`__)


Bug Fixes
---------

- Fix `ndcube.NDCollection.aligned_dimensions` so it does not crash when components of collection are NDCubeSequences. (`#264 <https://github.com/sunpy/ndcube/pull/264>`__)
- Generalize int type checking so it is independent of the bit-type of the OS. (`#269 <https://github.com/sunpy/ndcube/pull/269>`__)
- Fix ``axis_world_coord_values`` when the WCS is 1D and ensure it always returns
  Quantities (`#287 <https://github.com/sunpy/ndcube/pull/287>`__)
- Change name of ``NDCube.axis_world_coord_values`` to ``NDCube.axis_world_coords_values`` to be consistent with NDCube.axis_world_coords (`#293 <https://github.com/sunpy/ndcube/pull/293>`__)
- Remove NDCubeSequence animation dependence of deprecated sunpy ImageAnimator and LineAnimator classes in favour of ArrayAnimatorWCS class. (`#294 <https://github.com/sunpy/ndcube/pull/294>`__)
- Fix bug whereby common axis was not updated appropriately when slicing an NDCubeSequence. (`#310 <https://github.com/sunpy/ndcube/pull/310>`__)
- Fix bug in ``NDCube.axis_world_coords_values`` when number of pixel and world dimensions differ. (`#319 <https://github.com/sunpy/ndcube/pull/319>`__)
- Fixes bug in `~ndcube.utils.wcs.array_indices_for_world_objects` when the WCS input does not have a world_axis_object_components attribute. The fix causes the low_level_wcs version is tried before the code fails. This enables `ndcube.NDCube.combined_wcs` to be used with this function. (`#344 <https://github.com/sunpy/ndcube/pull/344>`__)
- Fixes IndexError in `~ndcube.utils.wcs.array_indices_for_world_objects` which occurred when some of the world axes are dependent. (`#344 <https://github.com/sunpy/ndcube/pull/344>`__)
- Stop `ndcube.NDCube.explode_along_axis` setting a common axis to the output `~ndcube.NDCubeSequence`.  The output sequence should have no common axis. (`#358 <https://github.com/sunpy/ndcube/pull/358>`__)
- Enable 2-D NDCubes to be visualized as a 1-D animated line. (`#381 <https://github.com/sunpy/ndcube/pull/381>`__)
- Ensure corner inputs to :meth:`ndcube.NDCube.crop` are converted to units stored in WCS as `~astropy.wcs.WCS.world_to_array_index_values` does not handle units. (`#382 <https://github.com/sunpy/ndcube/pull/382>`__)
- updated ndcube github repository link in "ndcube.docs.installation.rst". (`#392 <https://github.com/sunpy/ndcube/pull/392>`__)
- Fix bug in NDCube.axis_world_coords_values when axes_coords is initially a
  bare astropy coordinate object rather than a list/tuple of coordinate objects. (`#400 <https://github.com/sunpy/ndcube/pull/400>`__)
- Change the implementation of `.NDCube.crop` so that it takes into account all
  the corners of the world region specified by the upper and lower corners, not
  just those two points. (`#438 <https://github.com/sunpy/ndcube/pull/438>`__)
- Ensure `~ndcube.NDCube` init forces WCS to become high level.

  This patches a bug in astropy. (`#447 <https://github.com/sunpy/ndcube/pull/447>`__)
- Fix bug in `~ndcube.NDCube.axis_world_coords_values` which caused the units to be stripped when an ``axes`` input was given. (`#461 <https://github.com/sunpy/ndcube/pull/461>`__)
- Fix bug in `~ndcube.utils.wcs.get_dependent_world_axes` where an erroneous matrix transpose caused an error for non-square axis correlation matrices and wrong results for diagonally non-symmetric ones. (`#471 <https://github.com/sunpy/ndcube/pull/471>`__)
- Extend support for cropping an `~ndcube.NDCube` using an `~ndcube.ExtraCoords` instance as the wcs. (`#472 <https://github.com/sunpy/ndcube/pull/472>`__)
- Fix check as to whether user inputs to `ndcube.wcs.wrappers.compound_wcs.CompoundLowLevelWCS.world_to_pixel_values` result in consistent pixel values when world dimensions share pixel dimensions.  Previously this check was unreliable when non-trivial mapping between world and pixel dimensions was used. (`#472 <https://github.com/sunpy/ndcube/pull/472>`__)
- Fix slicing `~ndcube.ExtraCoords` made of lookup tables. This bug meant that mapping of coords to arrays axes was not adjusted when an axis was dropped. (`#482 <https://github.com/sunpy/ndcube/pull/482>`__)


Improved Documentation
----------------------

- Document accepted input to ``lookup_table`` in `~ndcube.ExtraCoords` setting its ``physical_types``. (`#451 <https://github.com/sunpy/ndcube/pull/451>`__)
- Improved information and formatting of ``__str__`` methods. (`#453 <https://github.com/sunpy/ndcube/pull/453>`__)


Trivial/Internal Changes
------------------------

- Simplify and speed up implementation of NDCubeSequence slicing. (`#251 <https://github.com/sunpy/ndcube/pull/251>`__)
- Fix docstring formatting to help docs build. (`#262 <https://github.com/sunpy/ndcube/pull/262>`__)
- Use pytest-mpl for figure tests. (`#312 <https://github.com/sunpy/ndcube/pull/312>`__)
- Port the tests for NDCube to use pytest fixtures (`#318 <https://github.com/sunpy/ndcube/pull/318>`__)
- Allow corner inputs to :meth:`~ndcube.NDCube.crop` to not be wrapped in a `tuple` is only one high level coordinate objects required. (`#380 <https://github.com/sunpy/ndcube/pull/380>`__)
- Make sunpy an optional dependence. Without it, the _animate_cube plotting
  functionality will be disabled. (`#393 <https://github.com/sunpy/ndcube/pull/393>`__)
- Adds a function to compare the physical types of two WCS objects. (`#433 <https://github.com/sunpy/ndcube/pull/433>`__)
- Propagate reference to NDCube object through `~ndcube.ExtraCoords` string slicing. (`#454 <https://github.com/sunpy/ndcube/pull/454>`__)
- Adds a function to identify invariant axes between two WCS objects. (`#459 <https://github.com/sunpy/ndcube/pull/459>`__)
- The matplotlib animators code has been moved from `sunpy` to a new package
  `mpl_animators` so ndcube no longer has an optional dependency on sunpy. (`#484 <https://github.com/sunpy/ndcube/pull/484>`__)


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
- Added a feature to get the pixel_edges from :meth:`ndcube.NDCube.axis_world_coords` (`#174 <https://github.com/sunpy/ndcube/pull/174>`__)


Bug Fixes
---------

- `ndcube.NDCube.wcs.world_axis_physical_types <astropy.wcs.wcsapi.BaseWCSWrapper>` now sets the axis label to the WCS CTYPE if no corresponding IVOA name can be found. (`#164 <https://github.com/sunpy/ndcube/pull/164>`__)
- Fixed the bug of using ``pixel_edges`` instead of ``pixel_values`` in plotting (`#176 <https://github.com/sunpy/ndcube/pull/176>`__)
- Fix 2D plotting from crashing when both data and WCS are 2D. (`#182 <https://github.com/sunpy/ndcube/pull/182>`__)
- Fix the ability to pass a custom Axes to `ndcube.NDCube.plot` for a 2D cube. (`#204 <https://github.com/sunpy/ndcube/pull/204>`__)


Trivial/Internal Changes
------------------------

- Include more helpful error when invalid item type is used to slice an `~ndcube.NDCube`. (`#158 <https://github.com/sunpy/ndcube/pull/158>`__)


1.1
===

API-Breaking Changes
--------------------
- ``~ndcube.NDCubeBase.crop_by_extra_coord`` API has been broken and
  replaced.
  Old version:
  ``crop_by_extra_coord(min_coord_value, interval_width, coord_name)``.
  New version:
  ``crop_by_extra_coord(coord_name, min_coord_value,  max_coord_value)``.
  [#142]

New Features
------------
- Created a new `~ndcube.NDCubeBase` which has all the functionality
  of `~ndcube.NDCube` except the plotting.  The old ``NDCubeBase``
  which outlined the `ndcube.NDCube` API was renamed ``NDCubeABC``.
  ``~ndcube.NDCube`` has all the same functionality as before except is
  now simply inherits from ``~ndcube.NDCubeBase`` and
  ``~ndcube.mixins.plotting.NDCubePlotMixin``. [#101]
- Moved NDCubSequence plotting to a new mixin class,
  NDCubSequencePlotMixin, making the plotting an optional extra.  All
  the non-plotting functionality now lives in the NDCubeSequenceBase
  class. [#98]
- Created a new ``~ndcube.NDCubeBase.explode_along_axis`` method that
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
- ``~ndcube.NDCubeBase.world_axis_physical_types`` is now not
  case-sensitive to the CTYPE values in the WCS. [#109]
- ``~ndcube.NDCubeBase.plot`` now generates a 1-D line animation when
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
