.. _ndcube_20_migration:

*********************************
Upgrading from ``1.x`` to ``2.x``
*********************************

Why 2.0?
========

ndcube 2.0 is a major API-breaking rewrite of ndcube.
It has been written to take advantage of many new features not available when ndcube 1.0 was written.
Some of these have been made possible by the moving some of the functionalities of ndcube into astropy.
Others are due to the fruition of long running projects such as the implementation of astropy's `WCS API (APE 14) <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`__ and the maturing of the `gWCS <https://gwcs.readthedocs.io/en/latest/>`__ package.
These developments encouraged the reassessment of the state of ndcube, leading to the development of ndcube 2.0.

The main feature of ndcube 2.0 is the removal and migration of almost all specific WCS handling code to astropy and the use of the astropy's WCS (APE-14) API.
This has the consequence of bringing high-level coordinate objects into the realm of ndcube.
This includes astropy's `~astropy.coordinates.SkyCoord` object which combines coordinate and reference frame information to give users a full description of their coordinates.

However users can continue to deal with raw coordinate values without reference frame information if they so choose.
ndcube's visualization code has been rewritten to exclusively use `~astropy.visualization.wcsaxes.WCSAxes`, tremendously simplifying it's implementation, at the expense of some flexibility.
However, it also allows for a more complete and accurate representation of coordinates along plot axes and animations.
`~ndcube.NDCube.extra_coords` has been completely re-written to serve as an extra WCS, which can be readily constructed from lookup tables.
This enables users to easily include the extra coordinates when visualizing the data.

Finally, a new `~ndcube.GlobalCoords` class can hold coordinates that do not refer to any axis.
This is particularly useful when the dimensionality of an `~ndcube.NDCube` is reduced by slicing.
The value of a coordinate at the location along the dropped axis at which the `~ndcube.NDCube` was sliced can be retained.


As discussed above, the ``ndcube`` 2.0 package aims to be a framework for data with an APE-14-compliant World Coordinate System object.
This large refactor and associated API changes means that if you are familiar with ``ndcube`` 1.x there is a lot which will be different.
This section aims to cover the main points, if you notice anything we have forgotten please `open an issue <https://github.com/sunpy/ndcube/issues/new/choose>`__.

Coordinates and WCS
===================

The type of the ``.wcs`` object
-------------------------------

In ``ndcube`` 1.x the ``NDCube.wcs`` property was always an instance of `astropy.wcs.WCS`.
This is no longer true **even if you pass such an instance to NDCube**.
The reason for this is that operations like slicing may change the type of the ``.wcs`` object to represent different views into the original WCS.

The ``.wcs`` property will always be an instance of `astropy.wcs.wcsapi.BaseHighLevelWCS`.
You should therefore adjust any code which needs to work with any `~ndcube.NDCube` object to only use this (and associated `~astropy.wcs.wcsapi.BaseLowLevelWCS`) APIs.

Future work in astropy or ndcube may increase the chances that the original WCS type be preserved, but it is highly unlikely that it will ever be possible to always carry the type of the WCS through all slicing operations.


No more ``.missing_axes``
-------------------------

As a corollary to the above, there is no longer a ``.missing_axes`` property on `~ndcube.NDCube` as all the slicing operations now happen inside the ``.wcs`` property inside astropy.


Dropped dimensions moved from ``.wcs`` to ``.global_coords``
------------------------------------------------------------

As another consequence of the WCS slicing, when dimensions are dropped those world coordinates are no longer accessible through the ``.wcs``.
To overcome this, and also to provide a structured place for future or custom cube-wide scalar coordinates, the ``.global_coords`` property was added.

``.global_coords`` will automatically be populated by any dimensions dropped via slicing the `~ndcube.NDCube`, via functionality in `~astropy.wcs.wcsapi.SlicedLowLevelWCS`.
Scalar coordinates can also be added to the ``.global_coords`` object explicitly using the :meth:`~ndcube.GlobalCoords.add` method.

The saga of ``extra_coords``
----------------------------

As part of the transition to using APE 14 compliant WCS objects everywhere we have transitioned ``.extra_coords`` to use ``gWCS`` underneath to provide a APE-14 compliant API to the extra coords lookup tables.
Due to the extra functionality and therefore complexity of the `~ndcube.ExtraCoords` object (over the previous `dict` implementation) the ``extra_coords=`` keyword argument has been removed from the `ndcube.NDCube` constructor.
Extra coordinates can be added individually using the :meth:`~ndcube.ExtraCoords.add` method on the ``.extra_coords`` property.

If you wish to build an `~ndcube.NDCube` object from lookup tables without a WCS object you might find the extra coords infrastructure useful.
This is documented in :ref:`tabular_coordinates`.

``.wcs``, ``.extra_coords`` and ``.combined_wcs``
-------------------------------------------------

There are now three different WCS-like properties on `~ndcube.NDCube`:

* ``.wcs``: The WCS object passed in through the constructor or a wrapper around it.
* ``.extra_coords``: A coordinate object that can be used in place of a WCS object in `~ndcube.NDCube` methods.
* ``.combined_wcs``: A WCS wrapper that combines the coordinates described by ``.wcs`` and ``.extra_coords`` into a single APE-14-compliant WCS object.

Various methods on `~ndcube.NDCube` now accept a ``wcs=`` keyword argument, which allows the use of any of these attributes, the default is still ``.wcs``.

In the future the default may change to ``.combined_wcs``.
However, there are various technical reasons why this hasn't been done in the initial release, such as a significant performance penalty of using ``.combined_wcs``.

`~ndcube.NDCube` Methods
========================

``crop_by_coords`` is now ``crop`` and ``crop_by_values``
---------------------------------------------------------

The old ``NDCube.crop_by_coords`` method has been replaced with two new methods `ndcube.NDCube.crop` and `ndcube.NDCube.crop_by_values`.
The new methods accept high-level (e.g. `~astropy.coordinates.SkyCoord`) objects and quantities respectively.
The new methods also use a different algorithm to ``crop_by_coords``, which has been selected to work with data of all dimensionality and coordinates.
Both the crop methods take N points as positional arguments where each point must have an entry for each world axis.
The cube will then be cropped to the smallest pixel box containing the input points.
Note that in this algorithm the input points are not interpreted as corners of a bounding box, although is some cases the result will be equivalent to that interpretation.
For more information see :ref:`ndcube_crop`.


``.world_to_pixel`` and ``.pixel_to_world`` removed
---------------------------------------------------

As part of the transition to relying on APE-14-compliant WCS objects ``NDCube.world_to_pixel`` and ``pixel_to_world`` are now redundant as the APE-14-WCS API specifies that the WCS object must provide these methods with equivalent functionality.
Therefore you should now use ``NDCube.wcs.pixel_to_world`` and ``NDCube.wcs.world_to_pixel``; in addition to this you can also make use of the ``_values`` or ``array_index`` variants of these methods (see `~astropy.wcs.wcsapi.BaseLowLevelWCS`).

Removed Arithmetic Operations
-----------------------------

During the rewrite the decision was taken for `ndcube.NDCube` not inherit the `astropy.nddata.NDArithmeticMixin` class.
The primary reason for this is that the operations supported by this mixin are not coordinate aware.
It is intended that in the future, `~ndcube.NDCube` will support operations such as add and subtract with scalars and array-like objects.
Future support for arithmetic operations between coordinate-aware objects will involve first checking that pixel grids are aligned.

Visualization Changes
=====================

The final major change in 2.0 is a rework of the built in visualization tooling in ndcube.
While the visualization code in 1.x was very powerful, that power came with a very high level of complexity, which made maintaining that functionality difficult.
When we were migrating ndcube to use the new WCS APIs we needed to modify large amounts of the existing visualization code, which just became untenable with the amount of time available.
We therefore took the decision to significantly reduce the scope of the built in visualization functionality.

The visualization code included in 2.0 only uses `~astropy.visualization.wcsaxes`, which means that **all plots are made in pixel space** with ticks and gridlines overplotted to show world coordinates.
This has dramatically simplified the code in ndcube, as almost all the complexity is now delegated to ``wcsaxes``.
In addition to this we have made it easier for users and developers to replace, customize, or disable the built in functionality by use of the ``.plotter`` attribute.
Learn more in :ref:`customizing_plotter`.
