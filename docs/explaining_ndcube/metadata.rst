.. _ndmeta:

*****************
Handling Metadata
*****************

`ndcube`'s data objects do not enforce any requirements on the object assigned to their ``.meta`` attributes.
However, it does provide an optional object for handling metadata, `~ndcube.NDMeta`, with capabilities beyond a plain `dict`.
Chief among these is the ability to associate metadata with data array axes.
This enables `~ndcube.NDMeta` to update itself through certain operations, e.g. slicing, so that the metadata remains consistent with the associated ND object.
In this section, we explain the needs that `~ndcube.NDMeta` serves, the concepts underpinning it, and the functionalities it provides.

.. _meta_concepts:

Key Concepts
============

.. _coords_vs_meta:

Coordinates vs. Axis-aware Metadata
-----------------------------------

The difference between coordinates and axis-aware metadata is a subtle but important one.
Formally, a coordinate is a physical space sampled by one or more data axis, whereas axis-aware metadata is information describing the data that can alter along one or more physical dimension.
An informative example is the difference between time and exposure time.
The temporal axis of a 3-D image cube samples the physical dimension of time in a strictly increasing monotonic way.
The times along the temporal axis are therefore coordinate values, not metadata.
Additionally, a scalar timestamp of a 2-D image is also considered a coordinate in the `ndcube` framework.
This is because it describes where in the physical dimension of time the data has been sampled.
The fact that it's not associated with an array/pixel axis of the data does not change this.
it does, however, determine that the scalar coordinate is stored in `~ndcube.GlobalCoords`, rather than the WCS or `~ndcube.ExtraCoords`.
(See the :ref:`global_coords` section for more discussion on the difference between global and other coordinates.)
By contrast, exposure time describes the interval over which each image was accumulated.
Exposure time can remain constant, increase or decrease with time, and may switch between these during the time extent of the image cube.
Like a coordinate, it should be associated with the image cube's temporal axis.
However, exposure time is reflective of the telescope's operational mode, not a sampling of a physical dimension.
Exposure time is therefore metadata, not a coordinate.

One reason why it is important for `ndcube` to distinguish between coordinates and axis-aware metadata is its dependence on WCS.
Most WCS implementations require that there be a unique invertible mapping between pixel and world coordinates, i.e., there is only one pixel value that corresponds to a specific real world value (or combination of such if the coordinate is multi-dimensionsal), and vice versa.
Therefore, while there may be exceptions for rare and exotic WCS implementations, a good rule of thumb for deciding whether something is a coordinate is:
coordinates are numeric and strictly monotonic.
If either of these characteristics do not apply, you have metadata.

.. _axis_and_grid_aligned_meta:

Types of Axis-aware Metadata: Axis-aligned vs. Grid-aligned
-----------------------------------------------------------

There are two types of axis-aware metadata: axis-aligned and grid-aligned.
Axis-aligned metadata assigned a scalar or string to each of mutliple array azes.
For example, the data produced by a scanning slit spectrograph is associated with real world values.
But each axis also corresponds to features of the instrument: dispersion (spectral), pixels along the slit (spatial), position of the slit in the rastering sequence (spatial and short timescales), and the raster number (longer timescales).
The axis-aligned metadata concept allows us to avoid ambiguity by assigning each axis with a label (e.g. ``("dispersion", "slit", "slit step", "raster")``).

By contrast, grid aligned metadata assigns a value to each pixel along axes.
The exposure time discussion above is an example of 1-D grid-aligned metadata.
However, grid-aligned metadata can also be multi-dimensional.
For example, a pixel-dependent response function could be represented as grid-aligned metadata associated with 2 spatial axes.

`~ndcube.NDMeta` supports both axis-aligned and grid-aligned metadata with the same API, which will be discussed in the next section.

.. _ndmeta:


NDMeta
======
`~ndcube.NDMeta` is a `dict`-based object for handling metadata that provides a additional functionalities beyond those of a plain `dict`.
Chief among these are the ability to support axis-aware metadata (see :ref:`key_concepts`) and assigning comments to individual pieces of metadata.

Initializing an NDMeta
----------------------

