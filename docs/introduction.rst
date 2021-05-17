=========================
An Introduction to ndcube
=========================

N-dimensional Data in Astronomy
===============================

N-dimensional data sets are common in all areas of science and beyond.
For example, a series of images taken sequentially with a CCD camera can be stored as a single 3-D array with two spatial axes and one temporal axis.
The value in each array-element can represent the reading in a pixel at a given time.
In astronomy, the relationship between the array element and the location and time in the Universe being observed is often represented by the World Coordinate System (WCS) framework.
WCS's ability to handle many different physical types (e.g. spatial, temporal, spectral, etc.) and projections (e.g. RA and Dec., helioprojective latitude and longitude, etc.) make it a succinct, standardized and powerful way to relate array axes to the physical properties they represent.

Due of the prevalence of N-D data and the importance of coordinate transformations, there exist mature Python packages that handle them.
For example, arrays can be handled by numpy and dask and coordinates by astropy's WCS and coordinates modules.
If you want to treat these components separately, then these existing tools work well.
However, they are not suited to treating data and coordinates in a combined way.

What is ndcube?
===============

ndcube is a free, open-source, community-developed Python package whose purpose is to link astronomical data and coordinates in single objects.
These objects can be manipulated via array-like slicing operations which modify both the data and coordinate systems simultaneously.
They also allow coordinate transformations to be performed with reference to the size of the data array and produce visualizations whose axes are automatically described by the coordinates.
This coupling of data and coordinates allows users to analyze their data more quickly and accurately, thus helping to boost their scientific output.

In this guide we will introduce you to ndcube's primary data classes, `~ndcube.NDCube`, `~ndcube.NDCubeSequence`, and `~ndcube.NDCollection` (:ref:`data_classes`).
We will then discuss the functionalities they provide including :ref:`coordinates` and :ref:`plotting`.
There are also helpful sections on :ref:`installation`, :ref:`getting_help` and :ref:`contributing`.

.. _axes_definitions:

Important Concepts: Array, Pixel, and World Axes
================================================

Throughout this guide we will refer to array axes, pixel axes and world axes, a nomenclature drawn from astropy.
To help the reader we will briefly clarify their meaning here.

A WCS object describes any number of physical types.
These are referred to as world axes and the order in which they are stored in the WCS object is referred to as world order (or world axis order).
These physical types are mapped through the WCS to one or more "pixel" axes.
Although in the simplest case, one world axis would uniquely map to one pixel axis, it is possible for multiple world axes can be associated with multiple pixel axes and vice versa.
This is especially common when dealing with projections of the sky onto 2-D image planes.
Take the example of an image of the Sun taken from Earth.
The world axis of helioprojective longitude is dependent on helioprojective latitude, i.e. there is not one pixel axis for longitude and another for latitude.
Both world axes are associated with both pixel axes.
In a WCS object, the mapping between pixel and world axes is described by the `~astropy.wcs.wcsapi.BaseLowLevelWCS.axis_correlation_matrix`.

Due to unfortunate convention, WCS orders its pixel axes in the inverse order to numpy.
Therefore we use the term "array axes" to refer to pixel axes which have been reversed to reflect the axis order of the numpy data array.
Take for example a numpy array with three dimensions.
Since the array axes are simply the reverse of the pixel axes, the first axis of the array corresponds to the 3rd pixel axis.
And the 2nd array axis corresponds to the 2nd pixel axis.
If the array had four axes, the first array axis would correspond to the fourth pixel axis and the second array axis would correspond to the third pixel axis and so on.

In ndcube, inputs and outputs are given in either array axis order or world axis order, depending on the types of information.
Throughout these docs and in the docstrings we will endeavor to highlight which order is relevant.
However a good rule of thumb is that if you are using a sequence of coordinate objects to describe locations in the data cube -- for example in the input of `ndcube.NDCube.crop` or the output of `ndcube.NDCube.axis_world_coords` -- they should be in world axis order.
In almost all other cases, array axis order is used.

Why ndcube?
===========

It's worth addressing the role ndcube plays within the scientific Python ecosystem and why it exists separately from its most similar package, xarray.
The fundamental reason to opt for ndcube is to harness the astronomy-specific World Coordinate System (WCS).
The data model of xarray centers on the requirements and conventions of the geosciences.
Although very similar to those of astronomy in conception, they are sufficiently different in construction to cause significant friction.
Moreover, utilizing the astropy WCS infrastructure enables us to directly read the most common file format in astronomy, FITS.
Although the FITS WCS data model is also commonly used outside of FITS files.
This data model would require translation of the source data to fit inside an xarray object.

That being said, xarray has a rich feature set and there is nothing beyond a lack of developer time hindering the astronomy and xarray communities from collaborating to provide a common set of tools which would suit everyone's purposes.
See `this issue <https://github.com/pydata/xarray/issues/3620#>`__.

Why ndcube 2.0?
===============

ndcube 2.0 is a major API-breaking rewrite of ndcube.
It has been written to take advantage of many new features not available when ndcube 1.0 was written.
Some of these have been made possible by the moving some of the functionalities of ndcube into astropy.
Others are due to the fruition of long running projects such as the implementation of astropy's `WCS API (APE 14) <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`__ and the maturing of the `gWCS <https://gwcs.readthedocs.io/en/latest/>`__ package.
These developments encouraged the reassessment of the state of ndcube, leading to the development of ndcube 2.0.

The main feature of ndcube 2.0 is the removal and moving of almost all specific WCS handling code to astropy and the use of the astropy's generalised WCS API.
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
