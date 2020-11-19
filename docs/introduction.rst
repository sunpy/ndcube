=========================
An Introduction to ndcube
=========================

N-dimensional Data in Astronomy
-------------------------------
N-dimensional data sets are common in all areas of science and beyond.  For example, a series of images taken sequentially with a CCD camera can be stored as a single 3-D array with two spatial axes and one temporal axis.  Each array-element can represent the reading in a pixel at a given time.  In astronomy, the relationship between the pixel coordinate and the location and time in the Universe being observed is often represented by the World Coordinate System (WCS) framework.  WCS's ability to handle many different types (e.g. spatial, temporal, spectral, etc.) of transformations make it a succinct, standardized and powerful way to relate pixels from an observation or cells in a simulation grid to the location in the Universe to which they correspond.  Due of the prevalence of N-D data and the importance of the transformations, there exist mature scientific Python packages (e.g. numpy and astropy) that contain powerful tools to handle N-D arrays and WCS transformations.

What is ndcube?
---------------
ndcube is a free, open-source, community-developed Python package whose purpose is to provide an interface uniting N-dimensional data arrays with the coordinate information that describes them.
There are many Python tools that provide half of these functionalities, e.g. array handling by numpy and dask or coordinate transformations by astropy's WCS and coordinates modules.
If you have only one of these components then these existing tools work well for you.
However, the value of ndcube is in enabling array-like operations that modify both the data array and coordinate system together.

ndcube provides a its features via two primary data classes: `~ndcube.NDCube` and `~ndcube.NDCubeSequence`.
The former is for managing a single array and set of WCS transformations, while the latter is for handling multiple arrays, each described by their own set of WCS transformations.
These classes provide unified slicing, visualization, coordinate conversion APIs as well as APIs for inspecting the data, coordinate transformations and metadata.
ndcube does this in a way that is not specific to any number or physical type of axis, and so can in principle be used for any type of data (e.g. images, spectra, timeseries, etc.) so long as the data are represented by an array and a set of WCS transformations.
Moreover, ndcube is agnostic to the fundamental array type in which the data is stored, so long as it behaves like a numpy array.
Meanwhile, the WCS object can be any class, as long as it adhere's to the AstroPy `wcsapi (APE 14) <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ specification.
These features make ndcube's data classes ideal to subclass when creating tools for specific types of data, while keeping the non-data-specific functionalities (e.g. slicing) common between classes.

Why ndcube?
-----------
The most similar project to ndcube is xarray and it’s worth addressing why ndcube exists in its own right, rather than devoting time and effort to building tools around xarray.
The fundamental reason to opt for ndcube is a requirement to harness the astronomy-specific World Coordinate System (WCS) tooling provided by packages like Astropy (which now also supports gWCS).
The data model of xarray centered on the requirements and conventions of the geosciences, which while being very similar to that of astronomy in conception, is sufficently different in construction to cause much friction.
Utilize the astropy WCS infrastructure enables us to directly read the most common file formats in astronomy (FITS), although the FITS WCS data model is also commonly used outside of FITS files.
This data model would require translation of the source data to fit inside an xarray object.

That being said, xarray has a richer feature set and there is nothing beyond a lack of developer time hindering the astronomy and xarray communities from collaborating to provide a common set of tools which would suit everyone’s purposes.
See for instance `this issue <https://github.com/pydata/xarray/issues/3620#>`_.

Why is ndcube 2.0?
------------------
ndcube 2.0 is a major API-breaking rewrite of ndcube.
It has been written to take advantage of many new features not available when ndcube 1.0 was written.
Some of these have been made possible by functionalities originally written for ndcube being upstreamed to astropy to improve the WCS support.
Others are the result of various long running projects maturing, such as the acceptance and implementation of astropy's WCS API (APE 14) and the maturing of the gWCS package.
These developments encouraged the reassesment of the state of ndcube and to rebase it onto this new functionality, leading to the development of ndcube 2.0.

The main feature of ndcube 2.0 is the removal of all the specific WCS handling code that was previously required in 1.0.
All WCS manipulation and slicing code has now been upstreamed to astropy and ndcube uses the generalised wcsapi functionality for all these features.
This has the consequence of bringing high-level coordinate objects into the realm of ndcube.
This includes astropy's SkyCoord object which combines coordinate and reference frame information to give users all the information they need to correctly interpret their coordinates.
However in many cases, users can continue to deal with raw coordinate values without reference frame information if they choose.
ndcube's visualisation code has been rewritten to exclusively apply WCSAxes, tremendously simplifying it’s implementation, at the expense of some flexibility.
However, it also allows for a more complete and accurate representation of coordinates along plot axes and animations.
Extra_coords has been completely re-written to serve as an extra WCS, which can be readily constructed from lookup tables.
This enables users to easily combine the extra_coords and WCS coords and to utilize the WCSAxes infrastructure for visualizing extra_coords in their plots.
