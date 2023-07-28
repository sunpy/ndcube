---
title: 'ndcube: Manipulating N-dimensional Astronomical Data in Python'
tags:
  - Python
  - astronomy
authors:
  - name: Daniel F. Ryan
    orcid: 0000-0001-8661-3825
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
    corresponding: true
  - name: Stuart Mumford^[co-first author] # note this makes a footnote saying 'co-first author'
    orcid: 0000-0003-4217-4642
    affiliation: 3
  - name: Yash Sharma
    affiliation: 4
  - name:  Ankit Kumar Baruah
    affiliation: 5
  - name: Adwait Bhope
    orcid: 0000-0002-7133-8776
    affiliation: 6
  - name: Nabil Freij
    orcid: 0000-0002-6253-082X
    affiliation: "7, 8"
  - name: Laura A. Hayes
    orcid: 0000-0002-6835-2390
    affiliation: 13
  - name: Will T. Barnes
    orcid: 0000-0001-6874-2594
    affiliation: "12, 2"
  - name: Baptiste Pellorce
    affiliation: "9, 10"
  - name: Richard O'Steen
    orcid: 0000-0002-2432-8946
    affiliation: 11
  - name: Derek Homeier
    orcid: 0000-0002-8546-9128
    affiliation: 3
  - name: J. Marcus Hughes
    orcid: 0000-0003-3410-7650
    affiliation: 14
  - name: David Stansby
    orcid: 0000-0002-1365-1908
    affiliation: 15
  - name: Albert Y. Shih
    orcid: 0000-0001-6874-2594
    affiliation: 12
  - name: Matthew J. West
    orcid: 0000-0002-0631-2393
    affiliation: 14
affiliations:
 - name: University of Applied Sciences Northwest Switzerland, Switzerland
   index: 1
 - name: American University, USA
   index: 2
 - name: Aperio Software Ltd, UK
   index: 3
 - name: Meta Platforms Inc., UK
   index: 4
 - name: Workato Gmbh, Germany
   index: 5
 - name: Uptycs India Pvt. Ltd., India
   index: 6
 - name: Lockheed Martin Solar and Astrophysics Laboratory, USA
   index: 7
 - name: Bay Area Environmental Research Institute, USA
   index: 8
 - name: Claude Bernard Lyon 1 University, France
   index: 9
 - name: Institute of Theoretical Astrophysics, Norway
   index: 10
 - name: Space Telescope Science Institute, USA
   index: 11
 - name: NASA Goddard Space Flight Center, USA
   index: 12
 - name: European Space Agency, ESTEC
   index: 13
 - name: Southwest Research Institute, USA
   index: 14
 - name: Advanced Research Computing Centre, University College London, UK
   index: 15
date: 24 February 2023
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/1538-4357/ace0bd
aas-journal: Astrophysical Journal
---

# Summary

ndcube is a free, open-source, community-developed Python package for inspecting,
manipulating, and visualizing n-dimensional coordinate-aware astronomical data.
Its features are agnostic to the number of data dimensions and the physical
coordinate types they represent.
Its data classes link data and their coordinates and provide analysis
methods to manipulate them self-consistently.
These aim to provide simple and intuitive ways of handling coordinate-aware data,
analogous to how users handle coordinate-agnostic data with arrays.
ndcube requires that coordinate transformations be expressed via the World
Coordinate System (WCS), a coordinate framework commonly used throughout
astronomy.
The WCS framework has multiple implementations (e.g. FITS-WCS, gWCS, and others),
each with a different incompatible API, which makes workflows and derived tools
non-transferable between implementations.
ndcube overcomes this by leveraging Astropy's WCS API [APE-14, @ape14]
which can be wrapped around any underlying WCS implementation.
This enables ndcube to use the same API to interact with any set of WCS transformations.
ndcube's data-WCS coupling allows users to analyze their data more easily and
reliably, thus helping to boost their scientific output.


# Statement of Need

N-dimensional data sets are common in all areas of science and beyond.
For example, a series of images taken sequentially with a CCD camera can be stored
as a single 3-D array with two spatial axes and one temporal axis.
In astronomy, the most commonly used framework for translating between array element
indices, and the location or time in the Universe being observed is the World
Coordinate System (WCS).
WCS's ability to handle many different physical types of coordinates (e.g. spatial,
temporal, spectral, etc.) and their projections onto a data array (e.g. right ascension
and declination, helioprojective latitude and longitude, etc.) make it a succinct,
standardized and powerful way to relate array axes to the physical coordinate types
they represent.

There are mature Python packages for handling N-D array operations --
for example, NumPy [@numpy], and Dask [@dask] -- and others for supporting WCS coordinate
transformations -- for example, Astropy [@astropy2013; @astropy2018; @astropy2022], and gWCS [@gwcs].
However, none treat data and coordinates in a combined, self-consistent way.
The closest alternative to ndcube is Xarray [@xarray].
However Xarray has been developed for the requirements and conventions of the
geosciences which, although similar to those of astronomy in concept, are sufficiently
different in construction to cause significant friction.
Crucially, Xarray does not currently support WCS coordinate transformations.
Tools that do support WCS-based coordinate-aware data analysis, such as the SunPy
[@sunpy] Map class for 2-D images of the Sun, tend to have APIs specific to particular
combinations of dimensions, physical types, coordinate systems and WCS implementations.
This limits their broader utility and makes the combined analysis of different types
of data more difficult.
It also inhibits collaboration by erecting technical barriers between sub-fields of
astronomy.

ndcube overcomes these challenges via its design policy that all functionalities and
APIs must be agnostic to the number of dimensions and coordinate types they represent.
Moreover, ndcube's employment of the Astropy WCS API makes it agnostic to the
underlying WCS implementation.


# The Role of ndcube and its Features

The ndcube package serves three specific purposes.
First, it formalizes the NDCube 2 API in software via its abstract base classes
(ABCs), `NDCubeABC`, `ExtraCoordsABC` and `GlobalCoordsABC`.
The NDCube 2 API is a standardized framework for inspecting and manipulating
coordinate-aware N-D astronomical data and is defined by 12th SunPy Enhancement
Proposal [SEP-12, @sep12].
A discussion of the philosophies underpinning the NDCube 2 API can be found
in @ndcube.
Second, the ndcube package implements the NDCube 2 API in corresponding data and
coordinate classes, `NDCubeBase`, `ExtraCoords` and `GlobalCoords`.
These are viable off-the-shelf tools for end users and developers who do
not want to implement the API themselves.
Third, it provides additional support for coordinate-aware manipulation and
visualization of N-D astronomical data via three high-level data classes:
`NDCube`, `NDCubeSequence` and `NDCollection`.
`NDCube` (note the different capitalization from the package name) inherits from
`NDCubeBase` and so adheres to the NDCube 2 API but adds some additional features,
such as a default visualization suite.
The other classes are designed to handle multiple `NDCube` instances simultaneously.

The features in the ndcube package are designed to be practical tools for end users.
But they are also powerful bases upon which to build tools for specific types of data.
This might be a specific number and/or combination of physical types
(spectrograms, image cubes, etc.), or data from specific instruments or simulations.
Thus, ndcube can enhance the productivity of developers by centralizing the
development and maintenance of the most useful and general functionalities.
This leaves more time for developing a greater range of tools for the community
and/or enables part-time developers to devote more effort to other aspects of their
jobs.

## High-level Data Classes

The three high-level data classes provided by the ndcube package are `NDCube`,
`NDCubeSequence` and `NDCollection`.
`NDCube` requires that the data is stored in a single array object and described by
a set of WCS transformations.
The array can be any object that exposes `.dtype` and `.shape` attributes and can
be sliced by the standard Python slicing API.
Thus `NDCube` not only supports NumPy arrays but also others such as Dask for
distributed computing [@dask], CuPy for GPU operations [@cupy], and others.
`NDCube` leverages the Astropy WCS API for interacting with and manipulating the WCS
transformations.
This means `NDCube` can support any WCS implementation (e.g. FITS-WCS, gWCS, and others),
so long as it's supplied in an Astropy-WCS-API-compliant object.
The components of an `NDCube` are supplied by the following keyword arguments and
accessed via attributes of the same name.

- `data`: The data array. (Required)
- `wcs`: The primary set of coordinate transformations. (Required)
- `uncertainty`: an `astropy.nddata.NDUncertainty` object giving the uncertainty of each element in the data array. (Optional)
- `mask`: a boolean array denoting which elements of the data array are reliable. A `True` value implies the data is masked, or unreliable. (Optional)
- `meta`: an object for storing metadata, (e.g. a Python dictionary). (Optional)
- `unit`: the unit of the data. (Optional)

`NDCube` also supports additional coordinate information. See the subsection on
Coordinate Classes.
`NDCube` provides several analysis methods such as slicing (by array indices),
cropping (by real-world coordinates), reprojecting to new WCS transformations,
visualization, rebinning data, arithmetic operations, and more.
All these methods manipulate the data, coordinates, and supporting data (e.g.
uncertainties) simultaneously and self-consistently.
This relieves users of well-defined, but tedious and error-prone tasks.

`NDCubeSequence` is designed to handle multiple `NDCube` instances that are arranged
in some order.
Cubes can be ordered along an additional axis to those represented by the cubes,
for example, a sequence of 2-D spatial images arranged along a 3rd time axis.
In this case, users can interact with the data as if it were a 3-D cube with a similar
API to `NDCube`.
Alternatively, the cubes can be ordered along one of the cubes' axes, for example, a
sequence of tiles in an image mosaic where each cube represents an adjacent region
of the sky.
`NDCubeSequence` provides APIs for both the (N+1)-D and extended N-D paradigms,
that are simultaneously available on each `NDCubeSequence` instance.
This enables users to switch between the paradigms without reformatting or copying
the underlying data.
`NDCubeSequence` also provides various methods to help with data analysis.
These APIs are similar to `NDCube` wherever possible (e.g. for slicing and visualization), to minimize friction between analyzing single and multiple cubes.

`NDCollection` allows unordered but related `NDCube` and `NDCubeSequence` objects
to be linked, similar to how a Python dictionary is used.
However, in addition to dictionary-like features, `NDCollection` allows axes of
different cubes with the same lengths to be marked as 'aligned'.
This enables these axes on all constituent cubes to be sliced at the `NDCollection`
level.
One application of this is linking derived data products, for example, a spectral image cube
and a Doppler map derived from one of its spectral lines.
Marking both cubes' spatial axes as 'aligned' and slicing the `NDCollection`
rather than the two cubes separately, simplifies the extraction of regions of
interest and guarantees both cubes continue to represent the same field of view.

More detailed discussion on the roles of the above data classes' features and
how to use them can be found in @ndcube and the ndcube documentation [@ndcube-docs].


## Coordinate Classes

The ndcube package provides two coordinate classes, `ExtraCoords` and
`GlobalCoords`.
`ExtraCoords` provides a mechanism for storing coordinate transformations
that are supplemental to the primary WCS transformations.
This can be very useful if, say, we have a spectral image cube whose images were
taken at slightly different times but whose WCS does not include time.
In this case, `ExtraCoords` can be used to associate an `astropy.time.Time` object
with the spectral axis without having to manually construct a new WCS which is
a potentially complicated task even for experienced users.
`ExtraCoords` supports both functional and lookup-table-based transformations.
It can therefore also be used as an alternative set of coordinate
transformations to those in the primary WCS and used interchangeably.

By contrast, `GlobalCoords` supports scalar coordinates that apply to the whole
`NDCube` rather than any of its axes, for example, the timestamp of a 2-D image.
Scalar coordinates are not supported by WCS because it requires all coordinates
to be associated with at least one array axis, hence the need for `GlobalCoords`.
When an axis is dropped from an `NDCube` via slicing, the values of the dropped
coordinates at the relevant location along the dropped axis are automatically
added to the associated `GlobalCoords` object, for example, the timestamp of a 2-D image
sliced from a 3-D space-space-time cube.
Thus coordinate information is never lost due to slicing.

`NDCube` objects are always instantiated with associated `ExtraCoords` and
`GlobalCoords` objects, even if empty.
Users can then add and remove coordinates subsequently.
For a more in-depth discussion of `ExtraCoords` and `GlobalCoords`, see @ndcube.


# Community Applications of ndcube

The importance of the ndcube package is demonstrated by the fact that it is
already a dependency of various software tools that support current ground-based
and satellite observatories.
These include the James Webb Space Telescope (JWST), Solar Orbiter,
the Interface Region Imaging Spectrograph (IRIS), Hinode, and the
Daniel K. Inouye Solar Telescope (DKIST) via the
specutils [@specutils-docs; @specutils-code], jdaviz [@jdaviz], sunraster [@sunraster],
irispy-lmsal [@irispy-docs; @irispy-code], EISPAC [@eispac-docs; @eispac-code] and
DKIST user tools packages which all depend on ndcube.
ndcube is also used in the data pipeline of the PUNCH mission [Polarimeter to UNify
the Corona and Heliosphere, @punch], scheduled for launch in 2025.
In addition, individual researchers are using the ndcube package in their own analysis
workflows.

A network benefit of ndcube is that it standardizes the APIs for handling
astronomical coordinate-aware N-D data.
Adoption across astronomy and heliophysics helps scientists to more easily work
with data from different missions and sub-communities.
This can simplify multi-instrument data analysis, foster inter-field collaborations,
and promote scientific innovation.


# Acknowledgements

We acknowledge financial support for ndcube from NASA's Heliophysics Data
Environment Enhancement program, the Daniel K. Inouye Solar Telescope, and
Solar Orbiter/SPICE (grant 80NSSC19K1000).
We also acknowledge the SunPy, Python in Heliophysics, and Astropy communities for
their contributions and support.

# References
