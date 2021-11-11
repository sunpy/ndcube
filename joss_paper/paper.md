---
title: 'ndcube: Manipulating N-dimensional Astronomical Data in Python'
tags:
  - Python
  - astronomy
authors:
  - name: Daniel F. Ryan^[co-first author] # note this makes a footnote saying 'co-first author'
    orcid: 0000-0001-8661-3825
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Stuart Mumford^[co-first author] # note this makes a footnote saying 'co-first author'
    affiliation: 2
  - name: Author with no affiliation^[corresponding author]
    affiliation: 3
affiliations:
 - name: University of Applied Sciences Northwest Switzerland
   index: 1
 - name: University of Sheffield
   index: 2
date: 11 December 2021
bibliography: paper.bib

---

# Summary

ndcube is a free, open-source, community-developed Python package whose purpose
is to faciliate the manipulation and visualization of astronomical data.
It does this by linking the data and its coordinates in single data objects.
These objects can be manipulated via array-like slicing operations which modify
the data and coordinates simultaneously.
They also allow coordinate transformations to be performed with reference to the
shape of the data array and produce visualizations whose axes are automatically
described by the coordinates.
This data-coordinate coupling allows users to analyze their data more
quickly and accurately, thus helping to boost their scientific output.

ndcube aims to be highly generalized.
It is agnostic to specifics of the data and the physical types
of coordinates that describe it.
This makes it a powerful base upon which to build tools for more specific types of data.
This might be a specific number and/or combination of physical types
(images, spectrograms, etc.), or data from specific instruments or simulations.
Thus, in addition to enhancing the productivity of scientists, ndcube can enhance
the productivity of developers by centralizing the development and maintenance
of the most useful and general functionalities.
This leaves more time for a greater range of tools to be developed for
the community and/or enables part-time developers to devote more effort to other
aspects of their jobs, e.g. scientific analysis.

# Statement of Need

N-dimensional data sets are common in all areas of science and beyond.
For example, a series of images taken sequentially with a CCD camera can be stored
as a single 3-D array with two spatial axes and one temporal axis.
The value in each array-element can represent the reading in a pixel at a given time.
In astronomy, the relationship between the array element and the location and time
in the Universe being observed is often represented by the World Coordinate System (WCS) framework.
WCS’s ability to handle many different physical types (e.g. spatial, temporal, spectral, etc.)
and projections (e.g. RA and Dec., helioprojective latitude and longitude, etc.)
make it a succinct, standardized and powerful way to relate array axes to the physical
properties they represent.

Due of the prevalence of N-D data and the importance of coordinate transformations,
there exist mature Python packages that handle them.
For example, arrays can be handled by numpy and dask and coordinates by astropy’s WCS
and coordinates modules.
If you want to treat these components separately, then these existing tools work well.
However, they are not suited to treating data and coordinates in a combined way.
This is the purpose of ndcube.

It’s worth addressing the role ndcube plays within the scientific Python ecosystem
and why it exists separately from its most similar package, xarray.
The fundamental reason to opt for ndcube is to harness the astronomy-specific
World Coordinate System (WCS).
The data model of xarray centers on the requirements and conventions of the geosciences.
Although similar to those of astronomy in conception, they are sufficiently different
in construction to cause significant friction.
Moreover, utilizing the astropy WCS infrastructure enables us to directly read the most
common file format in astronomy, FITS.

# Data Objects: The Pillars of ndcube

At the time of writing (ndcube v2.0), ndcube provides three primary data objects for
manipulating astronomical data: NDCube, NDCubeSequence, and NDCollection.
Each provide unified slicing, visualization, coordinate transformation and
self-inspection APIs which are independent of the number and physical types of axes.

## NDCube

NDCube is the primary data class the ndcube package.
It’s designed to manage a single data array and set of WCS transformations.
It can therefore be used for any type of data (e.g. images, spectra, timeseries, etc.)
so long as those data are represented by an object that behaves like a numpy ndarray [REFERENCE]
and the coordinates by an object that adheres to the Astropy WCS API [REFERNCE].

Thanks to its inheritance from astropy NDData [REFERENCE], NDCube can hold optional
supplementary data:
1. general metadata;
2. the unit of the data;
3. the uncertainty of each data value;
4. a mask marking unreliable data.

NDCube also provides support for tabular coordinates in addition to those stored in the
the primary WCS object via its ExtraCoords class.
Scalar coordinates that apply to the whole cube and are not associated with specific axis/axes
can be represented via the GlobalCoords class.
Instances of both these classes can be attached to an NDCube instance and self-consistently
handled, e.g. by the slicing or visualization infrastructure.

Figure \autoref{fig:ndcube} shows a schematic of an NDCube instance and
the relationships between its components.
Array-based components are in blue (data, uncertainty, and mask),
metadata components in green (meta and unit), and
coordinate components in red (wcs, extra coords, and global coords).
Yellow ovals represent methods for inspecting, visualizing, and analyzing the NDCube.

![Components of an NDCube.\label{fig:ndcube}](ndcube_diagram.png)

![The effects of slicing -- e.g. via ``my_cube[2:4, 8:16]`` -- on the components of an NDCube.\label{fig:ndcube_sliced}](ndcube_sliced_diagram.png)

To demonstrate the utility of NDCube, consider Figure \autoref{fig:ndcube_sliced}.
It represents what happens when the standard Python slicing API is applied to an NDCube,
e.g. ``my_cube[2:4, 8:16]``.
Note that with a single line of code, the shapes of the array-based components have
all been changed in accordance with the input slice item.
Meanwhile the coordinate components have been modified to ensure the same "pixels"
correspond to the same real world coordinate values, even though their array indices
have been altered by the slicing operation.
Manually altering and tracking each of these components, say if a user wanted to extract
a sub-region of the data, is a tedious process prone to mistakes.
However, due to the development and testing put into NDCube, this process is much
lower effort and users can be more confident that the operation is performed accurately.
This demonstrates the effort-saving nature of NDCube.

## NDCubeSequence

NDCubeSequence is a class for handling an ordered list of NDCubes with the same shape
and physical types as if they were one contiguous data cube.
The NDCubes can be ordered orthogonal to their data axes and/or parallel to one of them.
An example of the orthogonal case would be a sequence of 2-D images taken at different times.
As there is no time axis in the images themselves, the order in which they are arranged
constituents a quasi-axis known as the sequence axis.
An example of the parallel case would be a sequence of images in a horizontal mosaic,
each representing adjacent regions of the sky.
In this case, the right boundary of one image is also the left boundary the next.
Thus the images are ordered parallel to the horizontal axis of the images.
Both the othogonal and parallel arrangements are represented in Figure \autoref{fig:ndcubesequence}.

![Orthogonal and parallel ordering of NDCubes within an NDCubeSequence.\label{fig:ndcubesequence}](ndcubesequence_diagram.png)

NDCubeSequence provides slicing, coordinate transformation and visualization functionalities.
While not quite a powerful as NDCube, NDCubeSequence can nonetheless
perform many tasks just as though the data were stored in a single NDCube.

## NDCollection

NDCollection is a container class for grouping NDCubes or NDCubeSequences in an unordered way.
It therefore differs from NDCubeSequence in that the objects contained are
not considered to be in any order, are not assumed to represent measurements of the
same physical property, and can have different dimensionalities.

One possible application of NDCollection is linking observations with derived data products.
For example, say we have a 3-D NDCube representing a spectral image cube with physical axes of
space-space-wavelength.
Now say we fit a spectral line in each pixel and extract a doppler velocity.
This gives us a 2D spatial map of doppler velocity with the same spatial axes as the
original 3-D cube.
Due to the clear relationship between these two objects, it makes sense to store them together.
An NDCubeSequence is not appropriate because the physical properties represented are different.
Moreover they do not have an order within their common coordinate space
and they do not have the same dimensionality.
Instead we can bundle them in an NDCollection, each designated by a different key just
in a dictionary.
However, NDCollection is more powerful than a simple dictionaru because it enables us
to link the spatial axes of both NDCubes.
This means if we subsequently wish to extract a spatial region of interest,
we can slice the NDCollection rather than the slicing the observations and doppler map
separately.
As with NDCube and NDCubeSequence, NDCollection thus enables easier and more reliable
manipulation of N-D astronomical data sets.

# Community Use

Stable releases of ndcube are available through pip and conda-forge.
It is already used both byscientists in their data analysis and as
a dependency by other packages.
At time of writing packages depending on ndcube include: sunraster (a package for
handling data from solar slit spectrometer instruments); irispy (a package for
handling data from NASA's IRIS Small Explorer mission); user tools for the DKIST telescope;
specutils (an astropy-affiliated package for analyzing spectral data).
Further adoptions are expected, including by sunpy.

An added benefit of broad adoption of the ndcube APIs by other packages is that
scientists from sub-fields of astronomy and heliophysics will be able
to more easily work with data from other sub-fields.
This can help scientists combine observations from scientifically related, but
traditionally distinct disciplines.
This can help foster new collaborations and syngeries and promote scientific
innovation.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Acknowledgements

We acknowledge support for ndcube from NASA's Heliophysics Data Environment Enhancement
program and the Daniel K. Inoue Solar Telescope.
We also acknowledge the SunPy and Python in Heliophysics communities for their support.

# References

