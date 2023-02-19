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
  - name: Richard O'Steen
    orcid: 0000-0002-2432-8946
    affiliation: 9
  - name: Baptiste Pellorce
    affiliation: "10, 11"
  - name: Will T. Barnes
    orcid: 0000-0001-6874-2594
    affiliation: "12, 1"
  - name: Laura A. Hayes
    orcid: 0000-0002-6835-2390
    affiliation: 13
  - name: Derek Homeier
    orcid: 0000-0002-8546-9128
    affiliation: 3
  - name: J. Marcus Hughes
    orcid: 0000-0003-3410-7650
    affiliation: 14
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
 - name: Space Telescope Science Institute, USA
   index: 9
 - name: Claude Bernard Lyon 1 University, France
   index: 10
 - name: Institute of Theoretical Astrophysics, Norway
   index: 11
 - name: NASA Goddard Space Flight Center, USA
   index: 12
 - name: European Space Agency, ESTEC
   index: 13
 - name: Southwest Research Institute, USA
   index: 14
date: 11 December 2021
bibliography: paper.bib

---

# Summary

ndcube is a free, open-source, community-developed Python package for inspecting,
manipulating, and visualizing n-dimensional astronomical data.
It links data and the coordinates describing its axes in unified data objects.
These objects can be manipulated via array-like slicing operations which modify
the data and coordinates simultaneously.
They also allow coordinate transformations to be performed with reference to the
shape of the data and enable coordinate-aware visualizations.
ndcube leverages Astropy's WCS (World Coordinate System) API [APE-14; @ape14] to
enforce a standardized and widely used framework for performing astronomical
coordinate transformations.
The data-coordinate coupling provided by ndcube allows users to analyze their data
more easily and accurately, thus helping to boost their scientific output.

ndcube serves three specific purposes.
First, it formalizes the NDCube 2 API in software via its abstract base classes
(ABCs), NDCubeABC, ExtraCoordsABC and GlobalCoordsABC.
The NDCube 2 API is a standarized framework for inspecting and manipulating
coordinate-aware N-D data.
We refer readers to the 12th SunPy Enhancement Proposal [SEP-12; @sep12] for a
definition of the API and to [@ndcube] for a discussion of the philosophies
underpinning it.
Second, the ndcube package implements the NDCube 2 API in corresponding data and
coordinate classes, NDCubeBase, ExtraCoords and GlobalCoords.
These are viable off-the-shelf tools for end users and developers who do
not want to implement the API themselves.
Third, the ndcube package provides additional support for coordinate-aware
manipulation and visualization of N-D astronomical data via three high-level data
classes, NDCube, NDCubeSequence and NDCollection.
NDCube inherits from NDCubeBase and so adheres to the NDCube 2 API but adds some
additional features beyond the API's scope, such as a default visualization suite.
The other classes are designed to handle multiple NDCube instances simultaneously.

# Statement of Need

N-dimensional data sets are common in all areas of science and beyond.
For example, a series of images taken sequentially with a CCD camera can be stored
as a single 3-D array with two spatial axes and one temporal axis.
In astronomy, the most commonly used framework for translating between array element
indices and the location and time in the Universe being observed is the World
Coordinate System (WCS).
WCS’s ability to handle many different physical types (e.g. spatial, temporal, spectral, etc.)
and projections (e.g. RA and Dec., helioprojective latitude and longitude, etc.)
make it a succinct, standardized and powerful way to relate array axes to the physical
properties they represent.
However, while there exist Python packages for handling N-D array operations --
e.g. numpy [@numpy], dask [@dask], etc -- and others for supporting WCS coordinate
transformations -- e.g. astropy [@astropy], gWCS [gWCS] -- currently only ndcube is
suited to treating them in a combined way.

ndcube is agnostic to the physical properties represented by the data values and axes.
This makes it a powerful base upon which to build tools for specific types of data.
This might be a specific number and/or combination of physical types
(spectrograms, image cubes, etc.), or data from specific instruments or simulations.
Thus, ndcube can enhance the productivity of developers by centralizing the
development and maintenance of the most useful and general functionalities.
This leaves more time for developing a greater range of tools for the community
and/or enables part-time developers to devote more effort to other aspects of their
jobs, e.g. scientific analysis.

# Community Applications

ndcube already supports a variety of observatories and satellite missions including
the James Webb Space Telescope (JWST), Solar Orbiter, Interface Region Imaging
Spectrograph (IRIS), Hinode, and the Daniel K. Inouye Solar Telescope (DKIST),
through the specutils [@specutils], sunraster [@sunraster], irispy-lmsal [@irispy],
EISPAC [@eispac] and DKIST packages.
It is also used in the data pipeline of the PUNCH mission [Polarimeter to UNify the
Corona and Heliosphere; @punch], scheduled for launch in 2025.
A network benefit of ndcube is that it standardizes the APIs for handling N-D data.
Adoption across astronomy and heliophysics helps scientists to more easily work with
data from different missions and sub-fields of astronomy.
This can help facilitate synergies between new combinations of data, foster inter-field
collaborations, and promote scientific innovation.

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
program and the Daniel K. Inouye Solar Telescope.
We also acknowledge the SunPy and Python in Heliophysics communities for their support.

# References

