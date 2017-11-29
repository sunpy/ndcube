=============================
An Introduction to ndcube
=============================

N-dimensional data sets are common in all areas of science and beyond.  For example, a series of images taken sequentially with a CCD camera can be stored as a single 3-D array with two spatial axes and one of time.  Each array-element can represent a pixel and the value in that array-element can represent the reading in that pixel at a given time.  In astronomy, the relationship between the pixel number and the location and time in the Universe being observed is often represented by a World Coordinate System (WCS) transformation described by a set of well-defined parameters with standarized names.  This, coupled with WCS's ability to handle many different types (e.g. spatial, temporal, spectral, etc.) of transformations make it a succinct, standard and powerful way to relate pixels from an observation or cells in a simulation grid to the location in the Universe to which they correspond.

Due of the prevalence of ND data, there exist mature and powerful tools in scientific Python packages to handle ND arrays and WCS transformations (e.g. numpy, astropy).  The purpose of ndcube is to combine these functionalities into a single package of objects that provide unified slicing, visualization, coordinate conversion and inspection of both the data and the WCS transformation.  ndcube does this in a way that is not specific to any type of axis, and so can be in principle be used for any type of data (e.g. images, spectra, timeseries, etc.) so long as it's represented by an array and a WCS.  This makes ndcube ideal to subclass when creating data classes for specific types of data, while keeping the non-data-specific functionalities (e.g. slicing) common between classes.

The ndcube package if composed of two basic classes: NDCube and NDCubeSequence.  The former is for managing a single array and WCS object, while the other is for handling multiple arrays and WCS objects.


