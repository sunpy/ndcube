.. _ndcube:

======
NDCube
======

`~ndcube.NDCube` is the fundamental class of the ndcube package.  It is designed
to handle data contained in a single N-D array described by a single
set of WCS transformations.  `~ndcube.NDCube` is subclassed from
`astropy.nddata.NDData` and so inherits the same attributes for data,
wcs, uncertainty, mask, meta, and unit. Since v2.0, the ``wcs`` object must
adhere to astropy's, APE 14 WCS API and ndcube leverages the WCS slicing
functionality provided by astropy, much of which was upstreamed from ndcube 1.0.

Initialization
==============

To initialize the most basic `~ndcube.NDCube` object, all you need is a
`numpy.ndarray`-like array containing the data and an APE-14-compliant WCS object
(e.g. `astropy.wcs.WCS`) describing the coordinate transformation to and from
array-elements. Let's create a 3-D array of data with shape (3, 4, 5)
where every value is 1::

  >>> import numpy as np
  >>> data = np.ones((3, 4, 5))

Now let's create an `astropy.wcs.WCS` object describing the
translation from the array element coordinates to real world
coordinates.  Let the first data axis be helioprojective longitude,
the second be helioprojective latitude, and the third be wavelength.
Note that due to (confusing) convention, the order of the axes in the
WCS object is reversed relative to the data array.

  >>> import astropy.wcs
  >>> wcs_input_dict = {
  ... 'CTYPE1': 'WAVE    ', 'CUNIT1': 'Angstrom', 'CDELT1': 0.2, 'CRPIX1': 0, 'CRVAL1': 10, 'NAXIS1': 5,
  ... 'CTYPE2': 'HPLT-TAN', 'CUNIT2': 'deg', 'CDELT2': 0.5, 'CRPIX2': 2, 'CRVAL2': 0.5, 'NAXIS2': 4,
  ... 'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'deg', 'CDELT3': 0.4, 'CRPIX3': 2, 'CRVAL3': 1, 'NAXIS3': 3}
  >>> input_wcs = astropy.wcs.WCS(wcs_input_dict)

Now we can create an `~ndcube.NDCube`.::

  >>> from ndcube import NDCube
  >>> my_cube = NDCube(data, input_wcs)

The data array is stored in ``mycube.data`` while the
WCS object is stored in ``my_cube.wcs``.  However, when
manipulating/slicing the data is it better to slice the object as a
whole.  (See section on :ref:`ndcube_slicing`.)  So the ``.data`` attribute
should only be used to access specific raw data values.

Thanks to `~ndcube.NDCube`'s inheritance from `astropy.nddata.NDData`,
you can also supply additional data to the
`~ndcube.NDCube` instance.  These include: metadata located at `NDCube.meta`;
a data mask (boolean `numpy.ndarray`) located at `NDCube.mask` marking reliable and unreliable pixels; an uncertainty array located at
`NDCube.uncertainty` (subclass of `astropy.nddata.NDUncertainty`) describing
the uncertainty of each data array value;
and a unit (`astropy.units.Unit` or unit `str`). For example::

  >>> from astropy.nddata import StdDevUncertainty
  >>> uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))
  >>> # Like numpy masked array's False means the data is unmasked, i.e. good.
  >>> mask = np.zeros_like(my_cube.data, dtype=bool)
  >>> meta = {"Description": "This is example NDCube metadata."}
  >>> my_cube = NDCube(data, input_wcs, uncertainty=uncertainty,
  ...                         mask=mask, meta=meta, unit=u.ct)

Dimensions
----------

`~ndcube.NDCube` has useful properties for inspecting its data shape and
axis types, `~ndcube.NDCube.dimensions` and
`~ndcube.NDCube.array_axis_physical_types`::

  >>> my_cube.dimensions
  <Quantity [3., 4., 5.] pix>
  >>> my_cube.array_axis_physical_types
  [('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
   ('custom:pos.helioprojective.lat', 'custom:pos.helioprojective.lon'),
   ('em.wl',)]

`~ndcube.NDCube.dimensions` returns an `~astropy.units.Quantity` of
pixel units giving the length of each dimension in the
`~ndcube.NDCube` while `~ndcube.NDCube.array_axis_physical_types`
returns tuples of strings denoting the types of physical properties
represented by each array axis.  As more than one physical type can be associated
with an axis, the length of each tuple can be greater than 1.
This is the case for the 0th and 1st array axes which are associated with
the coupled physical axes of latitude and longitude. The axis names are
in accordance with the International Virtual Observatory Alliance (IVOA)
`UCD1+ controlled vocabulary <http://www.ivoa.net/documents/REC/UCD/UCDlist-20070402.html>`_.

.. _ndcube_slicing:

Slicing
-------

Arguably NDCube's most powerful capability is its slicing.  Slicing an
`~ndcube.NDCube` instance using the standard slicing notation allows
users to access sub-regions of their data while simultaneously slicing
not only the other array attributes (e.g. uncertainty, mask, etc.) but
also the WCS object.  This ensures that even though the data array has
changed size and shape, each array element will still correspond to
the same real world coordinates as they did before.  An example of how
to slice a 3-D `~ndcube.NDCube` object is::

  >>> my_cube_roi = my_cube[0:2, 1:4, 1:4]

Slicing can also reduce the dimension of an `~ndcube.NDCube`, e.g.::

  >>> my_2d_cube = my_cube[0, 1:4, 1:4]

In addition to slicing by index, `~ndcube.NDCube` supports a basic
version of slicing/indexing by real world coordinates via the
`~ndcube.NDCube.crop` method.  This takes two iterables of high level astropy objects -- e.g. `~astropy.time.Time`, `~astropy.coordinates.SkyCoord`,
`~astropy.coordinates.SpectralCoord`, `~astropy,units.Quantity` -- which depend
the physical types of the axes in the cube.  Each iterable describes a single
location in the data array in real world coordinates.  The first iterable
describes the lower corner of the region of interest and thus contains the lower limit
of all the real world coordinates.  The second iterable represents the upper corner
of the region of interest and thus contains the upper limit of all the real world coordinates.
The crop method indentifies the smallest rectangular region in the data array
that contains both the lower and upper limits in all the real world coordinates,
and crops the `~ndcube.NDCube` to that region. It does not rebin or interpolate the data.
The order of the high level coordinate objects in each iterable must be the same as
that expected by `astropy.wcs.WCS.world_to_array_index`, namely in world order.::

  >>> import astropy.units as u
  >>> from astropy.coordinates import SkyCoord, SpectralCoord
  >>> from sunpy.coordinates.frames import Helioprojective
  >>> wave_range = SpectralCoord([1.04e-9, 1.08e-9], unit=u.m)
  >>> sky_range = SkyCoord(Tx=[1, 1.5], Ty=[0.5, 1.5], unit=u.deg, frame=Helioprojective)
  >>> lower_corner = [wave_range[0], sky_range[0]]
  >>> upper_corner = [wave_range[-1], sky_range[-1]]
  >>> my_cube_roi = my_cube.crop(lower_corner, upper_corner)

Plotting
--------

To quickly and easily visualize N-D data, `~ndcube.NDCube` provides a
simple-to-use, yet powerful plotting method, `~ndcube.NDCube.plot`,
which produces a sensible visualization based on the dimensionality of
the data.  It is intended to be a useful quicklook tool and not a
replacement for high quality plots or animations, e.g. for
publications.  The plot method can be called very simply, like so::

  >>> my_cube.plot() # doctest: +SKIP

The type of visualization returned depends on the dimensionality of
the data within the `~ndcube.NDCube` object.  For 1-D data a line plot
is produced, similar to `matplotlib.pyplot.plot`.  For 2-D data, an
image is produced similar to that of `matplotlib.pyplot.imshow`.
While for a >2-D data, a
`sunpy.visualization.imageanimator.ImageAnimatorWCS` object is
returned.  This displays a 2-D image with sliders for each additional
dimension which allow the user to animate through the different values
of each dimension and see the effect in the 2-D image.

No args are required.  The necessary information to generate the plot
is derived from the data and metadata in the `~ndcube.NDCube`
itself. Setting the x and y ranges of the plot can be done simply by
indexing the `~ndcube.NDCube` object itself to the desired region of
interest and then calling the plot method, e.g.::

  >>> my_cube[0, 10:100, :].plot() # doctest: +SKIP

In addition, some optional kwargs can be used to customize the
plot.  The ``axis_ranges`` kwarg can be used to set the axes ticklabels.  See the
`~sunpy.visualization.imageanimator.ImageAnimatorWCS` documentation for
more detail.  However, if this is not set, the axis ticklabels are
automatically derived in real world coordinates from the WCS object
within the `~ndcube.NDCube`.

By default the final two data dimensions are used for the plot
axes in 2-D or greater visualizations, but this can be set by the user
using the ``images_axes`` kwarg::

  >>> my_cube.plot(image_axes=[0,1]) # doctest: +SKIP

where the first entry in the list gives the index of the data index to
go on the x-axis, and the second entry gives the index of the data
axis to go on the y-axis.

In addition, the units of the axes or the data can be set by the
``unit_x_axis``, ``unit_y_axis``, unit kwargs.  However, if not set,
these are derived from the `~ndcube.NDCube` wcs and unit attributes.

And Much More
=============
`~ndcube.NDCube` provides many more helpful features, specifically regarding coordinate transformations and visualization.  To learn more, see the :ref:plotting and :ref:coordinates sections.
