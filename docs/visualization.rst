.. _plotting:

======================
Visualizing ND Objects
======================

.. _cube_plotting:

Visualizing NDCubes
===================
`~ndcube.NDCube` provides a simple-to-use, yet powerful visualization method, `~ndcube.NDCube.plot`, which produces sensible visualizations based on the dimensionality of the data and optional user inputs.
It is intended to be a useful quicklook tool and not a replacement for high quality plots or animations, e.g. for publications.
Having defined our `~ndcube.NDCube` as before:

.. plot::
  :context: reset
  :nofigs:

  >>> import astropy.units as u
  >>> import astropy.wcs
  >>> import numpy as np
  >>> from astropy.nddata import StdDevUncertainty

  >>> from ndcube import NDCube

  >>> # Define data array.
  >>> data = np.random.rand(4, 4, 5)

  >>> # Define WCS transformations in an astropy WCS object.
  >>> wcs = astropy.wcs.WCS(naxis=3)
  >>> wcs.wcs.ctype = 'WAVE', 'HPLT-TAN', 'HPLN-TAN'
  >>> wcs.wcs.cunit = 'Angstrom', 'deg', 'deg'
  >>> wcs.wcs.cdelt = 0.2, 0.5, 0.4
  >>> wcs.wcs.crpix = 0, 2, 2
  >>> wcs.wcs.crval = 10, 0.5, 1

  >>> # Define mask.  Initially set all elements unmasked.
  >>> mask = np.zeros_like(data, dtype=bool)
  >>> mask[0, 0][:] = True  # Now mask some values.
  >>> # Define uncertainty, metadata and unit.
  >>> uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))
  >>> meta = {"Description": "This is example NDCube metadata."}
  >>> unit = u.ct

  >>> # Instantiate NDCube with supporting data.
  >>> my_cube = NDCube(data, wcs=wcs, uncertainty=uncertainty, mask=mask, meta=meta, unit=unit)

The plot method can be called very simply.

.. plot::
  :context:
  :include-source:

  >>> import matplotlib.pyplot as plt
  >>> ax = my_cube.plot()
  >>> plt.show()

For a >2 array axes, as is the case above, an animation object is returned displaying either a line or image with sliders for each additional array axis.
These sliders are used to sequentially update the line or image as it moves along its corresponding array axis, thus animating the data.

For for data with two array axes, an image is produced similar to that of `matplotlib.pyplot.imshow`.

.. plot::
  :context:
  :include-source:

  >>> ax = my_cube[0].plot()
  >>> plt.show()

For data with one array axis, a line plot is produced, similar to `matplotlib.pyplot.plot`.

.. plot::
  :context:
  :include-source:

  >>> ax = my_cube[0, 0,].plot()
  >>> plt.show()

Setting the x and y ranges of the plot can be done simply by indexing the `~ndcube.NDCube` object to the desired region of interest and then calling the plot method, e.g.

.. plot::
  :context:
  :include-source:

  >>> ax = my_cube[0, 1:3, :].plot()
  >>> plt.show()

No args are required.
The necessary information to generate the plot is derived from the data and metadata in the `~ndcube.NDCube`.
However optional keywords enable customization of the visualization.
For `~ndcube.NDCube` instances with more than one array axis, the ``plot_axes`` keyword is used to determine which array axes are displayed on which plot axes.It is set to a list with a length equal to the number of array axes.
The array axis to be displayed on the x-axis is marked by ``'x'`` in the corresponding element of the ``plot_axes`` list, while the array axis for the y-axis is marked with a ``'y'``.
If no ``'y'`` axis is provided, a line animation is produced.
By default the ``plot_axes`` argument is set so that the last array axis to shown on the x-axis and the penultimate array axis is shown on the y-axis.

.. plot::
  :context:
  :include-source:

  >>> ax = my_cube[0].plot(plot_axes=[..., 'y', 'x'])
  >>> plt.show()

`~ndcube.NDCube.plot` uses `~astropy.visualization.wcsaxes.WCSAxes` to produce all plots.
This enables a rigorous representation of the coordinates on the plot, including those that are not aligned to the pixel grid.
It also enables the coordinates along the plot axes to be updated between frames of an animation.
`ndcube.NDCube.plot` therefore allows users to decide which WCS object to use, either `~ndcube.NDCube.wcs` or `~ndcube.NDCube.combined_wcs` which also includes the `~ndcube.ExtraCoords`.
In principle, another third-part WCS can be used so long as it is a valid description of all array axes.

.. plot::
  :context:
  :include-source:

  >>> ax = my_cube[0].plot(wcs=my_cube[0].combined_wcs)
  >>> plt.show()

Visualizing NDCubeSequences
===========================
Since ndcube 2.0, the `~ndcube.NDCubeSequence` visualization support has been dropped.
The rationale for this is outlined in `Issue #321 <https://github.com/sunpy/ndcube/issues/321>`_ on the ndcube GitHub repo.
If you feel that `~ndcube.NDCubeSequence` visualization should be supported again, please let us know by commenting on that issue and telling us of your use case.
Better still, if you would like to work on the infrastructure required to support `~ndcube.NDCubeSequence` visualization is a post ndcube 2.0 world let us know by commenting on the issue.

Despite this the lack of `~ndcube.NDCubeSequence` visualization support, you can still visualize the data in `~ndcube.NDCubeSequence` in a number of ways.
You can slice out a single `~ndcube.NDCube` and use its `~ndcube.NDCube.plot` method.
You can extract the data and use the myriad of plotting packages available in the Python ecosystem.
Finally, if you want to be advanced, you can write your own mixin class to define the plotting methods.
Below, we will outline these latter two options in a little more detail.

Extracting and Plotting NDCubeSequence Data with Matplotlib
-----------------------------------------------------------
In order to produce plots (or perform other analysis) outside of the ``ndcube`` framework, it may be useful to extract the data from the `~ndcube.NDCubeSequence` into single `~numpy.ndarray` instances.
Let's first define an `~ndcube.NDCubeSequence` with a common axis of 0 and and time as an extra coord stretching across the cube along the common axis.
Then we show how to extract and plot the data.

.. code-block:: python

  >>> import astropy.units as u
  >>> import astropy.wcs
  >>> import numpy as np
  >>> from astropy.time import Time, TimeDelta

  >>> from ndcube import ExtraCoords, NDCube, NDCubeSequence

  >>> # Define data arrays.
  >>> shape = (3, 4, 5)
  >>> data0 = np.random.rand(*shape)
  >>> data1 = np.random.rand(*shape)
  >>> data2 = np.random.rand(*shape)

  >>> # Define WCS transformations. Let all cubes have same WCS.
  >>> wcs = astropy.wcs.WCS(naxis=3)
  >>> wcs.wcs.ctype = 'WAVE', 'HPLT-TAN', 'HPLN-TAN'
  >>> wcs.wcs.cunit = 'Angstrom', 'deg', 'deg'
  >>> wcs.wcs.cdelt = 0.2, 0.5, 0.4
  >>> wcs.wcs.crpix = 0, 2, 2
  >>> wcs.wcs.crval = 10, 0.5, 1

  >>> # Define time extra coordinates of time for each cube.
  >>> common_axis = 0
  >>> base_time = Time('2000-01-01', format='fits', scale='utc')
  >>> timestamps0 = Time([base_time + TimeDelta(60 * i, format='sec') for i in range(data0.shape[common_axis])])
  >>> extra_coords0 = ExtraCoords()
  >>> extra_coords0.add_coordinate('time', 2, timestamps0)
  >>> timestamps1 = Time([base_time + TimeDelta(60 * (i+1), format='sec') for i in range(data1.shape[common_axis])])
  >>> extra_coords1 = ExtraCoords()
  >>> extra_coords1.add_coordinate('time', 2, timestamps1)
  >>> timestamps2 = Time([base_time + TimeDelta(60 * (i+1), format='sec') for i in range(data2.shape[common_axis])])
  >>> extra_coords2 = ExtraCoords()
  >>> extra_coords2.add_coordinate('time', 2, timestamps2)

  >>> # Define the cubes
  >>> cube0 = NDCube(data0, wcs=wcs, extra_coords=extra_coords0)
  >>> cube1 = NDCube(data1, wcs=wcs, extra_coords=extra_coords1)
  >>> cube2 = NDCube(data2, wcs=wcs, extra_coords=extra_coords2)

  >>> # Define the sequence
  >>> my_sequence = NDCubeSequence([cube0, cube1, cube2], common_axis=common_axis)

To make a 4D array out of the data arrays within the `~ndcube.NDCubes` of `my_sequence`.

.. code-block:: python

    >>> data4d = np.stack([cube.data for cube in my_sequence.data], axis=0)
    >>> data4d.shape
    (3, 3, 4, 5)

The same applies to other array-like data in the `~ndcube.NDCubeSequence`, like ``uncertainty`` and ``mask``.
If instead, we want to define a 3D array where every `~ndcube.NDCube` in the `~ndcube.NDCubeSequence` is appended along the ``common_axis``, we can use `numpy.concatenate` function.

.. code-block:: python

    >>> data3d = np.concatenate([cube.data for cube in my_sequence.data],
    ...                         axis=my_sequence._common_axis)
    >>> data3d.shape
    (9, 4, 5)

Having extracted the data, we can now use matplotlib to visualize it.
Let's say we want to produce a timeseries of how intensity changes in a given pixel at a given wavelength.
We stored time in ``my_sequence.common_axis_coords`` and associated it with the ``common_axis``.
Therefore, we could do:

.. code-block:: python

    >>> import matplotlib.pyplot as plt
    >>> # Get intensity at pixel 0, 0, 0 in each cube.
    >>> intensity = np.array([cube.data[0, 0, 0] for cube in my_sequence])
    >>> times = Time([cube.axis_world_coords('time', wcs=cube.combined_wcs)[0][0] for cube in my_sequence])
    >>> plt.plot(times.datetime, intensity)  # doctest: +SKIP
    >>> plt.xlabel("Time")  # doctest: +SKIP
    >>> plt.ylabel("Intensity")  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP

Alternatively, we could produce a 2D dynamic spectrum showing how the spectrum in a given pixel changes over time.

.. code-block:: python

    >>> import matplotlib as mpl
    >>> import matplotlib.pyplot as plt
    >>> from astropy.time import Time
    >>> # Combine spectrum over time for pixel 0, 0.
    >>> spectrum_sequence = my_sequence[:, :, 0]
    >>> intensity = np.concatenate([cube.data for cube in spectrum_sequence.data], axis=0)
    >>> times = Time(np.concatenate([cube.axis_world_coords('time', wcs=cube.combined_wcs)[0].value for cube in my_sequence]), format='fits', scale='utc')
    >>> # Assume that the wavelength in each pixel doesn't change as we move through the sequence.
    >>> wavelength = spectrum_sequence[0].axis_world_coords("em.wl")[0]
    >>> # As the times may not be uniform, we can use NonUniformImage to show non-uniform pixel sizes.
    >>> fig, ax = plt.subplots(1, 1)  # doctest: +SKIP
    >>> im = mpl.image.NonUniformImage(
    ...     ax, extent=(times[0], times[-1], wavelength[0], wavelength[-1]))  # doctest: +SKIP
    >>> im.set_data(wavelength, times.mjd, intensity)  # doctest: +SKIP
    >>> ax.add_image(im)  # doctest: +SKIP
    >>> ax.set_xlim(times.mjd[0], times.mjd[-1])  # doctest: +SKIP
    >>> ax.set_xlabel("Time [Modified Julian Day]")  # doctest: +SKIP
    >>> ax.set_ylim(wavelength[0].value, wavelength[-1].value)  # doctest: +SKIP
    >>> ax.set_ylabel(f"Wavelength [{wavelength.unit}]")  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP

Now let's say we want to animate our data, for example, show how the intensity changes over wavelength and time.
For this we can use `~ndcube.visualization.animator.ImageAnimator`.
This class is not well suited to displaying the complex relationship between coordinates that we are used to with `~astropy.visualization.wcsaxes.WCSAxes`.
For example, non-linear coordinates non-independent coordinates.
The difficulty and complexity in correctly representing this in a generalized way when dealing with a sequence of WCS objects is one reason plotting is currently no longer supported by `~ndcube.NDCubeSequence`.
Nontheless, `~sunpy.visualization.animator.ImageAnimator` can still give us an idea of how the data is changing.
In ``my_sequence``, the sequence axis represents time, the 0th and 1st cube axes represent latittude and longitude, while the final axis represents wavelength.
Therefore, we could do the following.

.. code-block:: python

    >>> from sunpy.visualization.animator import ImageAnimator  # doctest: +SKIP
    >>> data = np.stack([cube.data for cube in my_sequence.data], axis=0)
    >>> # Assume that the field of view or wavelength grid is not changing over time.
    >>> # Also assume the coordinates are independent and linear with the pixel grid.
    >>> animation = ImageAnimator(data, image_axes=[2, 1])  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP

Alternatively we can animate how the one 1-D spectrum changes by using `~sunpy.visualization.animator.LineAnimator`.

.. code-block:: python

    >>> from sunpy.visualization.animator import LineAnimator  # doctest: +SKIP
    >>> data = np.stack([cube.data for cube in my_sequence.data], axis=0)
    >>> animation = LineAnimator(data, plot_axis_index=-1)  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP

Writing Your Own NDCubeSequence Plot Mixin
------------------------------------------
Just because ndcube no longer provides plotting support doesn't mean you can't write your own plotting functionality for `~ndcube.NDCubeSequence`.
In many cases, this might be simpler as you may be able to make some assumptions about the data you will be analyzing and therefore won't have to write as generalized a tool.
The best way to do this is to write your own mixin class defining the plot methods, e.g.

.. code-block:: python

   class MySequencePlotMixin:
       def plot(self, **kwargs):
           pass  # Write code to plot data here.

       def plot_as_cube(self, **kwargs):
           pass  # Write code to plot data concatenated along common axis here.

Then you can create your own ``NDCubeSequence`` by combining your mixin with `~ndcube.NDCubeSequenceBase` which holds all the non-plotting functionality of the `~ndcube.NDCubeSequence`.

.. code-block:: python

    class MySequence(NDCubeSequenceBase, MySequencePlotMixin):

This will create a new class, ``MySequence``, which contains all the functionality of `~ndcube.NDCubeSequence` plus the plot methods you've defined in ``MySequencePlotMixin``.

There are many other ways you could visualize the data in your `~ndcube.NDCubeSequence` and many other visualization packages in the Python ecosystem that you could use.
These examples show just a few simple ways.
But hopefully this has shown you that it's still possible to visualize the data in your `~ndcube.NDCubeSequence`, whether by creating your own mixin, following the above examples, or by using some other infrastructure.
