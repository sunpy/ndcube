.. _plotting

=============
Visualization
=============

.. _cube_plotting

Visualizing NDCubes
===================
`~ndcube.NDCube` provides a simple-to-use, yet powerful visualization method, `~ndcube.NDCube.plot`, which produces sensible visualizations based on the dimensionality of the data and optional user inputs.  It is intended to be a useful quicklook tool and not a replacement for high quality plots or animations, e.g. for publications.  The plot method can be called very simply.::

  >>> my_cube.plot() # doctest: +SKIP

For data with one array axis, a line plot is produced, similar to `matplotlib.pyplot.plot`.  For for data with two array axes, an image is produced similar to that of `matplotlib.pyplot.imshow`.  For a >2 array axes, an animation object is returned displaying either a line or image with sliders for each additional array axis.  These sliders are used to sequentially update the line or image as it moves along its corresponding array axis, thus animating the data.

Setting the x and y ranges of the plot can be done simply by indexing the `~ndcube.NDCube` object to the desired region of interest and then calling the plot method, e.g.::

  >>> my_cube[0, 10:100, :].plot() # doctest: +SKIP

No args are required. The necessary information to generate the plot is derived from the data and metadata in the `~ndcube.NDCube`. However optional keywords enable customization of the visualization.  For `~ndcube.NDCube` instances with more than one array axis, the ``plot_axes`` keyword is used to determine which array axes are displayed on which plot axes.  It is set to a list with a length equal to the number of array axes.  The array axis to be displayed on the x-axis is marked by ``'x'`` in the corresponding element of the ``plot_axes`` list, while the array axis for the y-axis is marked with a '``'y'``.  If no ``'y'`` axis is provided, a line animation is produced.  By default the ``plot_axes`` argument is set so that the last array axis to shown on the x-axis and the penultimate array axis is shown on the y-axis.::

  >>> my_cube.plot(plot_axes=[..., 'y', 'x']) # doctest: +SKIP
  
`~ndcube.NDCube.plot` uses `~astropy.visualization.wcsaxes.WCSAxes` to produce all plots.  This enables a rigorous representation of the coordinates on the plot, including those that are not aligned to the pixel grid.  It also enables the coordinates along the plot axes to be updated between frames of an animation. `ndcube.NDCube.plot` therefore allows users to decide which WCS object to use, either `~ndcube.NDCube.wcs` or `~ndcube.NDCube.combined_wcs` which also includes the `~ndcube.ExtraCoords`.  In principle, another third-part WCS can be used so long as it is a valid description of all array axes.::

  >>> my_cube.plot(wcs=my_cube.combined_wcs)   # doctest: +SKIP

Visualizing NDCubeSequences
===========================
Since ndcube 2.0, the `~ndcube.NDCubeSequence` visualization support has been dropped.
The rationale for this is outlined in `Issue #321 <https://github.com/sunpy/ndcube/issues/321>`_ on the ndcube GitHub repo.
If you feel that `~ndcube.NDCubeSequence` visualization should be supported again, please let us know by commenting on that issue and telling us of your use case.  Better still, if you would like to work on the infrastructure required to support `~ndcube.NDCubeSequence` visualization is a post ndcube 2.0 world let us know by commenting on the issue.

Despite this the lack of `~ndcube.NDCubeSequence` visualization support, you can still visualize the data in `~ndcube.NDCubeSequence` in a number of ways. You can slice out a single `~ndcube.NDCube` and use its `~ndcube.NDCube.plot` method.  You can extract the data and use the myriad of plotting packages available in the Python ecosystem.  Finally, if you want to be advanced, you can write your own mixin class to define the plotting methods.  Below, we will outline these latter two options in a little more detail.

Extracting and Plotting NDCubeSequence Data with Matplotlib
-----------------------------------------------------------

In order to produce plots (or perform other analysis) outside of the ``ndcube`` framework,
it may be useful to extract the data from the `~ndcube.NDCubeSequence` into single
`~numpy.ndarray` instances.
In the above examples we defined the `my_sequence` `~ndcube.NDCubeSequence` object.::

    >>> # Print dimensions of my_sequence as a reminder
    >>> print(my_sequence.dimensions)
    (<Quantity 3. pix>, <Quantity 3. pix>, <Quantity 4. pix>, <Quantity 5. pix>)

To make a 4D array out of the data arrays within the `~ndcube.NDCubes` of `my_sequence`.::

    >>> # Make a single 4D array of data in sequence with the sequence axis as the 0th.
    >>> data4d = np.stack([cube.data for cube in my_sequence.data], axis=0)
    >>> print(data.shape)
    (3, 3, 4, 5)

The same applies to other array-like data in the `~ndcube.NDCubeSequence`, like
``uncertainty`` and ``mask``.
If instead, we want to define a 3D array where every `~ndcube.NDCube` in the
`~ndcube.NDCubeSequence` is appended along the ``common_axis``,
we can use `numpy.concatenate` function::

    >>> # Make a 3D array
    >>> data3d = np.concatenate([cube.data for cube in my_sequence.data],
                                axis=my_sequence._common_axis)
    >>> print(data.shape)
    (9, 4, 5)

Having extracted the data, we can now use matplotlib to visualize it.
Let's say we want to produce a timeseries of how intensity changes in a
given pixel at a given wavelength.  We stored time in ``my_sequence.global_coords``
and associated it with the ``common_axis``.  Therefore, we could do::

    >>> import matplotlib.pyplot as plt
    >>> # Get intensity at pixel 0, 0, 0 in each cube.
    >>> intensity = np.array([cube.data[0, 0, 0] for cube in my_sequence])
    >>> times = my_sequence.common_axis_coords["time"]
    >>> plt.plot(times, intensity)
    >>> plt.show()

Alternatively, we could produce a 2D dynamic spectrum showing how the spectrum
in a given pixel changes over time.::

    >>> import matplotlib as mpl
    >>> import matplotlib.pyplot as plt
    >>> from astropy.time import Time
    >>> # Combine spectrum over time for pixel 0, 0.
    >>> spectrum_sequence = my_sequence[0, 0]
    >>> intensity = np.stack([cube.data for cube in spectrum_sequence[0, 0], axis=0)
    >>> times = Time(spectrum_sequence.sequence_axis_coords["time"])
    >>> # Assume that the wavelength in each pixel doesn't change as we move through the sequence.
    >>> wavelength = spectrum_sequence[0].axis_world_coords("em.wl")
    >>> # As the times may not be uniform, we can use NonUniformImage
    >>> # to show non-uniform pixel sizes.
    >>> fig, ax = plt.subplots(1, 1)
    >>> im = mpl.image.NonUniformImage(
    ...     ax, extent=(times[0], times[-1], wavelength[0], wavelength[-1]))
    >>> im.set_data(times, wavelength, intensity)
    >>> ax.add_image(im)
    >>> ax.set_xlim(times[0], times[-1])
    >>> ax.set_ylim(wavelength[0], wavelength[-1])
    >>> plt.show()

Now let's say we want to animate our data, for example, show how the intensity
changes over wavelength and time.
For this we can use `~ndcube.visualization.animator.ImageAnimator`.
This class is not well suited to displaying the complex relationship between coordinates
that we are used to with `~astropy.visualization.wcsaxes.WCSAxes`.
For example, non-linear coordinates non-independent coordinates.
The difficulty and complexity in correctly representing this in a generalized way
when dealing with a sequence of WCS objects is one reason plotting is currently
no longer supported by `~ndcube.NDCubeSequence`.
Nontheless, `~ndcube.visualization.animator.ImageAnimator` can still give us an idea
of how the data is changing.
In ``my_sequence``, the sequence axis represents time, the 0th and 1st cube axes
represent latittude and longitude, while the final axis represents wavelength.
Therefore, we could do the following::

    >>> from ndcube.visualization import ImageAnimator
    >>> data = np.stack([cube.data for cube in my_sequence.data], axis=0)
    >>> time_range = [my_sequence[0, 0].global_coords.get_coord("time"),
                      my_sequence[-1, 0].global_coords.get_coord("time")]
    >>> # Assume that the field of view or wavelength grid is not changing over time.
    >>> # Also assume the coordinates are independent and linear with the pixel grid.
    >>> lon, lat, wavelength = my_sequence[0].axis_world_coords_values(wcs=my_sequence[0].wcs)
    >>> lon_range = [lon[0], lon[-1]]
    >>> lat_range = [lat[0], lat[-1]]
    >>> wave_range = [wavelength[0], wavelength[-1]]
    >>> animation = ImageAnimator(data, image_axes=[2, 1],
                                  axis_ranges=[time_range, lon_range, lat_range, wave_range])
    >>> plt.show()

Alternatively we can animate how the one 1-D spectrum changes by using
`~ndcube.visualization.animator.LineAnimator`::

    >>> from ndcube.visualization import ImageAnimator
    >>> data = np.stack([cube.data for cube in my_sequence.data], axis=0)
    >>> time_range = [my_sequence[0, 0].global_coords.get_coord("time"),
                      my_sequence[-1, 0].global_coords.get_coord("time")]
    >>> # Assume that the field of view or wavelength grid is not changing over time.
    >>> # Also assume the coordinates are independent and linear with the pixel grid.
    >>> lon, lat, wavelength = my_sequence[0].axis_world_coords_values()
    >>> lon_range = [lon[0], lon[-1]]
    >>> lat_range = [lat[0], lat[-1]]
    >>> wave_range = [wavelength[0], wavelength[-1]]
    >>> animation = LineAnimator(data, plot_axis_index=-1,
                                 axis_ranges=[time_range, lon_range, lat_range, wave_range])
    >>> plt.show()

Writing Your Own NDCubeSequence Plot Mixin
------------------------------------------
Just because ndcube no longer provides plotting support doesn't mean you can't write your own
plotting functionality for `~ndcube.NDCubeSequence`.
In many cases, this might be simpler as you may be able to make some assumptions about the
data you will be analyzing and therefore won't have to write as generalized a tool.
The best way to do this is to write your own mixin class defining the plot methods, e.g.

.. code-block:: python

   class MySequencePlotMixin:
       def plot(self, **kwargs):
           pass  # Write code to plot data here.

       def plot_as_cube(self, **kwargs):
           pass  # Write code to plot data concatenated along common axis here.

Then you can create your own ``NDCubeSequence`` by combining your mixin with
`~ndcube.NDCubeSequenceBase` which holds all the non-plotting functionality of the
`~ndcube.NDCubeSequence`.

.. code-block:: python

    class MySequence(NDCubeSequenceBase, MySequencePlotMixin):

This will create a new class, ``MySequence``, which contains all the functionality of
`~ndcube.NDCubeSequence` plus the plot methods you've defined in ``MySequencePlotMixin``.

There are many other ways you could visualize the data in your `~ndcube.NDCubeSequence`
and many other visualization packages in the Python ecosystem that you could use.
These examples show just a few simple ways.  But hopefully this has shown you that
it's still possible to visualize the data in your `~ndcube.NDCubeSequence`,
whether by creating your own mixin, following the above examples, or by using
some other infrastructure.
