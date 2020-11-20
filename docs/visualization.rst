.. _plotting

=============
Visualization
=============

.. _ndcube_plotting

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
If you feel that `~ndcube.NDCubeSequence` visualization should be supported again, please let us know by commenting on that issue and telling us of your use case.  Better still, if you would like to worl on the infrastructure required to support `~ndcube.NDCubeSequence` visualization is a post ndcube 2.0 world let us know by commenting on the issue.
