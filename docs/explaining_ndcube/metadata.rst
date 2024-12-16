.. _metadata:

*****************
Handling Metadata
*****************

`ndcube`'s data objects do not enforce any requirements on the object assigned to their ``.meta`` attributes.
However, it does provide an optional object for handling metadata, `~ndcube.NDMeta`.
This class inherits from `dict`, but provides additional functionalities, chief among which is the ability to associate metadata with data array axes.
This enables `~ndcube.NDMeta` to update itself through certain operations, e.g. slicing, so that the metadata remains consistent with the associated ND object.
In this section, we explain the needs that `~ndcube.NDMeta` serves, the concepts underpinning it, and the functionalities it provides.

.. _meta_concepts:

Key Concepts
============

.. _coords_vs_meta:

Coordinates vs. Axis-aware Metadata
-----------------------------------

The difference between coordinates and axis-aware metadata is a subtle but important one.
Formally, a coordinate is a physical space sampled by one or more data axis, whereas axis-aware metadata is information describing the data that can alter along one or more physical dimension.
An informative example is the difference between time and exposure time.
The temporal axis of a 3-D image cube samples the physical dimension of time in a strictly increasing monotonic way.
The times along the temporal axis are therefore coordinate values, not metadata.
Additionally, a scalar timestamp of a 2-D image is also considered a coordinate in the `ndcube` framework.
This is because it describes where in the physical dimension of time the data has been sampled, even though it does not correspond to a data axis.
Because of this, such scalar coordinates are stored in `~ndcube.GlobalCoords`, while coordinates associated with array/pixel axes are stored in the WCS or `~ndcube.ExtraCoords`.
(See the :ref:`global_coords` section for more discussion on the difference between global and other coordinates.)

By contrast, exposure time describes the interval over which each image was accumulated.
Exposure time can remain constant, increase or decrease with time, and may switch between these regimes during the time extent of the image cube.
Like a coordinate, it should be associated with the image cube's temporal axis.
However, exposure time is reflective of the telescope's operational mode, not a sampling of a physical dimension.
Exposure time is therefore metadata, not a coordinate.

One reason why it is important to distinguish between coordinates and axis-aware metadata is `ndcube`'s dependence on WCS.
Most WCS implementations require that there be a unique invertible mapping between pixel and world coordinates, i.e., there is only one pixel value that corresponds to a specific real world value (or combination of such if the coordinate is multi-dimensionsal), and vice versa.
Therefore, while there may be exceptions for rare and exotic WCS implementations, a good rule of thumb for deciding whether something is a coordinate is:
coordinates are numeric and strictly monotonic.  Otherwise you have metadata.

The keen-eyed reader may have realised of the above framework that, while not all axis-aligned metadata can be treated as coordinates, all coordinates can be treated like axis-aware metadata.
This raises the question of why not dispense with coordinates altogether and only have axis-aligned metadata?
The reason is that the stricter requirements on coordinates have led to a host of powerful coordinate infrastructure that are not valid for generalised axis-aware metadata.
These include functional WCS implementations which save memory as well as saving compute time through operations such as interpolation, and `~astropy.visualization.wcsaxes.WCSAxes`, which make complicated coordinate-aware plotting easy.
Therefore, where appropriate, it is beneficial to store coordinates separately from axis-aware metadata.

.. _axis_and_grid_aligned_metadata:

Types of Axis-aware Metadata: Axis-aligned vs. Grid-aligned
-----------------------------------------------------------

There are two types of axis-aware metadata: axis-aligned and grid-aligned.
Axis-aligned metadata associates a scalar or string with an array axis.
It can also assign an array of scalars or strings to multiple array axes, so long as there is one value per associated axis.
For example, the data produced by a scanning slit spectrograph is associated with real world values.
But each axis also corresponds to features of the instrument: dispersion (spectral), pixels along the slit (spatial), position of the slit in the rastering sequence (spatial and short timescales), and the raster number (longer timescales).
The axis-aligned metadata concept allows us to avoid ambiguity by assigning each axis with a label (e.g. ``("dispersion", "slit", "slit step", "raster")``).

By contrast, grid aligned metadata assigns a value to each pixel along axes.
The exposure time discussion above is an example of 1-D grid-aligned metadata.
However, grid-aligned metadata can also be multi-dimensional.
For example, a pixel-dependent response function could be represented as grid-aligned metadata associated with 2 spatial axes.

`~ndcube.NDMeta` supports both axis-aligned and grid-aligned metadata with the same API, which will be discussed in the next section.

.. _ndmeta:


NDMeta
======

.. _initializing_ndmeta:

Initializing an NDMeta
----------------------

To initialize an `~ndcube.NDMeta`, simply provide it with a `~collections.abc.Mapping` object, e.g. a `dict` or `astropy.io.fits.header.Header`.

.. code-block:: python

  >>> from ndcube import NDMeta
  >>> raw_meta = {"salutation": "hello", "name": "world"}
  >>> meta = NDMeta(raw_meta)

We can now access each piece of metadata by indexing ``meta`` as if it were a `dict`:

.. code-block:: python

  >>> meta["name"]
  'world'

In this example we have provided a very simple set of metadata.
In fact, it is so simple that there is no practical difference between ``meta`` and a simple `dict`.
To demonstrate one of the additional functionalities of `~ndcube.NDMeta`, let reinstantiate ``meta``, adding some comments to the metadata.
To do this, we provide another `~collections.abc.Mapping`, e.g. a `dict`, with the same keys as the main metadata keys, or a subset of them, to the ``key_comments`` kwarg.

.. code-block:: python


  >>> key_comments = {"name": "Each planet in the solar system has a name."}
  >>> meta = NDMeta(raw_meta, key_comments=key_comments)

We can now access the comments by indexing the `~ndcube.NDMeta.key_comments` property:

.. code-block:: python

  >>> meta.key_comments["name"]
  'Each planet in the solar system has a name.'

Now let's discuss how to initialize how to `~ndcube.NDMeta` with axis-aware metadata.
(Here, we will specifically consider grid-aligned metadata.  Axis-aligned metadata is assigned in the same way.  But see the :ref:`assigning_axis_aligned_metadata` section for more details.)
Similar to ``key_comments``, we assign metadata to axes by providing a `~collections.abc.Mapping`, e.g. a `dict`, via its ``axes`` kwarg.
And like with ``key_comments``, the keys of ``axes`` must be the same, or a subset of, the main metadata keys.
The axis value must be an `int` or `tuple` of `int` giving the array axes of the data that correspond to the axes of the metadata.
Note that this means that metadata can be multidimensional.
Let's say we want to add exposure time that varies with the 1st (temporal) axis of that data, and a pixel response that varies with time and pixel column (1st and 3rd axes).

.. code-block:: python

  >>> import astropy.units as u
  >>> import numpy as np
  >>> raw_meta["exposure time"] = [1.9, 2.1, 5, 2, 2] * u.s
  >>> raw_meta["pixel response"] = np.array([[100., 100., 100., 90., 100.], [85., 100., 90., 100., 100.]]) * u.percent
  >>> axes = {"exposure time": 0, "pixel response": (0, 2)}
  >>> meta = NDMeta(raw_meta, axes=axes)

It is easy to see which axes a piece of metadata corresponds to by indexing the `~ndcube.NDMeta.axes` property:

.. code-block:: python

  >>> meta.axes["exposure time"]
  array([0])
  >>> meta.axes["pixel response"]
  array([0, 2])

Finally, it is possible to attach the shape of the associated data to the `~ndcube.NDMeta` instance via the ``data_shape`` kwarg:

.. code-block:: python

  >>> meta = NDMeta(raw_meta, axes=axes, key_comments=key_comments, data_shape=(5, 1, 2))

Or by directly setting the ``~ndcube.NDMeta.data_shape`` property after instantiation:

.. code-block:: python

  >>> meta = NDMeta(raw_meta, axes=axes, key_comments=key_comments)
  >>> meta.data_shape = (5, 1, 2)

Note that the ``data_shape`` must be compatible with the shapes and associated axes of any axis-aware metadata.
For example, we couldn't set the length of the first axis to ``6``, because ``meta["exposure time"]`` is associated with the first axis and has a length of ``5``.
If no ``data_shape`` is provided, it is determined from the axis-aware metadata, if any is provided.
See the :ref:`data_shape` section for more details.

.. _adding_removing_metadata:

Adding and Removing Metadata
----------------------------

Because `~ndcube.NDMeta` is a subclass of `dict`, it is possible to add new metadata via the simple ``__setitem__`` API, e.g ``meta[new_key] = new_value``.
However, this API is not sufficient if we want to add axis-aware or commented metadata.
This is why `~ndcube.NDMeta` provides an `~ndcube.NDMeta.add` method.
This method requires the key and value of the new metadata, an optionally accepts a comment and/or axes.
Let's use this method to add a voltage that varies with time, i.e. the first data axis.

.. code-block:: python

  >>> meta.add("voltage", u.Quantity([1.]*5, unit=u.V), key_comment="detector bias voltage can vary with time and pixel column.", axes=(0,))
  >>> meta["voltage"]
  <Quantity [1., 1., 1., 1., 1.] V>

If you try to add metadata with a pre-existing key, `~ndcube.NDMeta.add` will error.
To replace the value, comment, or axes values of pre-existing metadata, set the ``overwrite`` kwarg to ``True``.

.. code-block:: python

  >>> meta.add("voltage", u.Quantity([-300.]*5, unit=u.V), key_comment="detector bias voltage", axes=(0,), overwrite=True)
  >>> meta["voltage"]
  <Quantity [-300., -300., -300., -300., -300.] V>

Unwanted metadata can be removing by employing the ``del`` operator.

.. code-block:: python

  >>> del meta["voltage"]
  >>> meta.get("voltage", "deleted")
  'deleted'

Note that the ``del`` operator also removes associated comments and axes.

.. code-block:: python

  >>> meta.key_comments.get("voltage", "deleted")
  'deleted'
  >>> meta.axes.get("voltage", "deleted")
  'deleted'

.. _data_shape:

Data Shape
----------

The `~ndcube.NDMeta.data_shape` property tracks the shape of the data with which the metadata is associated.
We have already seen in the :ref:`initializing_ndmeta` section, that it can be assigned during initialization or by subsequently setting the `~ndcube.NDMeta.data_shape` property directly.
However, if the ``data_shape`` is not provided, it is inferred from the shapes of axis-aware metadata.
If no axis-aware metadata is present, `~ndcube.NDMeta.data_shape` is empty:

.. code-block:: python

  >>> from ndcube import NDMeta
  >>> raw_meta = {"salutation": "hello", "name": "world"}
  >>> meta = NDMeta(raw_meta)
  >>> meta.data_shape
  array([], dtype=int64)

If we now add the ``"pixel response"`` metadata that we used, earlier the `~ndcube.NDMeta.data_shape` will be updated.

.. code-block:: python

  >>> meta.add("pixel response", np.array([[100., 85], [100., 100], [100., 90], [90., 100.], [100., 100.]]) * u.percent, axes=(0, 2))
  >>> meta.data_shape
  array([5, 0, 2])

Note that since ``"pixel response"`` is associated with the 1st and 3rd axes, those axes now have the same shape as ``"pixel response"``.
The existence of a 3rd axis, implies the presence of a 2nd.
However, we have no metadata associated with it, and hence no knowledge of its length.
It has therefore been assigned a length of ``0``.

Now that the shape has been set for the 1st and 3rd axes, subsequently added grid-aligned metadata associated with those axes must be compatible with those axis lengths.
For example, if we add a 1-D ``"exposure time"`` and associate it with the 1st axis, it must have a length of of ``5``, otherwise an error will be raised:

.. code-block:: python

  >>> meta.add("exposure time", [1.9, 2.1, 5, 2, 2] * u.s, axes=0)

Moreover, if we now directly set the `~ndcube.NDMeta.data_shape` via ``meta.data_shape = new_shape``, we cannot change the length of axes already associated with grid-aligned metadata, without first removing or altering that metadata.
However, these restrictions do not apply if we want to change the shape of the 2nd axis, or add new metadata to it, because its length is ``0``, and hence considered undefined.

.. code-block:: python

  >>> meta.add("row temperature", [-10, -11, -12] * u.deg_C, axes=1)
  >>> meta.data_shape
  array([5, 3, 2])

.. _assigning_axis_aligned_metadata:

Assigning Axis-aligned Metadata
-------------------------------

So far, we have only dealt with grid-aligned metadata, i.e. axis-aware metadata which provides a value for each pixel.
To provide axis-aligned metadata, i.e. where each axis has a single value (see :ref:`axis_and_grid_aligned_metadata`), provide a scalar or string for a single axis, or a 1-D array-like with the same length as the number of associated axes for multi-axis-aligned metadata.

.. code-block:: python

  >>> meta.add("axis name", np.array(["a", "b", "c", "d"]), axes=(0, 1, 2, 3))

Note that the length of ``"axis name"`` is the same as the number of its associated axes.
Also note that we have now indicated that there is 4th axis.
``meta.data_shape`` has therefore been automatically updated accordingly.

.. code-block:: python

  >>> meta.data_shape
  array([5, 3, 2, 0])

However, because axis-aligned metadata does not tell us about the length of the axes, the 4th axis has been assigned a length of zero.
