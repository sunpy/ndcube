.. _arithmetic:

*********************
Arithmetic Operations
*********************

Arithmetic operations are a crucial tool in n-dimensional data analysis.
Applications include subtracting a background from a 1-D timeseries or spectrum, scaling an image by a vignetting function, and many more.
To this end, `~ndcube.NDCube` supports addition, subtraction, multiplication, and division with numbers, arrays, and `~astropy.units.Quantity`.
Raising an `~ndcube.NDCube` to a power is also supported.
These operations return a new `~ndcube.NDCube` with the data array (and, where appropriate, the uncertainties and unit) altered in accordance with the arithmetic operation.
Other attributes of the `~ndcube.NDCube` remain unchanged.

Arithmetic operations between `~ndcube.NDCube` and coordinate-less `~astropy.nddata.NDData` subclasses are also supported.
See the section below on :ref:`arithmetic_nddata` for further details.

.. _arithmetic_standard:

Arithmetic Operations with Numbers, Arrays and Quantities
=========================================================

Addition and Subtraction
------------------------

Numbers, arrays and `~astropy.units.Quantity` can be added to and subtracted from `~ndcube.NDCube` via the ``+`` and ``-`` operators.
Note that these only change the data values of the `~ndcube.NDCube` and units must be consistent with that of the `~ndcube.NDCube`.
Let's demonstrate with an example `~ndcube.NDCube`, ``cube``:

.. expanding-code-block:: python
  :summary: Expand to see cube instantiated.

  >>> import astropy.units as u
  >>> import astropy.wcs
  >>> import numpy as np
  >>> from astropy.nddata import StdDevUncertainty

  >>> from ndcube import NDCube

  >>> # Define data array.
  >>> data = np.arange(2*3).reshape((2, 3)) + 10

  >>> # Define WCS transformations in an astropy WCS object.
  >>> wcs = astropy.wcs.WCS(naxis=2)
  >>> wcs.wcs.ctype = 'HPLT-TAN', 'HPLN-TAN'
  >>> wcs.wcs.cunit = 'deg', 'deg'
  >>> wcs.wcs.cdelt = 0.5, 0.4
  >>> wcs.wcs.crpix = 2, 2
  >>> wcs.wcs.crval = 0.5, 1

  >>> # Define mask. Initially set all elements unmasked.
  >>> mask = np.zeros_like(data, dtype=bool)
  >>> mask[0, :] = True  # Now mask some values.
  >>> # Define uncertainty, metadata and unit.
  >>> uncertainty = StdDevUncertainty(np.abs(data) * 0.1)
  >>> meta = {"Description": "This is example NDCube metadata."}
  >>> unit = u.ct

  >>> # Instantiate NDCube with supporting data.
  >>> cube = NDCube(data, wcs=wcs, uncertainty=uncertainty, mask=mask, meta=meta)

.. code-block:: python

  >>> cube.data
  array([[10, 11, 12],
         [13, 14, 15]])

  >>> new_cube = cube + 1
  >>> new_cube.data
  array([[11, 12, 13],
         [14, 15, 16]])

Note that all the data values have been increased by 1.
We can also use an array if we want to add a different number to each data element:

.. code-block:: python

  >>> import numpy as np
  >>> arr = np.arange(cube.data.size).reshape(cube.data.shape)
  >>> arr
  array([[0, 1, 2],
         [3, 4, 5]])

  >>> new_cube = cube + arr
  >>> new_cube.data
  array([[10, 12, 14],
         [16, 18, 20]])

Subtraction works in the same way.

.. code-block:: python

  >>> new_cube = cube - 1
  >>> new_cube.data
  array([[ 9, 10, 11],
         [12, 13, 14]])

  >>> new_cube = cube - arr
  >>> new_cube.data
  array([[10, 10, 10],
         [10, 10, 10]])

Note that in the above examples, ``cube`` has no unit.
This is why we are able to add and subtract numbers and arrays.
If, however, we have an `~ndcube.NDCube` with a unit assigned,

.. code-block:: python

  >>> cube_with_unit = NDCube(cube, unit=u.ct)

then adding or subtracting an array or unitless number will raise an error.
In such cases, we must use a `~astropy.units.Quantity` with a compatible unit:

.. code-block:: python

  >>> cube_with_unit.data
  array([[10, 11, 12],
         [13, 14, 15]])

  >>> new_cube = cube_with_unit + 1 * u.ct  # Adding a scalar quantity
  >>> new_cube.data
  array([[11., 12., 13.],
         [14., 15., 16.]])

  >>> new_cube = cube_with_unit - 1 * u.ct  # Subtracting a scalar quantity
  >>> new_cube.data
  array([[ 9., 10., 11.],
         [12., 13., 14.]])

  >>> new_cube = cube_with_unit + arr * u.ct  # Adding an array-like quantity
  >>> new_cube.data
  array([[10., 12., 14.],
         [16., 18., 20.]])

  >>> new_cube = cube_with_unit - arr * u.ct  # Subtracting an array-like quantity
  >>> new_cube.data
  array([[10., 10., 10.],
         [10., 10., 10.]])

Multiplication and Division
---------------------------

An `~ndcube.NDCube` can be multiplied and divided by numbers, arrays, and `~astropy.units.Quantity` via the ``*`` and ``-`` operators.
These work similarly to addition and subtraction with a few minor differences:

- The uncertainties of the resulting `~ndcube.NDCube` are scaled by the same factor as the data.
- Classes with non-equivalent units can be combined.

  * e.g. an `~ndcube.NDCube` with a unit of ``ct`` divided by an `~astropy.units.Quantity` with a unit of ``s`` will result in an `~ndcube.NDCube` with a unit of ``ct / s``.
  * This also holds for cases were unitful and unitless classes can be combined.  In such cases, the unit of the resulting `~ndcube.NDCube` will be the same as that of the unitful object.

Below are some examples.

.. code-block:: python

  >>> # See attributes of original cube.
  >>> cube_with_unit.data
  array([[10, 11, 12],
         [13, 14, 15]])
  >>> cube_with_unit.unit
  Unit("ct")
  >>> cube_with_unit.uncertainty
  StdDevUncertainty([[1. , 1.1, 1.2],
                     [1.3, 1.4, 1.5]])

  >>> # Multiply by a unitless array.
  >>> arr = 1 + np.arange(cube_with_unit.data.size).reshape(cube_with_unit.data.shape)
  >>> arr
  array([[1, 2, 3],
         [4, 5, 6]])

  >>> new_cube = cube_with_unit * arr

  >>> # Inspect attributes of resultant cube.
  >>> new_cube.data
  array([[10, 22, 36],
         [52, 70, 90]])
  >>> new_cube.unit
  Unit("ct")
  >>> new_cube.uncertainty
  StdDevUncertainty([[1. , 2.2, 3.6],
                     [5.2, 7. , 9. ]])

  >>> # Divide by a scalar astropy Quantity.
  >>> new_cube = cube_with_unit / (2 * u.s)

  >>> # Inspect attributes of resultant cube.
  >>> new_cube.data
  array([[5. , 5.5, 6. ],
         [6.5, 7. , 7.5]])
  >>> new_cube.unit
  Unit("ct / s")
  >>> new_cube.uncertainty
  StdDevUncertainty([[0.5 , 0.55, 0.6 ],
                     [0.65, 0.7 , 0.75]])

Note that when performing arithmetic operations with `~ndcube.NDCube` and array-like objects, their shapes only have to be broadcastable, not necessarily the same.
For example:

.. code-block:: python

  >>> cube.data
  array([[10, 11, 12],
         [13, 14, 15]])
  >>> arr[0]
  array([1, 2, 3])

  >>> new_cube = cube + arr[0]
  >>> new_cube.data
  array([[11, 13, 15],
         [14, 16, 18]])

Raising NDCube to a Power
-------------------------

.. code-block:: python

  >>> cube_with_unit.data
  array([[10, 11, 12],
         [13, 14, 15]])

  >>> import warnings
  >>> with warnings.catch_warnings():
  ...     warnings.simplefilter("ignore")  # Catching warnings not needed but keeps docs cleaner.
  ...
  ...     new_cube = cube_with_unit**2

  >>> new_cube.data
  array([[100, 121, 144],
         [169, 196, 225]])
  >>> new_cube.unit
  Unit("ct2")
  >>> (new_cube.mask == cube_with_unit.mask).all()
  np.True_

Note that error propagation is delegated to the ``cube.uncertainty`` object.
Therefore, if this class supports error propagation by power, then ``new_cube`` will include uncertainty.
Otherwise, ``new_cube.uncertainty`` will be set to ``None``.


.. _arithmetic_nddata:

Arithmetic Operations Between NDCubes
=====================================

Why Arithmetic Operations between NDCubes Are Not Supported Directly (but Are Indirectly)
-----------------------------------------------------------------------------------------

Arithmetic operations between two `~ndcube.NDCube` instances are not supported directly.
(However, as we shall see, they are supported indirectly.)
This is because of the wide scope for enabling non-sensical coordinate-aware operations.
For example, what does it mean to multiply a spectrum and an image?
Getting the difference between two images may make physical sense in certain circumstances.
For example, subtracting two sequential images of the same region of the Sun is a common step in many solar image analyses.
But subtracting images of different parts of the sky, e.g. the Crab and Horseshoe Nebulae, does not produce a physically meaningful result.
Even when subtracting two images of the Sun, drift in the telescope's pointing may result in corresponding pixels representing different points on the Sun.
In this case, it is questionable whether even this operation makes physical sense.
Moreover, in all of these cases, it is not clear what the resulting WCS object should be.

One way to ensure physically meaningful, coordinate-aware arithmetic operations between `~ndcube.NDCube` instances would be to compare their WCS objects are the same within a certain tolerance.
Alternatively, the arithmetic operation could attempt to reproject one `~ndcube.NDCube` to the other's WCS.
However, these operations can be prohibitively slow and resource-hungry.
Despite this, arithmetic operations between two `~ndcube.NDCube` instances is supported, provided the coordinate-awareness of one is dropped.
Below we shall outline two ways of doing this.

Performing Arithmetic Operations between NDCubes Indirectly
-----------------------------------------------------------

Extracting One NDCube's Data and Unit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way to perform arithmetic operations between `~ndcube.NDCube` instances is to directly combine one with the data (an optionally the unit) of the other.
Thus, the operation can be performed as already described in the above section on :ref:`arithmetic_standard`:

.. expanding-code-block:: python
  :summary: Expand to see definition of cube1 and cube2.

  >>> cube1 = cube_with_unit
  >>> cube2 = cube_with_unit / 4

.. code-block:: python

  >>> new_cube = cube1 - cube2.data * cube2.unit

Enabling Arithmetic Operations between NDCubes with NDCube.to_nddata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes, however, more advanced arithmetic operations are required for which numbers, arrays and `~astropy.units.Quantity` are insufficient.
These include cases where both operands have:

- uncertainties which need to be propagated, and/or;
- masks that need to be combined, and/or;
- units and non-`numpy` data arrays unsuitable for representation as an `~astropy.units.Quantity`, e.g. ``dask.array``.

To achieve these operations, it would be preferable to perform arithmetic operations directly between the `~ndcube.NDCube` instances.
While this is not supported, as already outlined, the same result can be achieved by first dropping the coordinate-awareness of one `~ndcube.NDCube` via the `ndcube.NDCube.to_nddata` method.
The two datasets can then be combined using the standard arithmetic operators.
`ndcube.NDCube.to_nddata` enables the conversion of the `~ndcube.NDCube` instance to any `~astropy.nddata.NDData` subclass, while also enabling the values of specific attributes to be altered during the conversion.
Therefore, arithmetic operations between `~ndcube.NDCube` instances via:

.. code-block:: python

  >>> new_cube = cube1 + cube2.to_nddata(wcs=None)

where addition, subtraction, multiplication and division are all enabled by the ``+``, ``-``, ``*``, and ``/`` operators, respectively.

Note that `~ndcube.NDCube` attributes not supported by the constructor of the output type employed by `ndcube.NDCube.to_nddata` are dropped by the conversion.
Therefore, since `ndcube.NDCube.to_nddata` converts to `~astropy.nddata.NDData` by default, there was no need in the above example to explicitly set `~ndcube.NDCube.extra_coords` and `~ndcube.NDCube.global_coords` to ``None``.
Note that the output type of `ndcube.NDCube.to_nddata` can be controlled via the ``nddata_type`` kwarg.
For example:

  >>> from astropy.nddata import NDDataRef
  >>> nddataref2 = cube2.to_nddata(wcs=None, nddata_type=NDDataRef)
  >>> print(type(nddataref2) is NDDataRef)
  True

Requiring users to explicitly remove coordinate-awareness makes it clear that coordinates are not combined as part of arithmetic operations.
It also makes it unambiguous which operand's coordinates are maintained through the operation.

`ndcube.NDCube.to_nddata` is not limited to changing/removing the WCS.
The value of any input supported by the ``nddata_type``'s constructor can be altered by setting a kwarg for that input, e.g.:

.. code-block:: python

  >>> nddata_ones = cube2.to_nddata(data=np.ones(cube2.data.shape))
  >>> nddata_ones.data
  array([[1., 1., 1.],
         [1., 1., 1.]])

Handling of Data, Units and Meta
""""""""""""""""""""""""""""""""
The treatment of the ``data`` and ``unit`` attributes in operations between `~ndcube.NDCube` and coordinate-less `~astropy.nddata.NDData` subclasses are the same as for arrays and `~astropy.units.Quantity`.
However, only the metadata from the `~ndcube.NDCube` is retained.
This can be updated after the operation, if desired.
For example:

.. code-block:: python

  >>> cube1.meta
  {'Description': 'This is example NDCube metadata.'}

  >>> cube2.meta["more"] = True
  >>> cube2.meta
  {'Description': 'This is example NDCube metadata.', 'more': True}

  >>> new_cube = cube1 + cube2.to_nddata(wcs=None)
  >>> new_cube.meta
  {'Description': 'This is example NDCube metadata.'}

  >>> new_cube.meta.update(cube2.meta)
  >>> new_cube.meta
  {'Description': 'This is example NDCube metadata.', 'more': True}

Handling of Uncertainties
"""""""""""""""""""""""""
How uncertainties are handled depends on the uncertainty types of the operands:

- ``NDCube.uncertainty`` and ``NDData.uncertainty`` are both ``None`` => ``new_cube.uncertainty`` is ``None``;
- ``NDCube`` or ``NDData`` have uncertainty, but not both => the existing uncertainty is assigned to ``new_cube`` as is;
- ``NDCube`` and ``NDData`` both have uncertainty => uncertainty propagation is delegated to the ``NDCube.uncertainty.propagate`` method.

  * Note that not all uncertainty classes support error propagation, e.g. `~astropy.nddata.UnknownUncertainty`.  In such cases, uncertainties are dropped altogether and ``new_cube.uncertainty`` is set to ``None``.

If users would like to remove uncertainty from one of the operands in order to propagate the other without alteration, this can be done by casting the `~ndcube.NDCube` to a new instance with the uncertainty set to ``None`` via the `ndcube.NDCube.to_nddata` method before the operation:

.. code-block:: python

  >>> # Remove uncertainty from NDCube
  >>> new_cube = cube1.to_nddata(uncertainty=None, nddata_type=NDCube) + cube2.to_nddata(wcs=None)

Handling of Masks and NDCube.fill_masked Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The mask resulting from an arithmetic operation between an `~ndcube.NDCube` and coordinate-less `~astropy.nddata.NDData` subclass depends on the mask types of the operands:

- ``NDCube.mask`` and ``NDData.mask`` are both ``None`` => ``new_cube.mask`` is ``None``;
- ``NDCube`` or ``NDData`` have a mask, but not both => the existing mask is assigned to ``new_cube`` as is;
- ``NDCube`` and ``NDData`` both have masks => The masks are combined via `numpy.logical_or`.

The mask values do not affect the ``data`` values output by the operation.
However, in some cases, the mask may be used to identify regions of unreliable data that should not be included in the operation.
This can be achieved by altering the masked data values before the operation via the `ndcube.NDCube.fill_masked` method.

The NDCube.fill_masked Method
"""""""""""""""""""""""""""""

The `ndcube.NDCube.fill_masked` method returns a new `~ndcube.NDCube` instance with masked data elements (and optionally uncertainty elements) replaced with a user-defined ``fill_value``.
This can be used to effectively exclude masked values from an arithmetic operation by replacing masked values with the identity value for that operation.
For example, in the case of addition and subtraction, the identity ``fill_value`` is ``0``.

.. code-block:: python

  >>> new_cube = cube1.fill_masked(0) + cube2.to_nddata(wcs=None)

In this example, both operands have uncertainties, which means masked uncertainties are propagated through the operation, even though the masked data values have been set to ``0``.
Propagation of masked uncertainties can also be suppressed by setting the optional kwarg, ``uncertainty_fill_value`` to ``0``.

By default, the mask of the filled `~ndcube.NDCube` cube is not changed, and therefore is incorporated into the mask of ``new_cube``.
However, mask propagation can also be suppressed by unmasking the filled `~ndcube.NDCube`.
This can be done by setting the optional kwarg, ``unmask=True``, in `ndcube.NDCube.fill_masked`, which sets the mask of the filled `~ndcube.NDCube` to ``False``.

In the case of multiplication and division, the identity ``fill_value`` is ``1``.  (Note that in the below example we show the optional use of the ``uncertainty_fill_value`` and ``unmask`` kwargs.)

.. code-block:: python

  >>> cube_filled = cube1.fill_masked(1, uncertainty_fill_value=0, unmask=True)
  >>> new_cube = cube_filled * cube2.to_nddata(wcs=None)

Note that irrespective of the arithmetic operation, the ``uncertainty_fill_value`` should always be set to ``0`` to avoid propagating masked uncertainties.

By default, `ndcube.NDCube.fill_masked` returns a new `~ndcube.NDCube` instance.
However, in some cases it may be preferable to fill the masked values in-place, for example, because the data are very large and users want to control the number of copies in RAM.
In this case, the ``fill_in_place`` kwarg can be used.

.. code-block:: python

  >>> cube1.fill_masked(0, fill_in_place=True)
  >>> new_cube = cube1 + cube2.to_nddata(wcs=None)
