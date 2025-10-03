.. _arithmetic:

*********************
Arithmetic Operations
*********************

Arithmetic operations are a crucial tool in n-dimensional data analysis.
Applications include subtracting a background from a 1-D timeseries or spectrum, scaling an image by a vignetting function, and many others.
To aid with such workflows, `~ndcube.NDCube` supports addition, subtraction, multiplication, and division with numbers, arrays, `~astropy.units.Quantity`.
Raising an `~ndcube.NDCube` to a power is also supported.
These operations return a new `~ndcube.NDCube` with the data array (and, where appropriate, the uncertainties) altered in accordance with the arithmetic operation.
Other attributes of the `~ndcube.NDCube` remain unchanged.

In addition, combining `~ndcube.NDCube` with coordinate-less `~astropy.nddata.NDData` subclasses via these operations is also supported.
Such operations can be more complicated.  See the section below on :ref:`arithmetic_nddata` for a discussion separate, more detailed discussion.

.. _arithmetic_standard:

Standard Arithmetic Operations
==============================

Addition and Subtraction with Numbers, Arrays and Quantities
------------------------------------------------------------

Numbers, arrays and `~astropy.units.Quantity` can be added to and subtracted from `~ndcube.NDCube` via the ``+`` and ``-`` operators.
Note that addition and subtraction only change the data values of the `~ndcube.NDCube`.
Let's deomonstrate with an example `~ndcube.NDCube`, ``cube``

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
We can also add an array if we want to add a different number to each data element:

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

Note that int he above examples, ``cube`` has no unit.
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

Multiplying and Dividing with Numbers, Arrays and Quantities
------------------------------------------------------------

An `~ndcube.NDCube` can be multiplied and divided by numbers, arrays, and `~astropy.units.Quantity` via the ``*`` and ``-`` operators.
These work similarly to addition and subtraction with a few minor differences:

- The uncertainties of the resulting `~ndcube.NDCube` are scaled by the same factor as the data.
- Classes with different units can be combined.

  * e.g. an `~ndcube.NDCube` with a unit of counts divided by an `~astropy.units.Quantity` with a unit is seconds will result in an `~ndcube.NDCube` with a unit of counts per second.
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

  >>> # Divide by an astropy Quantity.
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

Note that when performing arithmetic operations with `~ndcube.NDCube` and array-like objects, their shapes only have to be broadcastable.
For example:

Raising NDCube to a Power
-------------------------

`~ndcube.NDCube` can be raised to a power.

.. code-block:: python

  >>> cube_with_unit.data
  array([[10, 11, 12],
         [13, 14, 15]])

  >>> new_cube = cube_with_unit**2

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

Arithmetic Operations with Coordinate-less NDData
=================================================

Sometimes more advanced arithmetic operations are required.
For example, we may want to create a sequence of running difference images which highlight changes between frames, and propagate the uncertainties associated with each image.
Alternatively, we may want to subtract one image from another, but exclude a certain region of the image with a mask.
In such cases, numbers, arrays and `~astropy.units.Quantity` are insufficient, and we would like to subtract two `~ndcube.NDCube` objects.
This is not directly supported, but can still be achieved in practice, as we shall see below.

Why Arithmetic Operations with Coordinate-aware NDData Are Not Directly Supported, and How This Can Be Overcome
---------------------------------------------------------------------------------------------------------------

Arithmetic operations between two `~ndcube.NDCube` instances (or equivalently, an `~ndcube.NDCube` and another coordinate-aware `~astropy.nddata.NDData` subclass) are not supported because of the possibility of supporting non-sensical operations.
For example, what does it mean to multiply a spectrum and an image in a coordinate-aware way?
Getting the difference between two images may make physical sense, but only in certain circumstances.
For example, subtracting two sequential images of the same region of the Sun is a common step in many solar image analysis workflows.
However, subtracting images of different parts of the sky, e.g. the Sun and the Crab Nebula, does not result in a physically meaningful image.
Even when subtracting two images of the Sun, drift in the telescope's pointing may result in the pixels in each image corresponding to different points in the Sun.
In this case, it is questionable whether this operation makes physical sense after all.
Moreover, in all of these cases, it is not at all clear what the resulting WCS object should be.

In many cases, a simple solution would be to extract the data (an optionally the unit) from one of the `~ndcube.NDCube` instances and perform the operation as described in the above section on :ref:`arithmetic_standard`:

.. expanding-code-block:: python
  :summary: Expand to see definition of cube1 and cube2.

  >>> cube1 = cube_with_unit
  >>> cube2 = cube_with_unit / 4

.. code-block:: python

  >>> new_cube = cube1 - cube2.data * cube2.unit

However, this does not allow for the propagation of uncertainties or masks associated with ``cube2``.
Therefore, `~ndcube.NDCube` does support arithmetic operations with instances of `~astropy.nddata.NDData` subclasses whose ``wcs`` attribute is ``None``.
This makes users explicitly aware that they are dispensing with coordinate-awareness on one of their operands.
It also leaves only one WCS involved in the operation, thus removing ambiguity regarding the WCS of the `~ndcube.NDCube` resulting from the operation.

Users who would like to drop coordinate-awareness from an `~ndcube.NDCube` can so simply by converting it to an `~astropy.nddata.NDData` and setting the ``wcs`` to ``None``:

.. code-block:: python

  >>> from astropy.nddata import NDData

  >>> cube2_nocoords = NDData(cube2, wcs=None)


Performing Arithmetic Operations with Coordinate-less NDData
------------------------------------------------------------

Addition, subtraction, multiplication and division between `~ndcube.NDCube` and coordinate-less `~astropy.nddata.NDData` classes are all supported via the ``+``, ``-``, ``*``, and ``/`` operators.
With respect to the ``data`` and ``unit`` attributes, the behaviors are the same as for arrays and `~astropy.units.Quantity`.
The power of using coordinate-less `~astropy.nddata.NDData` classes is the ability to handle uncertainties and masks.

Uncertainty Propagation
***********************

The uncertainty associated with the `~ndcube.NDCube` resulting from the arithmetic operation depends on the uncertainty types of the operands:

- ``NDCube.uncertainty`` and ``NDData.uncertainty`` are both ``None`` => ``new_cube.uncertainty`` is ``None``;
- ``NDCube`` or ``NDData`` have uncertainty, but not both => the existing uncertainty is assigned to ``new_cube`` as is;
- ``NDCube`` and ``NDData`` both have uncertainty => uncertainty propagation is delegated to the ``NDCube.uncertainty.propagate`` method.

  * Note that not all uncertainty classes support error propagation, e.g. `~astropy.nddata.UnknownUncertainty`.  In such cases, uncertainties are dropped altogether and ``new_cube.uncertainty`` is set to ``None``.

If users would like to remove uncertainty from one of the operands in order to propagate the other without alteration, this can be done before the arithmetic operation via:

.. code-block:: python

  >>> # Remove uncertainty from NDCube
  >>> cube1_nouncert = NDCube(cube2, wcs=None)
  >>> new_cube = cube1_nouncert + cube2_nocoords

  >>> # Remove uncertainty from coordinate-less NDData
  >>> cube2_nocoords_nouncert = NDData(cube2, wcs=None, uncertainty=None)
  >>> new_cube = cube1 / cube2_nocoords_nouncert

Mask Operations
***************

The mask associated with the `~ndcube.NDCube` resulting from the arithmetic operation depends on the mask types of the operands:

- ``NDCube.mask`` and ``NDData.mask`` are both ``None`` => ``new_cube.mask`` is ``None``;
- ``NDCube`` or ``NDData`` have a mask, but not both => the existing mask is assigned to ``new_cube`` as is;
- ``NDCube`` and ``NDData`` both have masks => The masks are combined via `numpy.logical_or`.

The mask values do not affect the ``data`` values output by the operation.
However, in some cases, the mask may be used to identify regions of unreliable data that should not be included in the operation.
This can be achieved by altering the masked data values before the operation via the `ndcube.NDCube.fill_masked` method.
In the case of addition and subtraction, the ``fill_value`` should be ``0``.

.. code-block:: python

  >>> cube_filled = cube1.fill_masked(0)
  >>> new_cube = cube_filled + cube2_nocoords

By replacing masked data values with ``0``, these pixels are effectively not included in the addition, and the data values from ``cube2_nocoords`` are passed into ``new_cube`` unchanged.
In the above example, both operands have uncertainties, which means masked uncertainties are propagated through the addition, even though the masked data values have been set to ``0``.
Propagation of masked uncertainties can also be suppressed by setting the optional kwarg, ``fill_uncertainty_value=0``.
By default, the mask of ``cube_filled`` is not changed, and therefore is incorporated into the mask of the output cube.
However, mask propagation can also be suppressed by setting the optional kwarg, ``unmask=True``, which sets ``cube_filled0.mask`` to ``False``.

In the case of multiplication and division, and ``fill_value`` of ``1`` will prevent masked values being including in the operations:

.. code-block:: python

  >>> cube_filled = cube1.fill_masked(1, fill_uncertainty_value=0, unmask=True)
  >>> new_cube = cube_filled * cube2_nocoords

By default, `ndcube.NDCube.fill_masked` returns a new `~ndcube.NDCube` instance.
However, in some case it may be preferable to fill the masked values in-place, e.g. because the data within the `~ndcube.NDCube` is very large and users want to control the number of copies in RAM.
In this case, the ``fill_in_place`` can be used.

.. code-block:: python

  >>> cube1.fill_masked(0, fill_in_place=True)
  >>> new_cube = cube1 + cube2_nocoords
