.. _arithmetic:

*********************
Arithmetic Operations
*********************

Arithmetic operations are a crucial tool in n-dimensional data analysis.
Applications include subtracting a background from a 1-D timeseries or spectrum, scaling an image by a vignetting function, any many others.
To aid with such workflows, `~ndcube.NDCube` supports addition, subtraction, multiplication, and division with numbers, arrays, `~astropy.units.Quantity`.
Raising an `~ndcube.NDCube` to a power is also supported.
These operations return a new `~ndcube.NDCube` with the data array (and where appropriate, the uncertainties) altered in accordance with the arithmetic operation.
Other attributes of the `~ndcube.NDCube` remain unchanged.

In addition, combining `~ndcube.NDCube` with coordinate-less `~astropy.nddata.NDData` subclasses via these operations is important.
Such operations can be more complicated.  Hence see the :ref:`arithmetic_nddata` section below for a discussion separate, more detailed discussion.

.. _arithmetic_standard:

Standard Arithmetic Operations
==============================

Addition and Subtraction with Numbers, Arrays and Quantities
------------------------------------------------------------

Numbers, arrays and `~astropy.units.Quantity` can be added to and subtracted from an `~ndcube.NDCube` via the ``+`` and ``-`` operators.
Note that addition and subtraction only changes the data values of the `~ndcube.NDCube`.
Let's deomonstrate with an example `~ndcube.NDCube` called ``cube``

.. expanding-code-block:: python
  :summary: Expand to see my_cube instantiated.

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
  array([[10., 11., 12.],
         [13., 14., 15.]])
  >>> new_cube = cube + 1
  >>> new_cube.data
  array([[11., 12., 13.],
         [14., 15., 16.]])

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
  array([[10., 12., 14.],
         [16., 18., 20.]])

Subtraction works in the same way.

.. code-block:: python

  >>> new_cube = cube - 1
  >>> new_cube.data
  array([[ 9., 10., 11.],
         [12., 13., 14.]])
  >>> new_cube = cube - arr
  >>> new_cube.data
  array([[10., 10., 10.],
         [10., 10., 10.]])

Note that ``cube`` has no unit, which is why we are able to add and subtract numbers and arrays.
If, however, we have an `~ndcube.NDCube` with a unit assigned,

.. code-block:: python

  >>> cube_unitful = NDCube(cube, unit=u.ct)

then adding or subtracting an array or unitless number will raise an error.
In such cases, we must use a `~astropy.unit.Quantity` with a compatible unit:

.. code-block:: python

  >>> cube_unitful.data
  array([[10., 11., 12.],
         [13., 14., 15.]])
  >>> new_cube = cube_unitful + 1 * u.ct  # Adding a scalar quantity
  >>> new_cube.data
  array([[11., 12., 13.],
         [14., 15., 16.]])
  >>> new_cube = cube_unitful - 1 * u.ct  # Subtracting a scalar quantity
  >>> new_cube.data
  array([[ 9., 10., 11.],
         [12., 13., 14.]])
  >>> new_cube = cube_unitful + arr * u.ct  # Adding an array-like quantity
  >>> new_cube.data
  array([[10., 12., 14.],
         [16., 18., 20.]])
  >>> new_cube = cube_unitful - arr * u.ct  # Subtracting an array-like quantity
  >>> new_cube.data
  array([[10., 10., 10.],
         [10., 10., 10.]])

Multiplying and Dividing with Numbers, Arrays and Quantities
------------------------------------------------------------

An `~ndcube.NDCube` can be multiplied and divided by numbers, arrays, and `~astropy.units.Quantity` via the ``*`` and ``-`` operators.
These work similarly to addition and subtraction with a few minor differences:
- The uncertainties of the resulting `~ndcube.NDCube` are scaled by the same factor as the data.
- Classes with different units can be combined.
  - e.g. an `~ndcube.NDCube` with a unit of counts divided by an `~astropy.units.Quantity` with a unit is seconds will result in an `~ndcube.NDCube` with a unit of counts per second.
  - This also holds for cases were unitful and unitless classes can be combined.  In such cases, the unit of the resulting `~ndcube.NDCube` will be the same as that of the unitful object.

Below are some examples.

.. code-block:: python

  >>> # See attributes of original cube.
  >>> cube_unitful.data
  array([[10., 11., 12.],
         [13., 14., 15.]])
  >>> cube_unitful.unit
  Unit("ct")
  >>> cube_unitful.uncertainty
  StdDevUncertainty([[1. , 1.1, 1.2],
                   [1.3, 1.4, 1.5]])

  >>> # Multiply by a unitless array.
  >>> arr = 1 + np.arange(cube_unitful.data.size).reshape(cube_unitful.data.shape)
  >>> arr
  array([[1, 2, 3],
         [4, 5, 6]])
  >>> new_cube = cube_unitful * arr

  >>> # Inspect attributes of resultant cube.
  >>> new_cube.data
  array([[10., 22., 36.],
         [52., 70., 90.]])
  >>> new_cube.unit
  Unit("ct")
  >>> new_cube.uncertainty
  StdDevUncertainty([[1. , 2.2, 3.6],
                     [5.2, 7. , 9. ]])

  >>> # Divide by an astropy Quantity.
  >>> new_cube = cube_unitful / (2 * u.s)

  >>> # Inspect attributes of resultant cube.
  >>> new_cube.data
  array([[5. , 5.5, 6. ],
         [6.5, 7. , 7.5]])
  >>> new_cube.unit
  Unit("ct / s")
  >>> new_cube.uncertainty
  StdDevUncertainty([[0.5 , 0.55, 0.6 ],
                     [0.65, 0.7 , 0.75]])


.. _arithmetic_nddata:

Arithmetic Operations with Coordinate-less NDData
=================================================
