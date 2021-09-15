import numpy as np
import pytest

from ndcube.utils import cube as cube_utils

cases = (
    ([[1, 0, 1],
      [0, 1, 1],
      [0, 0, 1],
      [0, 0, 1]],
    (('low', 'low', 'low', 'low'),
     ('low', 'low', 'high', 'high'),
     ('high', 'low', 'low', 'low'),
     ('high', 'low', 'high', 'high'),
     ('low', 'high', 'low', 'low'),
     ('low', 'high', 'high', 'high'),
     ('high', 'high', 'low', 'low'),
     ('high', 'high', 'high', 'high'))),
    ([[1,],
      [1,]],
     (('low', 'low'),
      ('high', 'high'))),
    ([[1, 1,],
      [1, 1,]],
     (('low', 'low'),
      ('low', 'high'),
      ('high', 'low'),
      ('high', 'high'))),
    ([[1, 1,],
      [1, 1,],
      [1, 1,],
      [1, 1,]],
     (('low', 'low', 'low', 'low'),
      ('high', 'high', 'high', 'high'),
      ('high', 'high', 'high', 'low'),
      ('low', 'low', 'low', 'high'))),
    ([[1, 1, 0, 0],
      [1, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]],
     (('low', 'low', 'low', 'low'),
      ('low', 'high', 'low', 'low'),
      ('high', 'low', 'high', 'high'),
      ('high', 'high', 'high', 'high'),
      )),
    ([[1, 0],
      [0, 1],
      [1, 1]],
     (
     ('low', 'low', 'low'),
     ('high', 'high', 'high'),
     )),
    ([[1, 1, 0, 0],
      [1, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
      [0, 0, 1, 1]],
     (
     ('low', 'low', 'low', 'low', 'low'),
     ('low', 'high', 'low', 'low', 'low'),
     ('high', 'low', 'high', 'high', 'high'),
     ('high', 'high', 'high', 'high', 'high'),
     )),
)


@pytest.mark.parametrize("acm, expected_corners", cases)
def test_bounding_box_to_corners(acm, expected_corners):
    # Define inputs.
    # Note that there are two world axes that correspond to the same single pixel axis.
    # These are dependent and therefore their values must always point to the same
    # pixel index as each other.
    # The axis correlation matrix is (world_index, pixel_index) so this matrix
    # has 4 world and 3 pixel dims
    axis_correlation_matrix = np.array(acm, dtype=bool)
    # Construct corner values.
    # For clarity use strings instead of numbers where 'low' represents
    # the world value of the lower corner for that world axis,
    # while 'high' represents the world value of the upper corner for that world axis.
    lower_corner_values = ["low"] * axis_correlation_matrix.shape[0]
    upper_corner_values = ["high"] * axis_correlation_matrix.shape[0]
    # Build expected result.
    # Note that the 3rd & 4th column entries (corresponding to the
    # 3rd & 4th world axes) always have the same string as each other.
    # Also note the algorithm does not duplicate valid corner combinations.
    output_corners = cube_utils.bounding_box_to_corners(lower_corner_values, upper_corner_values,
                                                        axis_correlation_matrix)
    assert set(output_corners) == set(expected_corners), output_corners
