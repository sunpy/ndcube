import numpy as np


def _update_aligned_axes(drop_aligned_axes_indices, aligned_axes):
    # Remove dropped axes from aligned_axes.  MUST BE A BETTER WAY TO DO THIS.
    if len(drop_aligned_axes_indices) > 0:
        new_aligned_axes = []
        for key in aligned_axes.keys():
            cube_aligned_axes = np.array(aligned_axes[key])
            for drop_axis_index in drop_aligned_axes_indices:
                drop_axis = cube_aligned_axes[drop_axis_index]
                cube_aligned_axes = np.delete(cube_aligned_axes, drop_axis_index)
                w = np.where(cube_aligned_axes > drop_axis)
                cube_aligned_axes[w] -= 1
                w = np.where(drop_aligned_axes_indices > drop_axis_index)
                drop_aligned_axes_indices[w] -= 1
            new_aligned_axes.append(tuple(cube_aligned_axes))
        new_aligned_axes = tuple(new_aligned_axes)
    else:
        new_aligned_axes = aligned_axes

    return new_aligned_axes

def slice_interval_is_1(item, axis_length):
    # Make start index numeric and positive.
    if item.start is None:
        start = 0
    else:
        start = item.start
    start = make_index_positive(start, axis_length)

    # Make stop index numeric and positive.
    if item.stop is None:
        stop = axis_length
    else:
        stop = item.stop
    stop = make_index_positive(stop, axis_length)

    # Make step numeric.
    if item.step is None:
        step = 1
    else:
        step = item.step

    # Check if slice interval is 1.
    if abs((stop - start) // step) == 1:
        result = True
    else:
        result = False

    return result

def make_index_positive(index, axis_length):
    if index < 0:
        pos_index = axis_length + index
    else:
        pos_index = index
    return pos_index
