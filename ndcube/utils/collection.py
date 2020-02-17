import numpy as np


def _generate_collection_getitems(item, collection_items, labels, aligned_axes, n_aligned_axes):
    # Determine whether any axes are dropped by slicing.
    # If so, remove them from aligned_axes.
    drop_aligned_axes_indices = []
    # If item is an int, first aligned axis is dropped.
    if isinstance(item, int):
        drop_aligned_axes_indices = [0]
        # Insert item to each cube's slice item.
        for i, key in enumerate(labels):
            collection_items[i][aligned_axes[key][0]] = item
    # If item is a slice such that only one element is in interval,
    # first aligned axis is dropped.
    elif isinstance(item, slice):
        if item.step is None:
            step = 1
        else:
            step = item.step
        if abs((item.stop - item.start) // step) < 2:
            drop_aligned_axes_indices = [0]
        # Insert item to each cube's slice item.
        for i, key in enumerate(labels):
            collection_items[i][aligned_axes[key][0]] = item
    # If item is tuple, search sub-items for ints or 1-interval slices.dd
    elif isinstance(item, tuple):
        # If item is tuple, search sub-items for ints or 1-interval slices.
        if len(item) > n_aligned_axes:
            raise IndexError("Too many indices")
        for i, axis_item in enumerate(item):
            if isinstance(axis_item, int):
                drop_aligned_axes_indices.append(i)
            elif isinstance(axis_item, slice):
                if axis_item.step is None:
                    step = 1
                else:
                    step = axis_item.step
                if abs((axis_item.stop - axis_item.start) // step) < 2:
                    drop_aligned_axes_indices.append(i)
            else:
                raise TypeError("Unsupported slicing type: {0}".format(axis_item))
            # Enter slice item into correct index for slice tuple of each cube.
            for j, key in enumerate(labels):
                collection_items[j][aligned_axes[key][i]] = axis_item
    else:
        raise TypeError("Unsupported slicing type: {0}".format(axis_item))

    return np.array(drop_aligned_axes_indices)

def _update_aligned_axes(drop_aligned_axes_indices, labels, aligned_axes):
    # Remove dropped axes from aligned_axes.  MUST BE A BETTER WAY TO DO THIS.
    if len(drop_aligned_axes_indices) > 0:
        new_aligned_axes = []
        for label in labels:
            cube_aligned_axes = np.array(aligned_axes[label])
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
