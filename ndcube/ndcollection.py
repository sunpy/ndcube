
import numpy as np

from ndcube import NDCube
from ndcube.utils.cube import convert_extra_coords_dict_to_input_format

class NDCollection():
    def __init__(self, data, labels, wcs):
        """
        A class for holding and manipulating a collection of aligned NDCube or NDCubeSequences.

        Data cubes/sequences must be aligned, i.e. have the same WCS and be the same shape.

        Parameters
        ----------
        data: sequence of `~ndcube.NDCube` or `~ndcube.NDCubeSequence`
            The data cubes/sequences to held in the collection.

        labels: sequence of `str`
            Name of each cube/sequence. Each label must be unique and
            there must be one per element in the data input.

        wcs: `~astropy.wcs.WCS`
            Single WCS that describes all cubes/sequences.

        """
        # Check inputs
        # Ensure there are no duplicate labels
        if len(set(labels)) != len(labels):
            raise ValueError("Duplicate labels detected.")
        if len(labels) != len(data):
            raise ValueError("Data and labels inputs of different lengths.")

        # Overwrite cubes' wcs with one supplied by user
        # to ensure all cubes have same WCS.
        # This should be relaxed in future versions.
        new_data = [NDCube(cube.data, wcs, uncertainty=cube.uncertainty, mask=cube.mask,
                           unit=cube.unit, meta=cube.meta,
                           extra_coords=convert_extra_coords_dict_to_input_format(
                               cube.extra_coords, cube.missing_axes))
                    for cube in data]
        data = new_data

        self.data = dict([(labels[i], data[i]) for i in range(len(data))])
        self.labels = labels

    def __getitem__(self, item):
        try:
            item_strings = [isinstance(_item, str) for _item in item]
            item_is_strings = all(item_strings)
            # Ensure strings are not mixed with slices.
            if (not item_is_strings) and (not all(np.invert(item_strings))):
                raise TypeError("Cannot mix labels and non-labels when indexing instance.")
        except:
            item_is_strings = False
        if isinstance(item, str) or item_is_strings:
            if isinstance(item, str):
                item = [item]
            new_data = [self.data[_item] for _item in item]
            new_labels = item
        else:
            new_labels = self.labels
            new_data = [self.data[key][item] for key in new_labels]
        return self.__class__(new_data, labels=new_labels, wcs=self.data[new_labels[0]].wcs)

    def __repr__(self):
        cube_types = type(self.data[self.labels[0]])
        n_cubes = len(self.labels)
        return ("""NDCollection
----------
Cube labels: {labels}
Number of Cubes: {n_cubes}
Cube Types: {cube_types}
Aligned dimensions: {aligned_dims}""".format(labels=self.labels, n_cubes=n_cubes,
                                             cube_types=cube_types,
                                             aligned_dims=self.dimensions))

    @property
    def dimensions(self):
        return self.data[self.labels[0]].dimensions

    @property
    def world_axis_physical_types(self):
        return self.data[self.labels[0]].world_axis_physical_types


