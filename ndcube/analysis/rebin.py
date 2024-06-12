import astropy.nddata
import numpy as np

from ndcube import utils
from ndcube.utils.exceptions import warn_user
from ndcube.wcs.wrappers import ResampledLowLevelWCS

__all__ = ["rebin"]

# Create mapping to masked array types based on data array type for use in analysis methods.
ARRAY_MASK_MAP = {}
ARRAY_MASK_MAP[np.ndarray] = np.ma.masked_array
_NUMPY_COPY_IF_NEEDED = False if np.__version__.startswith("1.") else None
try:
    import dask.array
    ARRAY_MASK_MAP[dask.array.core.Array] = dask.array.ma.masked_array
except ImportError:
    pass


def rebin(cube, bin_shape, operation=np.mean, operation_ignores_mask=False, handle_mask=np.all,
          propagate_uncertainties=False, new_unit=None, **kwargs):
    """
    Downsample array by combining contiguous pixels into bins.

    Values in bins are determined by applying a function to the pixel values within it.
    The number of pixels in each bin in each dimension is given by the bin_shape input.
    This must be an integer fraction of the cube's array size in each dimension.
    If the NDCube instance has uncertainties attached, they are propagated
    depending on binning method chosen.

    Parameters
    ----------
    cube : `ndcube.ndcube.NDCubeABC`
        The cube to rebin.
    bin_shape : array-like
        The number of pixels in a bin in each dimension.
        Must be the same length as number of dimensions in data.
        Each element must be in int. If they are not they will be rounded
        to the nearest int.
    operation : function
        Function applied to the data to derive values of the bins.
        Default is `numpy.mean`
    operation_ignores_mask: `bool`
        Determines how masked values are handled.
        If False (default), masked values are excluded when calculating rebinned value.
        If True, masked values are used in calculating rebinned value.
    handle_mask: `None` or function
        Function to apply to each bin in the mask to calculate the new mask values.
        If `None` resultant mask is `None`.
        Default is `numpy.all`
    propagate_uncertainties: `bool` or function.
        If False, uncertainties are dropped.
        If True, default algorithm is used (`~ndcube.utils.cube.propagate_rebin_uncertainties`)
        Can also be set to a function which performs custom uncertainty propagation.
        Additional kwargs provided to this method are passed onto this function.
        See Notes section on how to write a custom ``propagate_uncertainties`` function.
    new_unit: `astropy.units.Unit`, optional
        If the rebinning operation alters the data unit, the new unit can be
        provided here.
    kwargs
        All kwargs are passed to the error propagation function.

    Returns
    -------
    new_cube: `~ndcube.NDCube`
        The resolution-degraded cube.

    Notes
    -----
    **Rebining Algorithm**
    Rebinning is achieved by reshaping the N-D array to a 2N-D array and
    applying the function over the odd-numbered axes. To demonstrate,
    consider the following example. Let's say you have an array::

         x = np.array([[0, 0, 0, 1, 1, 1],
                       [0, 0, 1, 1, 0, 0],
                       [1, 1, 0, 0, 1, 1],
                       [0, 0, 0, 0, 1, 1],
                       [1, 0, 1, 0, 1, 1],
                       [0, 0, 1, 0, 0, 0]])

    and you want to sum over 2x2 non-overlapping sub-arrays. This summing can
    be done by reshaping the array::

         y = x.reshape(3,2,3,2)

    and then summing over the 1st and third directions::

         y2 = y.sum(axis=3).sum(axis=1)

    which gives the expected array::

         array([[0, 3, 2],
                [2, 0, 4],
                [1, 2, 2]])

    **Defining Custom Error Propagation**
    To perform custom uncertainty propagation, a function must be provided via the
    propgate_uncertainty kwarg. This function must accept, although doesn't have to
    use, the following args:

    uncertainty: `astropy.nddata.NDUncertainty` but not `astropy.nddata.UnknownUncertainty`
        The uncertainties associated with the data.
    data: array-like
        The data associated with the above uncertainties.
        Must have same shape as uncertainty.
    mask: array-like of `bool` or `None`
        Indicates whether any uncertainty elements should be ignored in propagation.
        True elements cause corresponding uncertainty elements to be ignored.
        False elements cause corresponding uncertainty elements to be propagated.
        Must have same shape as above.
        If None, no uncertainties are ignored.

    All kwarg inputs to the rebin method are also passed on transparently to the
    propagation function. Hence additional inputs to the propagation function can be
    included as kwargs to :meth:`ndcube.NDCube.rebin`.

    The shape of the uncertainty, data and mask inputs are such that the first
    dimension represents the pixels in a given bin whose data and uncertainties
    are aggregated by the rebin process. The shape of the remaining dimensions
    must be the same as the final rebinned data. A silly but informative
    example of a custom propagation function might be::

         def my_propagate(uncertainty, data, mask, **kwargs):
             # As a silly example, propagate uncertainties by summing those in same bin.
             # Note not all args are used, but function must accept them.
             n_pixels_per_bin = data.shape[0]  # 1st dimension of inputs gives pixels in bin.
             final_shape = data.shape[1:]  # Trailing dims give shape of put rebinned data.
             # Propagate uncerts by adding them.
             new_uncert = numpy.zeros(final_shape)
             for i in range(n_pixels_per_bin):
                 new_uncert += uncertainty.array[i]
             # Alternatively: new_uncerts = uncertainty.array.sum(axis=0)
             return type(uncertainty)(new_uncert)  # Convert to original uncert type and return.
    """
    # Sanitize input.
    new_unit = new_unit or cube.unit
    # Make sure the input bin dimensions are integers.
    bin_shape = np.rint(bin_shape).astype(int)
    if all(bin_shape == 1):
        return cube
    # Ensure bin_size has right number of entries and each entry is an
    # integer fraction of the array shape in each dimension.
    data_shape = cube.shape
    naxes = len(data_shape)
    if len(bin_shape) != naxes:
        raise ValueError("bin_shape must have an entry for each array axis.")
    if (np.mod(data_shape, bin_shape) != 0).any():
        raise ValueError(
            "bin shape must be an integer fraction of the data shape in each dimension. "
            f"data shape: {data_shape};  bin shape: {bin_shape}")

    # Reshape array so odd dimensions represent pixels to be binned
    # then apply function over those axes.
    data, sanitized_mask = _create_masked_array_for_rebinning(cube.data, cube.mask,
                                                              operation_ignores_mask)
    reshape = np.empty(len(data_shape) + len(bin_shape), dtype=int)
    new_shape = (data_shape / bin_shape).astype(int)
    reshape[0::2] = new_shape
    reshape[1::2] = bin_shape
    reshape = tuple(reshape)
    reshaped_data = data.reshape(reshape)
    operation_axes = tuple(range(len(reshape) - 1, 0, -2))
    new_data = operation(reshaped_data, axis=operation_axes)
    if isinstance(new_data, ARRAY_MASK_MAP[np.ndarray]):
        new_data = new_data.data
    if handle_mask is None:
        new_mask = None
    elif isinstance(cube.mask, (type(None), bool)):  # Preserve original mask type.
        new_mask = cube.mask
    else:
        reshaped_mask = cube.mask.reshape(reshape)
        new_mask = handle_mask(reshaped_mask, axis=operation_axes)

    # Propagate uncertainties if propagate_uncertainties kwarg set.
    new_uncertainty = None
    if propagate_uncertainties and _uncertainties_can_propagate(cube.uncertainty, cube.mask, operation_ignores_mask):
        if propagate_uncertainties is True:
            propagate_uncertainties = utils.cube.propagate_rebin_uncertainties
        # If propagate_uncertainties, use astropy's infrastructure.
        # For this the data and uncertainty must be reshaped
        # so the first dimension represents the flattened size of a single bin
        # while the rest represent the shape of the new data. Then the elements
        # in each bin can be iterated (all bins being treated in parallel) and
        # their uncertainties propagated.
        bin_size = bin_shape.prod()
        flat_shape = [bin_size] + list(new_shape)
        dummy_axes = tuple(range(1, len(reshape), 2))
        flat_data = np.moveaxis(reshaped_data, dummy_axes, tuple(range(naxes)))
        flat_data = flat_data.reshape(flat_shape)
        reshaped_uncertainty = cube.uncertainty.array.reshape(tuple(reshape))
        flat_uncertainty = np.moveaxis(reshaped_uncertainty, dummy_axes, tuple(range(naxes)))
        flat_uncertainty = flat_uncertainty.reshape(flat_shape)
        flat_uncertainty = type(cube.uncertainty)(flat_uncertainty)
        if sanitized_mask is not None:
            reshaped_mask = cube.mask.reshape(tuple(reshape))
            flat_mask = np.moveaxis(reshaped_mask, dummy_axes, tuple(range(naxes)))
            flat_mask = flat_mask.reshape(flat_shape)
        else:
            flat_mask = None
        # Propagate uncertainties.
        new_uncertainty = propagate_uncertainties(
            flat_uncertainty, flat_data, flat_mask,
            operation=operation, operation_ignores_mask=operation_ignores_mask,
            handle_mask=handle_mask, new_unit=new_unit, **kwargs)

    # Resample WCS
    new_wcs = ResampledLowLevelWCS(cube.wcs.low_level_wcs, bin_shape[::-1])

    # Reform NDCube.
    new_cube = type(cube)(
        data=new_data,
        wcs=new_wcs,
        uncertainty=new_uncertainty,
        mask=new_mask,
        meta=cube.meta,
        unit=new_unit
    )
    new_cube._global_coords = cube._global_coords
    # Reconstitute extra coords
    if not cube.extra_coords.is_empty:
        new_cube._extra_coords = cube.extra_coords.resample(bin_shape, ndcube=new_cube)

    return new_cube


def rebin_by_edges(
    cube,
    axes_idx_edges,
    operation=np.mean,
    operation_ignores_mask=False,
    handle_mask=np.all,
    propagate_uncertainties=False,
    new_unit=None,
    **kwargs,
):
    """
    Downsample array by combining irregularly sized, but contiguous, blocks of elements into bins.

    Parameters
    ----------
    data:

    axes_idx_edges: iterable of iterable of `int`
        The array indices for each axis defining the edges of the contiguous blocks
        to rebin the data to. Each block includes the lower index and up to but not
        including the upper index.

    operation: func
        The operation defining how the elements in each block should be combined.
        Must be able to be applied over a specific axis via an axis= kwarg.
        Default=`numpy.mean`.

    Returns
    -------
    :

    Example
    -------
    >>> import numpy as np
    >>> from stixpy.product.tools import rebin_by_edges
    >>> axes_idx_edges = [0, 2, 3], [0, 2, 4], [0, 3, 5]
    >>> data = np.ones((3, 4, 5))
    >>> rebin_by_edges(data, axes_idx_edges, operation=np.sum) # doctest: +SKIP
    array([[[12.,  8.],
        [12.,  8.]],
       [[ 6.,  4.],
        [ 6.,  4.]]])
    """
    # Sanitize inputs
    ndim = len(cube.data.shape)
    if len(axes_idx_edges) != ndim:
        raise ValueError(f"length of axes_idx_edges must be {ndim}")

    # Extract coordinate grids for rebinning and other coord info.
    orig_wcs = cube.combined_wcs
    new_coord_values = list(cube.axis_world_coords_values(wcs=orig_wcs))
    new_coord_physical_types = orig_wcs.world_axis_physical_types[::-1]
    new_coord_pix_idxs = [physical_type_to_pixel_axes(phys_type, orig_wcs) for phys_type in new_coord_physical_types]
    new_coord_names = (
        name if name else phys_type
        for name, phys_type in zip(orig_wcs.world_axis_names[::-1], new_coord_physical_types)
    )
    wc_arr_idxs = [convert_between_array_and_pixel_axes(idx, ndim) for idx in new_coord_pix_idxs]

    # Combine data and mask and determine whether mask also needs rebinning.
    new_data = _create_masked_array_for_rebinning(cube.data, cube.mask, operation_ignores_mask)
    rebin_mask = False
    if handle_mask is None or operation_ignores_mask:
        new_mask = None
    else:
        new_mask = cube.mask
        if not isinstance(cube.mask, (type(None), bool)):
            rebin_mask = True

    # Determine whether to propagate uncertainties
    propagate_uncertainties = propagate_uncertainties and _uncertainties_can_propagate(cube.uncertainty, cube.mask, operation_ignores_mask)
    new_uncertainty = cube.uncertainty if propagate_uncertainties else None

    # Iterate through dimensions and perform rebinning operation.
    for i, edges in enumerate(axes_idx_edges):
        # If no edge indices provided for dimension, skip to next one.
        if edges is None:
            continue
        # Iterate through pixel blocks and collapse via rebinning operation.
        tmp_data = []
        tmp_mask = []
        tmp_uncertainties = []
        tmp_coords = [[]] * len(new_coord_values)
        item = [slice(None)] * ndim
        coord_item = np.array([slice(None)] * ndim)
        for j in range(len(edges) - 1):
            # Rebin data
            item[i] = slice(edges[j], edges[j + 1])
            tuple_item = tuple(item)
            data_chunk = new_data[tuple_item]
            tmp_data.append(operation(data_chunk, axis=i))
            # Rebin mask
            if rebin_mask is True:
                mask_chunk = new_mask[tuple_item]
                tmp_mask.append(handle_mask(mask_chunk, axis=i))
            # Rebin uncertainties
            if propagate_uncertainties:
                # Reorder axis i to be 0th axis as this is format required by propagate_rebin_uncertainties
                uncertainty_chunk = type(new_uncertainty)(
                    np.moveaxis(new_uncertainty.array[tuple_item], i, 0), unit=new_uncertainty.unit
                )
                data_unc = np.moveaxis(data_chunk, i, 0)
                mask_unc = np.moveaxis(mask_chunk, i, 0) if rebin_mask else new_mask
                tmp_uncertainty = propagate_rebin_uncertainties(
                    uncertainty_chunk,
                    data_unc,
                    mask_unc,
                    operation,
                    operation_ignores_mask=operation_ignores_mask,
                    handle_mask=handle_mask,
                    new_unit=new_unit,
                    **kwargs,
                )
                tmp_uncertainties.append(tmp_uncertainty.array)
            # Rebin coordinates
            coord_item[i] = np.array([edges[j], edges[j + 1] - 1])
            for k, (coord, coord_arr_idx) in enumerate(zip(new_coord_values, wc_arr_idxs)):
                if i in coord_arr_idx:
                    idx_i = np.where(coord_arr_idx == i)[0][0]
                    tmp_coords[k].append(np.mean(coord[tuple(coord_item[coord_arr_idx])], axis=idx_i))
        # Restore rebinned axes.
        if rebin_mask:
            new_mask = np.stack(tmp_mask, axis=i)
        new_data = np.ma.masked_array(np.stack(tmp_data, axis=i).data, mask=new_mask)
        if propagate_uncertainties:
            new_uncertainty = type(new_uncertainty)(np.stack(tmp_uncertainties, axis=i), unit=new_uncertainty.unit)
        for k, coord_arr_idx in enumerate(wc_arr_idxs):
            if i in coord_arr_idx:
                idx_i = np.where(coord_arr_idx == i)[0][0]
                new_coord_values[k] = np.stack(tmp_coords[k], axis=idx_i)

    # Build new WCS.
    ec = ExtraCoords()
    for name, axis, coord, physical_type in zip(
        new_coord_names, new_coord_pix_idxs, new_coord_values, new_coord_physical_types
    ):
        if len(axis) != 1:
            raise NotImplementedError("Rebinning multi-dimensional coordinates not supported.")
        ec.add(name, axis, coord, physical_types=physical_type)
    new_wcs = ec.wcs

    # Rebin metadata if possible.
    if hasattr(cube.meta, "__ndcube_can_rebin__") and cube.meta.__ndcube_can_rebin__:
        new_shape = [len(ax) - 1 for ax in axes_idx_edges]
        rebinned_axes = set(tuple(ax) != tuple(range(n + 1)) for ax, n in zip(axes_idx_edges, cube.shape))
        new_meta = cube.meta.rebin(rebinned_axes, new_shape)
    else:
        new_meta = cube.meta

    # Build new cube.
    new_cube = type(cube)(new_data.data, new_wcs, uncertainty=new_uncertainty, mask=new_data.mask, meta=new_meta)

    return new_cube


def _create_masked_array_for_rebinning(data, mask, operation_ignores_mask):
    m = None if (mask is None or mask is False or operation_ignores_mask) else mask
    if m is None:
        return data, m
    else:
        for array_type, masked_type in ARRAY_MASK_MAP.items():
            if isinstance(data, array_type):
                break
        else:
            masked_type = np.ma.masked_array
            warn_user("data and mask arrays of different or unrecognized types. Casting them into a numpy masked array.")
        return masked_type(data, m), m


def _uncertainties_can_propagate(uncertainty, mask, operation_ignores_mask):
    if uncertainty is None:
        warn_user("Uncertainties cannot be propagated as there are no uncertainties, "
                  "i.e., the `uncertainty` keyword was never set on creation of this NDCube.")
        return False
    elif isinstance(uncertainty, astropy.nddata.UnknownUncertainty):
        warn_user("The uncertainty on this NDCube has no known way to propagate forward and so will be dropped. "
                  "To create an uncertainty that can propagate, please see "
                  "https://docs.astropy.org/en/stable/uncertainty/index.html")
        return False
    elif (not operation_ignores_mask
          and (mask is True or (mask is not None and not isinstance(mask, bool) and mask.all()))):
        warn_user("Uncertainties cannot be propagated as all values are masked and "
                  "operation_ignores_mask is False.")
        return False
    return True
