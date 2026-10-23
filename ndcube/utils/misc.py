import copy
import numbers
from collections.abc import Sequence
from types import EllipsisType

try:
    from numpy.lib.array_utils import normalize_axis_index
except ImportError:  # numpy < 2
    from numpy.core.multiarray import normalize_axis_index

import astropy.units as u
from astropy.utils import minversion

__all__ = ["convert_quantities_to_units", "unique_sorted"]


def unique_sorted(iterable):
    """
    Return unique values in the order they are first encountered in the iterable.
    """
    lookup = set()  # a temporary lookup set
    return [ele for ele in iterable if ele not in lookup and lookup.add(ele) is None]


def convert_quantities_to_units(coords, units):
    """Converts a sequence of Quantities to units used in the WCS.

    Non-Quantity types in the sequence are allowed and ignored.

    Parameters
    ----------
    coords: iterable of `astropy.units.Quantity` or `None`
        The coordinates to be converted.

    units: iterable of `astropy.units.Unit` or `str`
        The units to which the coordinates should be converted.

    Returns
    -------
    converted_coords: iterable of `astropy.units.Quantity` or `None`
        The coordinates converted to the units.
        Non-quantity types remain.
    """
    return [coord.to(unit) if isinstance(coord, u.Quantity) else coord
            for coord, unit in zip(coords, units)]


def _deepcopy_without_parent(obj, memo):
    """
    Deep copy an object without copying the parent cube it links to through ``_ndcube``.

    Assign this as ``__deepcopy__`` on the class.
    """
    new = copy.copy(obj)
    memo[id(obj)] = new
    for key, value in vars(obj).items():
        if key != "_ndcube":
            setattr(new, key, copy.deepcopy(value, memo))
    # A plotter subclass that skips BasePlotter.__init__ has no _ndcube.
    if hasattr(obj, "_ndcube"):
        # Use the cube's copy if deepcopy reached the cube first, otherwise keep the original cube.
        new._ndcube = memo.get(id(obj._ndcube), obj._ndcube)
    return new


if minversion("astropy", "8.0.2"):
    from astropy.utils.shapes import simplify_basic_index as _simplify_basic_index
else:
    def _simplify_basic_index(
        basic_index: int | slice | Sequence[int | slice | EllipsisType | None],
        *,
        shape: Sequence[int | None],
    ) -> tuple[int | slice, ...]:
        """
        Given a Numpy basic index, return a tuple of integers and slice objects
        with no default values (`None`) if possible.

        If one of the dimensions has a slice and the step is negative and the stop
        value of the slice was originally `None`, the new stop value of the slice
        may still be set to `None`.

        For more information on valid basic indices, see
        https://numpy.org/doc/stable/user/basics.indexing.html#basic-indexing

        Parameters
        ----------
        basic_index
            A valid Numpy basic index
        shape
            The shape of the array being indexed. Entries can be `None` if the
            size along a dimension is not known, in which case the index for that
            dimension is returned unchanged, except that a negative integer for
            slice start raises an error since it cannot be interpreted.
        """
        ndim = len(shape)

        if not isinstance(basic_index, (tuple, list)):  # We just have a single int
            basic_index = (basic_index,)

        new_index = list(basic_index)

        # Identity rather than equality checks are used here since the latter
        # would fail if any of the elements were arrays.
        ellipsis = [i for i, idx in enumerate(new_index) if idx is Ellipsis]
        if len(ellipsis) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        if ellipsis:
            # Replace the Ellipsis with the correct number of slice(None)s
            e_ind = ellipsis[0]
            new_index[e_ind : e_ind + 1] = [slice(None)] * (ndim - len(new_index) + 1)

        if len(new_index) > ndim:
            raise ValueError(
                f"The dimensionality of the basic index {basic_index} can not be greater "
                f"than the dimensionality ({ndim}) of the data."
            )

        new_index += [slice(None)] * (ndim - len(new_index))

        for i, (slc, size) in enumerate(zip(new_index, shape)):
            if isinstance(slc, slice):
                if size is None:
                    if slc.start is not None and slc.start < 0:
                        raise ValueError(
                            f"Cannot use a negative slice start for dimension {i} "
                            "since its size is not known"
                        )
                else:
                    indices = list(slc.indices(size))
                    # The following case is the only one where slice(*indices) does
                    # not give the 'correct' answer because it will set stop to -1
                    # which means the last element in the array.
                    if indices[1] == -1:
                        indices[1] = None
                    new_index[i] = slice(*indices)
            elif isinstance(slc, numbers.Integral):
                if size is None:
                    if slc < 0:
                        raise ValueError(
                            f"Cannot use a negative index for dimension {i} since "
                            "its size is not known"
                        )
                    new_index[i] = int(slc)
                else:
                    new_index[i] = normalize_axis_index(int(slc), size)
            else:
                raise ValueError(f"Unexpected index element in basic index: {slc}")

        return tuple(new_index)
