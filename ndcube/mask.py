import copy
import textwrap

all = ["Mask"]


class Mask:
    """A class for holding and combining boolean mask arrays.

    Each mask can be activated or deactivated and thereby included or excluded
    from the combined mask. See the activate/deactive methods.

    Parameters
    ----------
    masks: dict-like or parseable by `dict`
        The names and arrays of each mask.
        Each mask array must be the same shape and of boolean type.

    meta: dict-like
        Any metadata associated with the masks.
    """
    def __init__(self, masks, meta=None):
        self._masks = dict(masks)
        shape = self.shape
        for mask in self._masks.values():
            if mask.shape != shape:
                print(mask.shape, shape)
                raise ValueError("All masks must have same shape.")
                self._masks[key] = mask.astype(bool)
        self.meta = meta
        self._active = dict((key, True) for key in self)

    @property
    def active_masks(self):
        """Return the names of active masks"""
        return tuple(key for key, active in self._active.items() if active)

    @property
    def mask(self):
        """Return the combined mask.

        The active masks are compared element-wise. Any element which has a value
        of True is any active mask is given a value of True in the combined mask.
        """
        return sum(self[key] for key in self if self.is_active(key)) > 0

    @property
    def names(self):
        """Return the names of all masks, whether active or not."""
        return tuple(self._masks.keys())

    @property
    def shape(self):
        """Return the shape of the masks.

        Note all masks by definition have the same shape.
        """
        return self._masks[list(self._masks.keys())[0]].shape

    def add(self, name, mask, activate=True, overwrite=False):
        """Add a new mask.

        Parameters
        ----------
        name: `str`
            Name of the new mask.

        mask: array-like of boolean type.
            The mask values.  Must be same shape are masks already present.

        activate: `bool`
            Whether the mask should be activate or not. Default=True

        overwrite: `bool`
            If False, an error will be raise if there is already a mask with the same name.
            Otherwise the mask info will be overwritten.
        """
        if name in self and overwrite is not True:
            raise ValueError(
                "A mask with this name already exists. Set overwrite=True to overwrite the mask.")
        if mask.shape != self.shape:
            raise ValueError(
                f"New mask must have same shape as masks already present: {self.shape}.")
        self._masks[name] = mask
        self._active[name] = bool(activate)

    def remove(self, name):
        """Remove a mask.

        Parameters
        ----------
        name: `str`
            The name of the mask to remove.
        """
        del self._masks[name]
        del self._active[name]

    def activate(self, names=None, all_masks=False):
        """Activate a mask.

        Only active masks are included in the calculation of the combined mask.

        Parameters
        ----------
        names: `str` or iterable of `str`  (optional)
            The name or names of the masks to activate.
            must be set if all_masks=False.

        all_masks: `bool`
            If True, the names input will be ignored and all masks will be activated.
        """
        self._set_active_status(True, names, all_masks)

    def deactivate(self, names=None, all_masks=False):
        """Deactivate a mask.

        Only active masks are included in the calculation of the combined mask.

        Parameters
        ----------
        names: `str` or iterable of `str`  (optional)
            The name or names of the masks to deactivate.
            must be set if all_masks=False.

        all_masks: `bool`
            If True, the names input will be ignored and all masks will be deactivated.
        """
        self._set_active_status(False, names, all_masks)

    def _set_active_status(self, activate, names, all_masks):
        if not (names or all_masks):
            raise ValueError("If mask names not provided, all_masks must be True.")
        if all_masks:
            names = self._active.keys()
        elif isinstance(names, str):
            names = (names,)
        activate = bool(activate)
        for key in names:
            self._active[key] = activate

    def is_active(self, key):
        """Return True is mask is active, False otherwise."""
        return self._active[key]

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._masks[item]
        sliced_mask = type(self)([(key, mask[item]) for key, mask in self._masks.items()])
        sliced_mask.meta = copy.deepcopy(self.meta)
        sliced_mask._active = dict(self._active)
        return sliced_mask

    def __str__(self):
        return textwrap.dedent(f"""\
                Mask
                ----
                Component Masks:\t{self.mask_names}
                Active Masks:\t{self.active_masks}
                Mask:
                {self.mask}""")

    def __repr__(self):
        return f"{object.__repr__(self)}\n{str(self)}"

    def __iter__(self):
        return iter(self._masks)

    def __contains__(self, name):
        return name in self._masks

    def __len__(self):
        return len(self._masks)
