import astropy.units as u


def _expand_ellipsis(ndim, plist):
    if Ellipsis in plist:
        if plist.count(Ellipsis) > 1:
            raise IndexError("Only single ellipsis ('...') is permitted.")

        # Replace the Ellipsis with the correct number of slice(None)s
        e_ind = plist.index(Ellipsis)
        plist.remove(Ellipsis)
        n_e = ndim - len(plist)
        for i in range(n_e):
            ind = e_ind + i
            plist.insert(ind, None)

    return plist


def _expand_ellipsis_axis_coordinates(plist, wapt):
    if Ellipsis in plist:
        if plist.count(Ellipsis) > 1:
            raise IndexError("Only single ellipsis ('...') is permitted.")

        # Replace the Ellipsis with the correct number of slice(None)s
        e_ind = plist.index(Ellipsis)
        plist.remove(Ellipsis)
        n_e = len(wapt) - len(plist)
        for i in range(n_e):
            ind = e_ind + i
            plist.insert(ind, wapt[i])

    return plist


def prep_plot_kwargs(naxis, wcs, plot_axes, axes_coordinates, axes_units):
    """
    Prepare the kwargs for the plotting functions.

    This function accepts things in array order and returns things in WCS order.
    """
    # If plot_axes, axes_coordinates, axes_units are not None and not lists,
    # convert to lists for consistent indexing behaviour.
    if (not isinstance(plot_axes, (tuple, list))) and (plot_axes is not None):
        plot_axes = [plot_axes]
    if (not isinstance(axes_coordinates, (tuple, list))) and (axes_coordinates is not None):
        axes_coordinates = [axes_coordinates]
    if (not isinstance(axes_units, (tuple, list))) and (axes_units is not None):
        axes_units = [axes_units]
    # Set default value of plot_axes if not set by user.
    if plot_axes is None:
        plot_axes = [..., 'y', 'x']

    # We flip the plot axes here so they are in the right order for WCSAxes
    plot_axes = plot_axes[::-1]

    plot_axes = _expand_ellipsis(naxis, plot_axes)
    if 'x' not in plot_axes:
        raise ValueError("'x' must be in plot_axes.")

    if axes_coordinates is not None:
        axes_coordinates = _expand_ellipsis_axis_coordinates(axes_coordinates, wcs.world_axis_physical_types)
        # Ensure all elements in axes_coordinates are of correct types.
        ax_coord_types = (str, type(None))
        for axis_coordinate in axes_coordinates:
            if isinstance(axis_coordinate, str):
                # coordinates can be accessed by either name or type
                if axis_coordinate not in set(wcs.world_axis_physical_types).union(set(wcs.world_axis_names)):
                    raise ValueError(f"{axis_coordinate} is not one of this cubes world axis physical types.")
            if not isinstance(axis_coordinate, ax_coord_types):
                raise TypeError(f"axes_coordinates must be one of {ax_coord_types} or list of those, not {type(axis_coordinate)}.")

    if axes_units is not None:
        axes_units = _expand_ellipsis(wcs.world_n_dim, axes_units)
        if len(axes_units) != wcs.world_n_dim:
            raise ValueError(f"The length of the axes_units argument must be {wcs.world_n_dim}.")
        # Convert all non-None elements to astropy units
        axes_units = list(map(lambda x: u.Unit(x) if x is not None else None, axes_units))[::-1]
        for i, axis_unit in enumerate(axes_units):
            wau = wcs.world_axis_units[i]
            if axis_unit is not None and not axis_unit.is_equivalent(wau):
                raise u.UnitsError(
                    f"Specified axis unit '{axis_unit}' is not convertible to world axis unit '{wau}'")

    return plot_axes, axes_coordinates, axes_units


def set_wcsaxes_format_units(coord_map, wcs, axes_units=None):
    """
    Given an `~astropy.visualization.wcsaxes.coordinates_map.CoordinatesMap`
    object set the format units.
    """
    for i, coord in enumerate(coord_map):
        if axes_units is not None and axes_units[i] is not None:
            coord.set_format_unit(axes_units[i])
