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


def prep_plot_kwargs(cube, plot_axes, axes_coordinates, axes_units):
    naxis = len(cube.dimensions)

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
        plot_axes = [..., 'x', 'y']

    plot_axes = _expand_ellipsis(naxis, plot_axes)
    if 'x' not in plot_axes:
        raise ValueError("'x' must be in plot_axes.")

    if axes_coordinates is not None:
        axes_coordinates = _expand_ellipsis(naxis, axes_coordinates)
        # Now axes_coordinates have been converted to a consistent convention,
        # ensure their length equals the number of sequence dimensions.
        if len(axes_coordinates) != naxis:
            raise ValueError(f"length of axes_coordinates must be {naxis}.")
        # Ensure all elements in axes_coordinates are of correct types.
        ax_coord_types = (str,)
        for axis_coordinate in axes_coordinates:
            if isinstance(axis_coordinate, str):
                if axis_coordinate not in cube.world_axis_physical_types:
                    raise ValueError(f"{axis_coordinate} is not one of this cubes world axis physical types.")
            if axis_coordinate is not None and not isinstance(axis_coordinate, ax_coord_types):
                raise TypeError(f"axes_coordinates must be one of {ax_coord_types} or list of those.")

    if axes_units is not None:
        axes_units = _expand_ellipsis(naxis, axes_units)
        if len(axes_units) != naxis:
            raise ValueError(f"length of axes_units must be {naxis}.")
        # Convert all non-None elements to astropy units
        axes_units = list(map(lambda x: u.Unit(x) if x is not None else None, axes_units))
        for i, axis_unit in enumerate(axes_units):
            wau = cube.wcs.world_axis_units[i]
            if not axis_unit.is_equivalent(wau):
                raise u.UnitsError(
                    f"Specified axis unit '{axis_unit}' is not convertible to world axis unit '{wau}'")

    return plot_axes, axes_coordinates, axes_units


def set_wcsaxes_labels_units(coord_map, wcs, axes_units=None):
    """
    Given an `~astropy.visualization.wcsaxes.coordinates_map.CoordinatesMap`
    object set the format units and labels.
    """
    for i, coord in enumerate(coord_map):
        if axes_units is not None and axes_units[0] is not None:
            coord.set_format_unit(axes_units[0])

        # Use wcs here for ordering to match wcsaxes
        physical_type = wcs.world_axis_physical_types[coord.coord_index]
        format_unit = coord.get_format_unit()

        if coord.coord_type in ('longitude', 'latitude'):
            # Don't set unit for lon/lat axes as the ticks are formatted
            # with the unit
            coord.set_axislabel(f"{physical_type}")
        else:
            coord.set_axislabel(f"{physical_type} [{format_unit:latex}]")
