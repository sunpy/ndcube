from ndcube import NDCubeABC
from ndcube.utils.wcs import physical_type_to_array_axes

class NDCoords:
    def __init__(self, ndcube):
        if not isinstance(ndcube, NDCubeABC):
            raise TypeError("ndcube must be an instance of NDCubeABC.")
        self._ndcube = ndcube
        # Create aliases to coordinate methods already on NDCube.
        self.axis_world_coords = self._ndcube.axis_world_coords
        self.axis_world_coords_values = self._ndcube.axis_world_coords_values
        # Expose properties to get values of coordinate types based on
        # physical types present in NDCube.wcs and NDCube.extra_coords.
        self._celestial_words = ["pos"]
        self._time_words = ["time"]
        self._spectral_words = ["em"]
        self._stokes_words = ["stokes"]
        if _words_in_physical_types(self._celestial_words)
            self.celestial = self._celestial
            self.celestial_array_axes = self._celestial_array_axes
        if _words_in_physical_types(self._time_words)
            self.time = self._time
            self.time_array_axes = self._time_array_axes
        if _words_in_physical_types(self._spectral_words)
            self.spectral = self._spectral
            self.spectral_array_axes = self._spectral_array_axes
        if _words_in_physical_types(self._stokes_words)
            self.stokes = self._stokes
            self.stokes_array_axes = self._stokes_array_axes

    @property
    def wcs(self):
        return self._ndcube.wcs

    @property
    def extra_coords(self):
        return self._ndcube.extra_coords

    @property
    def combined_wcs(self):
        return self._ndcube.combined_wcs

    @property
    def global_coords(self):
        return self._ndcube.global_coords

    @property
    def array_axis_physical_types(self):
        return self._ndcube.array_axis_physical_types

    @property
    def _celestial(self):
        """
        Returns celestial coordinates of associated NDCube.
        
        Returns
        -------
        : `astropy.coordinates.SkyCoord`
        """
        return self._get_coords_by_word(self._celestial_words)

    @property
    def _celestial_array_axes(self):
        """Returns array axis indices associated with celestial coordinates."""
        return _get_array_axes_from_coord_words(self._celestial_words)
        

    @property
    def _time(self):
        """
        Returns time coordinates for associated NDCube.

        Returns
        -------
        : `astropy.time.Time`
        """
        return self._get_coords_by_word(self._time_words)

    @property
    def _time_array_axes(self):
        """Returns array axis indices associated with time coordinates."""
        return _get_array_axes_from_coord_words(self._time_words)

    @property
    def _spectral(self):
        """
        Returns spectral coordinates of associated NDCube.

        Returns
        -------
        : `astropy.coordinates.SpectralCoord` or `astropy.units.Quantity`
        """
        return self._get_coords_by_word(self._spectral_words)

    @property
    def _spectral_array_axes(self):
        """Returns array axis indices associated with spectral coordinates."""
        return _get_array_axes_from_coord_words(self._spectral_words)

    @property
    def _stokes(self):
        """
        Returns stokes polarization coordinates of associated NDCube.
        """
        return self._get_coords_by_word(self._stokes_words)

    @property
    def _stokes_array_axes(self):
        """Returns array axis indices associated with stokes coordinates."""
        return _get_array_axes_from_coord_words(self._stokes_words)

    def _get_coords_by_word(self, coord_words):
        """
        Returns coordinates from a WCS corresponding to a world axis physical type.
        """
        if isinstance(coord_words, str):
            coord_words = [coord_words]
        # Check if words are in physical types of main WCS and extra coords.
        wcs_phys_types, ec_phys_types = _get_physical_types_from_words(coord_words, [self.wcs, self.extra_coords.wcs])
        # Determine type of WCS to use based on whether the words were
        # found in WCS and/or extra coords.
        if len(wcs_phys_types) > 0:
           if len(ec_phys_types) > 0:
               wcs = self.combined_wcs
               physical_types = wcs_phys_types + ec_phys_types
           else:
               wcs = self.wcs
               physical_types = wcs_phys_types
        elif len(ec_phys_types) > 0:
            wcs = self.extra_coords
            physical_types = ec_phys_types
        else:
            return None
        # Calculate coordinates.
        coords = self.axis_world_coords(*physical_types, wcs=wcs)
        if len(coords) == 1:
            coords = coords[0]
        return coords


def _get_physical_types_from_words(coord_words, wcses):
    found_phys_types = []
    for i, wcs in enumerate(wcses):
        found_phys_types.append([])
        if wcs is not None:
            for phys_type in wcs.world_axis_physical_types:
                words = set(phys_type.replace(":", ".").split("."))
                if any(word in words for word in coord_words):
                    found_phys_types[i].append(name)
    return found_phys_types


def _words_in_physical_types(coord_words, wcses):
    found_phys_types = _get_physical_types_from_words(coord_words, wcses)
    if any(len(fpt) > 0 for fpt in found_phys_types):
        return True
    else:
        return False


def _get_array_axes_from_coord_words(coord_words, wcs):
    physical_types = _get_physical_types_from_words(coord_words, wcs)[0]
    dims = set()
    for phys_type in physical_types:
        dims = dims.union(set(physical_type_to_array_axes(phys_type, wcs))
    return tuple(dims)

