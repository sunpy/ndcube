from ndcube import NDCubeABC

class NDCoords:
    def __init__(self, ndcube):
        if not isinstance(ndcube, NDCubeABC):
            raise TypeError("ndcube must be an instance of NDCubeABC.")
        self._ndcube = ndcube
        # Create aliases to coordinate methods already on NDCube.
        self.axis_world_coords = self._ndcube.axis_world_coords
        self.axis_world_coords_values = self._ndcube.axis_world_coords_values
        # Exposure properties to get values of coordinate types based on
        # physical types present in NDCube.wcs and NDCube.extra_coords.
        self._celestial_words = ["pos"]
        self._time_words = ["time"]
        self._spectral_words = ["em"]
        self._stokes_words = ["stokes"]
        if _words_in_physical_types(self._celestial_words)
            self.celestial = self._celestial
        if _words_in_physical_types(self._time_words)
            self.time = self._time
        if _words_in_physical_types(self._spectral_words)
            self.spectral = self._spectral
        if _words_in_physical_types(self._stokes_words)
            self.stokes = self._stokes

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
        Returns celestial coordinates for all array indices in relevant axes.
        Celestial world axis physical type name must contain 'pos.'.
        Parameters
        ----------
        wcs : `astropy.wcs.wcsapi.BaseHighLevelWCS`
           The WCS object with which to calculate the coordinates. Must be
           self.wcs, self.extra_coords, or self.combined_wcs
        Returns
        -------
        : `astropy.coordinates.SkyCoord`
        """
        return self._get_coords_by_word(self._celestial_words)

    @property
    def _time(self):
        """
        Returns time coordinates for all array indices in relevant axes.
        Time world axis physical type name must contain 'time'.
        Parameters
        ----------
        wcs : `astropy.wcs.wcsapi.BaseHighLevelWCS`
           The WCS object with which to calculate the coordinates. Must be
           self.wcs, self.extra_coords, or self.combined_wcs
        Returns
        -------
        : `astropy.time.Time`
        """
        return self._get_coords_by_word(self._time_words)

    @property
    def _spectral(self):
        """
        Returns spectral coordinates from WCS.
        Spectral world axis physical type name must contain 'em.'.
        Parameters
        ----------
        wcs : `astropy.wcs.wcsapi.BaseHighLevelWCS`
           The WCS object with which to calculate the coordinates. Must be
           self.wcs, self.extra_coords, or self.combined_wcs
        Returns
        -------
        : `astropy.coordinates.SpectralCoord` or `astropy.units.Quantity`
        """
        return self._get_coords_by_word(self._spectral_words)

    @property
    def _stokes(self):
        """
        Returns stokes polarization for all array indices in relevant axes.
        Stokes world axis physical type name must contain 'stokes'.
        Parameters
        ----------
        wcs : `astropy.wcs.wcsapi.BaseHighLevelWCS`
           The WCS object with which to calculate the coordinates. Must be
           self.wcs, self.extra_coords, or self.combined_wcs
        Returns
        -------
        :
        """
        return self._get_coords_by_word(self._stokes_words)

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
