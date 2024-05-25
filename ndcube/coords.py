from ndcube import NDCubeABC

class NDCoords:
    def __init__(self, ndcube):
        if not isinstance(ndcube, NDCubeABC):
            raise TypeError("ndcube must be an instance of NDCubeABC.")
        self._ndcube = ndcube
        self.axis_world_coords = self._ndcube.axis_world_coords
        self.axis_world_coords_values = self._ndcube.axis_world_coords_values

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
    def celestial(self):
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
        return self._get_coords_by_word("pos")

    @property
    def time(self):
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
        return self._get_coords_by_word("time")

    @property
    def spectral(self):
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
        return self._get_coords_by_word("em")

    @property
    def stokes(self):
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
        return self._get_coords_by_word("stokes")

    def _get_coords_by_word(self, coord_words):
            """
            Returns coordinates from a WCS corresponding to a world axis physical type.
            """
            if isinstance(coord_words, str):
                coord_words = [coord_words]
            # Check if words are in physical types of main WCS and extra coords.
            wcs_phys_types = []
            ec_phys_types = []
            for wcs, physical_types in zip([self.wcs, self.extra_coords.wcs], [wcs_phys_types, ec_phys_types]):
                if wcs is not None:
                    for name in wcs.world_axis_physical_types:
                        words = set(name.replace(":", ".").split("."))
                        if any(word in words for word in coord_words):
                            physical_types.append(name)
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
