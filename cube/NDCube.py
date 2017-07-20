import astropy.nddata

__all__ = ['NDCube', 'Cube2D', 'Cube1D']


class NDCube(astropy.nddata.NDData):
    """docstring for NDCube"""

    def __init__(self, data=None, wcs=None, **kwargs):
        super(NDCube, self).__init__(data=data, **kwargs)
        self.axes_wcs = wcs

    def pixel_to_world(self):
        pass

    def world_to_pixel(self):
        pass

    def to_sunpy(self):
        pass

    def dimension(self):
        pass

    def plot(self):
        pass

    def __getitem__(self, item):
        pass


class Cube2D(NDCube):
	"""docstring for Cube2D"""

	def __init__(self, data=None, wcs=None, **kwargs):
		super(Cube2D, self).__init__(data=data, wcs=wcs, **kwargs)

	def plot(self):
		pass

	def __getitem__(self, item):
		pass


class Cube1D(NDCube):
	"""docstring for Cube1D"""

	def __init__(self, data=None, wcs=None, **kwargs):
		super(Cube1D, self).__init__(data=data, wcs=wcs, **kwargs)

	def plot(self):
		pass

	def __getitem__(self, item):
		pass
