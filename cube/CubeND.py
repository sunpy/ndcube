import astropy.nddata

__all__ = ['CubeND']

class CubeND(astropy.nddata.NDData):
	"""docstring for CubeND"""
	def __init__(self, data=None, wcs=None, **kwargs):
		mask = kwargs.get('mask', np.zeros(data.shape, dtype=bool))
		super(CubeND, self).__init__(data=data, mask=mask, **kwargs)
		self.arg = arg
		