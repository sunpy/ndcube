import astropy.units as u

from astropy.nddata import StdDevUncertainty

uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))
mask = np.zeros_like(my_cube.data, dtype=bool)
meta = {"Description": "This is example NDCube metadata."}
unit = u.ct
