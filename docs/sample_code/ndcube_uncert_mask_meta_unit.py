import numpy as np
from ndcube_data_wcs import data

import astropy.units as u
from astropy.nddata import StdDevUncertainty

uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))
mask = np.zeros_like(data, dtype=bool)
meta = {"Description": "This is example NDCube metadata."}
unit = u.ct
