from ndcube_data_wcs import data, wcs
from ndcube_uncert_mask_meta_unit import uncertainty, mask, meta, unit

my_cube = NDCube(data, wcs=wcs, uncertainty=uncertainty, mask=mask, meta=meta, unit=unit)
