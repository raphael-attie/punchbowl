# Core Python imports
from datetime import datetime
from typing import Tuple, List

# Third party imports
import numpy as np
import reproject
from astropy.wcs import WCS
from prefect import task, flow, get_run_logger

# Punchbowl imports
from punchbowl.data import PUNCHData
from punchbowl.level2.image_merge import reproject_array


# core module flow
@flow
def quickpunch_merge_flow(data: List) -> PUNCHData:
    logger = get_run_logger()
    logger.info("QuickPUNCH_merge module started")

    # TODO - Think about how to extract as much code as possible from the flow here into a shared function

    # Define output WCS from file
    trefoil_wcs = WCS('level2/data/trefoil_hdr.fits')
    trefoil_shape = (4096,4096)
    trefoil_wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
    #trefoil_shape = trefoil_wcs.array_shape

    # Unpack input data objects
    data_input, uncertainty_input, wcs_input = [], [], []
    for obj in data:
        data_input.append(obj.data)
        uncertainty_input.append(obj.uncertainty)
        wcs_input.append(obj.wcs)

    reprojected_data = np.zeros([trefoil_shape[0], trefoil_shape[1], len(data_input)])
    reprojected_uncertainty = np.zeros([trefoil_shape[0], trefoil_shape[1], len(data_input)])

    data_result = [reproject_array.submit(idata, iwcs, trefoil_wcs, trefoil_shape)
                   for idata, iwcs in zip(data_input, wcs_input)]
    uncertainty_result = [reproject_array.submit(iuncertainty.array, iwcs, trefoil_wcs, trefoil_shape)
                          for iuncertainty, iwcs in zip(uncertainty_input, wcs_input)]

    for i, d in enumerate(data_result):
        reprojected_data[:, :, i] = d.result()

    for i, d in enumerate(uncertainty_result):
        reprojected_uncertainty[:, :, i] = d.result()

    # Merge these data
    # Carefully deal with missing data (NaN) by only ignoring a pixel missing from all observations
    trefoil_data = np.nansum(reprojected_data * reprojected_uncertainty, axis=2) / np.nansum(reprojected_uncertainty, axis=2)
    trefoil_uncertainty = np.amax(reprojected_uncertainty)

    # Pack up an output data object
    data_object = PUNCHData(trefoil_data, uncertainty=trefoil_uncertainty, wcs=trefoil_wcs)
    
    logger.info("QuickPUNCH_merge flow finished")
    data_object.meta.history.add_now("LEVEL2-module", "QuickPUNCH_merge ran")
    return data_object