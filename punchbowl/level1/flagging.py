# Core Python imports
from datetime import datetime

# Third party imports
import numpy as np
from prefect import task, get_run_logger
# from astropy.io import fits

# Punchbowl imports
from punchbowl.data import PUNCHData


def flag_punchdata(data_object: PUNCHData, bad_pixel_map: np.ndarray = None) -> PUNCHData:

    # Flag bad data in the primary data array
    data_object.data[bad_pixel_map] = 0

    # Flag bad data in the associated uncertainty array (coded with -1)
    data_object.uncertainty.array[bad_pixel_map] = -1

    return data_object


@task
def flag_task(data_object: PUNCHData) -> PUNCHData:
    logger = get_run_logger()
    logger.info("flagging started")

    # Read bad pixel map from file
    bad_pixel_filename = '../data/badpixelmap-' + data_object.meta["TYPECODE"] + data_object.meta["OBSRVTRY"] + '.npz'
    bad_pixel_map = np.load(bad_pixel_filename)['arr_0']

    # Call data flagging function
    data_object = flag_punchdata(data_object, bad_pixel_map)

    logger.info("flagging finished")
    data_object.add_history(datetime.now(), "LEVEL1-flagging", "image flagged")
    return data_object
