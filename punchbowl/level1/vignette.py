import pathlib

from astropy.io import fits
from prefect import get_run_logger, task

from punchbowl.data import PUNCHData
from punchbowl.exceptions import InvalidDataError


@task
def correct_vignetting_task(data_object: PUNCHData, vignetting_file: pathlib) -> PUNCHData:
    """Prefect task to correct the vignetting of an image

    Parameters
    ----------
    data_object : PUNCHData
        data to operate on
    vignetting_file: pathlib
        path to vignetting function to apply to input data

    Returns
    -------
    PUNCHData
        modified version of the input with the vignetting corrected
    """
    logger = get_run_logger()
    logger.info("correct_vignetting started")

    if vignetting_file is None:
        data_object.meta.history.add_now("LEVEL1-correct_vignetting", "Vignetting skipped")
    elif not vignetting_file.exists():
        raise InvalidDataError(f"File {vignetting_file} does not exist.")
    else:
        with fits.open(vignetting_file) as hdul:
            vignetting_header = hdul[1].header
            vignetting_function = hdul[1].data

        if vignetting_header['TELESCOP'] != data_object.meta['TELESCOP'].value:
            raise InvalidDataError(f"Incorrect TELESCOP value within {vignetting_file}")
        elif vignetting_header['OBSLAYR1'] != data_object.meta['OBSLAYR1'].value:
            raise InvalidDataError(f"Incorrect polarization state within {vignetting_file}")
        elif vignetting_function.shape != data_object.data.shape:
            raise InvalidDataError(f"Incorrect vignetting function shape within {vignetting_file}")
        else:
            data_object.data[:,:] /= vignetting_function[:,:]
            data_object.meta.history.add_now("LEVEL1-correct_vignetting", "Vignetting corrected")

    logger.info("correct_vignetting finished")
    return data_object
