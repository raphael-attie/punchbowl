import numpy as np
from typing import Optional
from datetime import datetime
from punchbowl.data import PUNCHData
from prefect import task, get_run_logger


def create_coefficient_image(flat_coefficients: np.ndarray, image_shape: tuple) -> np.ndarray:
    """Given a set of coefficients that should apply for every pixel,
        converts them to the required coefficient_image format.

    Parameters
    ----------
    flat_coefficients : np.ndarray
        A one-dimensional list of coefficients that should apply to every pixel in the image.
        Coefficients should be ordered from highest power to lowest as expected in `photometric_calibration`, e.g.
        f(i,j) = a+b*IMG[i,j]+c*IMG[i,j]^2 would have flat_coefficients of [c, b, a]
    image_shape : tuple
        A tuple of the shape of the image that will be calibrated using `photometric_calibration`

    Returns
    -------
    An image of coefficients that apply to every pixel as expected by `photometric_calibration`

    """
    return np.stack([np.ones(image_shape)*coeff for coeff in flat_coefficients], axis=2)


def photometric_calibration(image: np.ndarray, coefficient_image: np.ndarray) -> np.ndarray:
    """Computes a non-linear photometric callibration of PUNCH images

    Each instrument is subject to an independent non-linear photometric response,
    which needs to be corrected. The module converts from raw camera digitizer 
    number (DN) to photometric units at each pixel. Each pixel is replaced with 
    the corresponding value of the quartic polynomial in the current CF data 
    product for that particular camera. 

    A quartic polynomial is applied as follows:
        X[i,j]=a[i,j]+b[i,j]*DN[i,j]+c[i,j]*DN[i,j]^2+d[i,j]*DN[i,j]^3+e[i,j]*DN[i,j]^4
    for each pixel in the detector. Where each quantity (a, b, c, d, e) is a function
    of pixel location [i,j], and is generated using dark current and Stim lamp 
    maps. Where:
        - a = offset (dark and the bias).
        - b, b, c, d, e = higher order terms.
        Specifically coefficient_image[i,j,:] = [e, d, c, b, a] (highest order terms first)

    As each pixel is independent, a quartic fit calibration file (CF) of 
    dimensions 2k*2k*5 is constructed, with each layer containing one of the five 
    polynomial coefficients for each pixel.

    Parameters
    ----------
    image : np.ndarray
        Image to be corrected.
    coefficient_image : np.ndarray
        Frame containing uncertainty values.
        The first two dimensions are the spatial dimensions of the image.
        The last dimension iterates over the powers of the coefficients. Starting with index 0 being the highest power
        and counting down.

    Returns
    -------
    np.ndarray
        a photometrically corrected frame

    Examples
    --------
    # TODO: add example
    """

    # inspect dimensions
    assert len(image.shape) == 2, "`image` must be a 2-D image"
    assert len(coefficient_image.shape) == 3, "`coefficient_image` must be a 3-D image"

    # inspect dimensions of correction map and data_frame
    assert coefficient_image.shape[:-1] == image.shape, "`coefficient_image` and `image` must have the same shape`"

    # find the number of quartic fit coefficients
    num_coeffs = coefficient_image.shape[2]
    corrected_data = np.sum([coefficient_image[..., i] * np.power(image, num_coeffs-i-1) for i in range(num_coeffs)],
                            axis=0)
    return corrected_data


@task
def perform_quartic_fit_task(data_object):
    logger = get_run_logger()
    logger.info("perform_quartic_fit started")
    # TODO: perform calibration
    logger.info("perform_quartic_fit finished")
    data_object.add_history(datetime.now(), "LEVEL1-perform_quartic_fit", "Quartic fit correction completed")
    return data_object
