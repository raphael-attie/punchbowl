import warnings
import numpy as np

from astropy.time import Time
from ndcube import NDCube
from prefect import get_run_logger, task

from punchbowl.data import load_ndcube_from_fits
from punchbowl.exceptions import (
    IncorrectPolarizationStateWarning,
    IncorrectTelescopeWarning,
    InvalidDataError,
    LargeTimeDeltaWarning,
    NoCalibrationDataWarning,
    DataShapeError,
)

def create_coefficient_image(flat_coefficients: np.ndarray, image_shape: tuple) -> np.ndarray:
    """
    Given a set of coefficients that should apply for every pixel, convert them to the required format.

    Parameters
    ----------
    flat_coefficients : np.ndarray
        A one-dimensional list of coefficients that should apply to every pixel in the image.
        Coefficients should be ordered from the highest power to lowest as expected in `photometric_calibration`, e.g.
        f(i,j) = a+b*IMG[i,j]+c*IMG[i,j]^2 would have flat_coefficients of [c, b, a]
    image_shape : tuple
        A tuple of the shape of the image that will be calibrated using `photometric_calibration`

    Returns
    -------
    np.ndarray
        An image of coefficients that apply to every pixel as expected by `photometric_calibration`

    """
    return np.stack([np.ones(image_shape) * coeff for coeff in flat_coefficients], axis=2)


def create_constant_quartic_coefficients(img_shape: tuple) -> np.ndarray:
    """
    Create a constant coefficients image that preserves the original values, b = 1 and all other coefficients are 0.

    Parameters
    ----------
    img_shape : tuple[Int]
        size of the image to create the coefficients for

    Returns
    -------
    np.ndarray
        An image of coefficients that apply to every pixel as expected by `photometric_calibration`

    """
    return create_coefficient_image(np.array([0, 0, 0, 1, 0]), img_shape)


def photometric_calibration(image: np.ndarray, coefficient_image: np.ndarray) -> np.ndarray:
    """
    Compute a non-linear photometric calibration of PUNCH images.

    Parameters
    ----------
    image : np.ndarray
        Image to be corrected.

    coefficient_image : np.ndarray
        Frame containing uncertainty values.
        The first two dimensions are the spatial dimensions of the image.
        The last dimension iterates over the powers of the coefficients, starting with index 0 being the highest power
        and counting down.

    Returns
    -------
    np.ndarray
        a photometrically corrected frame

    Notes
    -----
    Each instrument is subject to an independent non-linear photometric response,
    which needs to be corrected. The module converts from raw camera digitizer
    number (DN) to photometric units at each pixel. Each pixel is replaced with
    the corresponding value of the quartic polynomial in the current calibration file data
    product for that particular camera.

    A quartic polynomial is applied as follows:

    .. math:: X_{i,j} = a_{i,j}+b_{i,j}*DN_{i,j}+c_{i,j}*DN_{i,j}^2+d_{i,j}*DN_{i,j}^3+e_{i,j}*DN_{i,j}^4

    for each pixel in the detector. Each quantity (a, b, c, d, e) is a function
    of pixel location (i,j), and is generated using dark current and Stim lamp
    maps. a = offset (dark and the bias). b, c, d, e = higher order terms.
    Specifically ``coefficient_image[i,j,:] = [e, d, c, b, a]`` (highest order terms first)

    As each pixel is independent, a quartic fit calibration file of
    dimensions 2k*2k*5 is constructed, with each layer containing one of the five
    polynomial coefficients for each pixel.

    Examples
    --------
    >>> punch_image = np.ones((100,100))
    >>> coefficient_image = create_coefficient_image(np.array([0, 0, 0, 1, 0]), punch_image.shape)
    >>> data = photometric_calibration(punch_image, coefficient_image)

    """
    if len(coefficient_image.shape) != 3:
        msg = "`coefficient_image` must be a 3-D image"
        raise DataShapeError(msg)

    # find the number of quartic fit coefficients
    num_coefficients = coefficient_image.shape[2]
    return np.sum(
        [coefficient_image[..., i] * np.power(image, num_coefficients - i - 1) for i in range(num_coefficients)],
        axis=0,
    )

@task
def perform_quartic_fit_task(data_object: NDCube, quartic_coefficients_path: str | None = None) -> NDCube:
    """
    Prefect task to perform the quartic fit calibration on the data.

    Parameters
    ----------
    data_object : PUNCHData
        a data object that needs calibration
    quartic_coefficients_path: Optional[str]
        path to a  cube of coefficients as produced by `create_coefficients_image` or `create_ones_coefficients_image`,
        skips correction if it is None

    Returns
    -------
    PUNCHData
        modified version of the input with the quartic fit correction applied

    See Also
    --------
    photometric_calibration

    """
    logger = get_run_logger()
    logger.info("perform_quartic_fit started")

    if quartic_coefficients_path is None:
        data_object.meta.history.add_now("LEVEL1-quartic_fit", "Quartic fit skipped")
        msg=f"Calibration file {quartic_coefficients_path} is unavailable, quartic fit correction not applied"
        warnings.warn(msg, NoCalibrationDataWarning)
    elif not quartic_coefficients_path.exists():
        msg = f"File {quartic_coefficients_path} does not exist."
        raise InvalidDataError(msg)
    else:
        quartic_coefficients = load_ndcube_from_fits(quartic_coefficients_path)
        if len(data_object.data.shape) != 2:
            msg = "`image` must be a 2-D image"
            raise DataShapeError(msg)
        if len(quartic_coefficients.data.shape) != 3:
            msg = "`coefficient_image` must be a 3-D image"
            raise DataShapeError(msg)
        quartic_coefficients_date = Time(quartic_coefficients.meta["DATE-OBS"].value)
        observation_date = Time(data_object.meta["DATE-OBS"].value)
        if abs((quartic_coefficients_date - observation_date).to("day").value) > 365:
            msg=(f"Calibration file {quartic_coefficients_path}"
                " contains data created greater than 1 year from the obsveration")
            warnings.warn(msg, LargeTimeDeltaWarning)
        if quartic_coefficients.meta["TELESCOP"].value != data_object.meta["TELESCOP"].value:
            msg=f"Incorrect TELESCOP value within {quartic_coefficients_path}"
            warnings.warn(msg, IncorrectTelescopeWarning)
        elif quartic_coefficients.meta["OBSLAYR1"].value != data_object.meta["OBSLAYR1"].value:
            msg=f"Incorrect polarization state within {quartic_coefficients_path}"
            warnings.warn(msg, IncorrectPolarizationStateWarning)
        elif quartic_coefficients.data.shape[1:] != data_object.data.shape:
            msg=f"Incorrect Quartic Fit function shape within {quartic_coefficients_path}"
            raise InvalidDataError(msg)
        else:
            new_data = photometric_calibration(data_object.data, quartic_coefficients.data)
            data_object.data[...] = new_data[...]
            data_object.meta.history.add_now("LEVEL1-quartic_fit",
                                             f"Quartic fit correction completed with {quartic_coefficients_path}")
    logger.info("perform_quartic_fit finished")

    return data_object
