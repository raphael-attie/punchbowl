from datetime import datetime

import numpy as np
from prefect import task, get_run_logger

from punchbowl.data import PUNCHData
from punchbowl.util import validate_image_is_square


def streak_correction_matrix(n: int, exposure_time: float,
                             readout_line_time: float,
                             reset_line_time: float) -> np.ndarray:
    """Computes a matrix used in correcting streaks in PUNCH images

    Computes the inverse of a matrix of size n where the major diagonal
        contains the value diag, the lower triangle contains below and the
        upper triangle contains the value above.
            i.e. X[i,i]=diag, X[0:i-1,i]=below, X[0,i+1:n-1]=above

    Adapted from solarsoft sc_inverse

    Parameters
    ----------
    n : int
        size of the matrix (n x n)
    exposure_time : float
        the exposure time, i.e. value on the diagonal of the matrix
    readout_line_time : float
        the readout line time, i.e. value in the lower triangle
    reset_line_time : float
        the reset line time, i.e. value in the upper triangle

    Returns
    -------
    np.ndarray
        value of specified streak correction array

    Raises
    ------
    np.linalg.LinAlgError: Singular matrix
        Matrix isn't invertible

    Notes
    -----
    As long as the units are consistent, this should work. For PUNCH, we use milliseconds.

    Examples
    --------
    >>> streak_correction_matrix(3, 1, 2, 3)
    array([[-0.38461538,  0.23076923,  0.46153846],
           [ 0.30769231, -0.38461538,  0.23076923],
           [ 0.15384615,  0.30769231, -0.38461538]])
    """
    lower = np.tril(np.ones((n, n)) * readout_line_time, -1)
    upper = np.triu(np.ones((n, n)) * reset_line_time, 1)
    diagonal = np.diagflat(np.ones(n) * exposure_time)
    full_matrix = lower + upper + diagonal
    return np.linalg.inv(full_matrix)


def correct_streaks(image: np.ndarray,
                    exposure_time: float,
                    readout_line_time: float,
                    reset_line_time: float) -> np.ndarray:
    """Corrects an image for streaks

    Parameters
    ----------
    image : np.ndarray (2D)
        image to be corrected
    exposure_time : float
        exposure time in consistent units (e.g. milliseconds) with readout_line_time and reset_line time
    readout_line_time : float
        time to read out a line in consistent units (e.g. milliseconds) with exposure_time and reset_line time
    reset_line_time : float
        time to reset CCD in consistent units (e.g. milliseconds) with readout_line_time and exposure_time

    Returns
    -------
    np.ndarray
        a streak-corrected image

    Raises
    ------
    ValueError
        If the image is not 2D or not square
    TypeError
        If the image is not a numpy array
    np.linalg.LinAlgError: Singular matrix
        Matrix isn't invertible

    Examples
    --------
    >>> correct_streaks(np.arange(9).reshape(3,3), 1, 2, 3)
    array([[ 3.46153846,  3.76923077,  4.07692308],
       [ 0.23076923,  0.38461538,  0.53846154],
       [-1.38461538, -1.30769231, -1.23076923]])

    """
    validate_image_is_square(image)
    correction_matrix = streak_correction_matrix(image.shape[0],
                                                 exposure_time,
                                                 readout_line_time,
                                                 reset_line_time)
    return correction_matrix @ image


@task
def destreak_task(data_object: PUNCHData) -> PUNCHData:
    logger = get_run_logger()
    logger.info("destreak started")
    # TODO: do destreaking in here
    logger.info("destreak finished")
    data_object.add_history(datetime.now(), "LEVEL1-destreak", "image destreaked")
    return data_object

