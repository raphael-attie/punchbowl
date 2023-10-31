from typing import List

import numpy as np
from prefect import get_run_logger, task

import astropy.units as u

from punchbowl.data import PUNCHData


def define_amatrix() -> np.ndarray:
    """
    Define an A matrix with which to convert MZP' (camera coords) = A x MZP (solar coords)

    Returns
    -------
    ndarray
        Output A matrix used in converting between camera coordinates and solar coordinates

    """

    # TODO - Compose coordinate arrays from specified input WCS if needed

    # Ideal MZP wrt Solar North
    thmzp = [-60, 0, 60] * u.degree

    long_arr, lat_arr = np.meshgrid(np.linspace(-20, 20, 2048), np.linspace(-20, 20, 2048))

    # Foreshortening (IMAX) effect on polarizer angle
    phi_m = np.arctan2(np.tan(thmzp[0]) * np.cos(long_arr * u.degree), np.cos(lat_arr * u.degree)) * 180 * u.degree / (
                np.pi * u.radian)
    phi_z = np.arctan2(np.tan(thmzp[1]) * np.cos(long_arr * u.degree), np.cos(lat_arr * u.degree)) * 180 * u.degree / (
                np.pi * u.radian)
    phi_p = np.arctan2(np.tan(thmzp[2]) * np.cos(long_arr * u.degree), np.cos(lat_arr * u.degree)) * 180 * u.degree / (
                np.pi * u.radian)

    phi = np.stack([phi_m, phi_z, phi_p])

    # Define the A matrix
    mat_a = np.empty((2048, 2048, 3, 3))

    for i in range(3):
        for j in range(3):
            mat_a[:, :, i, j] = (4 * np.cos(phi[i] - thmzp[j]) ** 2 - 1) / 3

    return mat_a


def resolve_polarization(data_list: List[PUNCHData]) -> List[PUNCHData]:
    """
    Takes a set of input data in the camera MZP frame and converts to the solar MZP frame.

    Parameters
    ----------
    data_list

    Returns
    -------
    List[PUNCHData]
        modified version of the input with polarization resolved

    """

    # Unpack input data - originally in MZP w/r/t camera coordinates

    # Generate the required A-matrix (MZP' (camera coords) = A x MZP (solar coords))
    # This folds in the IMAX effect
    mat_a = define_amatrix()

    # Transform polarization from camera coordinates to solar coordinates

    # Repackage data

    return data_list


@task
def resolve_polarization_task(data_list: List[PUNCHData]) -> List[PUNCHData]:
    """Prefect task for polarization resolving

    Parameters
    ----------
    data_list : List[PUNCHData]
        List of PUNCHData objects on which to resolve polarization

    Returns
    -------
    List[PUNCHData]
        modified version of the input with polarization resolved
    """

    logger = get_run_logger()
    logger.info("resolve_polarization started")

    data_list = resolve_polarization(data_list)

    logger.info("resolve_polarization ended")

    for data_object in data_list:
        data_object.meta.history.add_now("LEVEL2-resolve_polarization", "polarization resolved")
    return data_list
