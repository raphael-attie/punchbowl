from typing import List

import numpy as np
from prefect import get_run_logger, task

import astropy.units as u

from punchbowl.data import PUNCHData

from ndcube import NDCollection

from solpolpy import resolve
from solpolpy.polarizers import mzp_to_bpb


def define_amatrix(arrayshape) -> np.ndarray:
    """
    Define an A matrix with which to convert MZP' (camera coords) = A x MZP (solar coords)

    Parameters
    -------
    arrayshape
        Defined input WCS array shape for matrix generation

    Returns
    -------
    ndarray
        Output A matrix used in converting between camera coordinates and solar coordinates

    """

    # Ideal MZP wrt Solar North
    thmzp = [-60, 0, 60] * u.degree

    long_arr, lat_arr = np.meshgrid(np.linspace(-20, 20, arrayshape[0]), np.linspace(-20, 20, arrayshape[1]))

    # Foreshortening (IMAX) effect on polarizer angle
    phi_m = np.arctan2(np.tan(thmzp[0]) * np.cos(long_arr * u.degree), np.cos(lat_arr * u.degree)) * 180 * u.degree / (
            np.pi * u.radian)
    phi_z = np.arctan2(np.tan(thmzp[1]) * np.cos(long_arr * u.degree), np.cos(lat_arr * u.degree)) * 180 * u.degree / (
            np.pi * u.radian)
    phi_p = np.arctan2(np.tan(thmzp[2]) * np.cos(long_arr * u.degree), np.cos(lat_arr * u.degree)) * 180 * u.degree / (
            np.pi * u.radian)

    phi = np.stack([phi_m, phi_z, phi_p])

    # Define the A matrix
    mat_a = np.empty((arrayshape[0], arrayshape[1], 3, 3))

    for i in range(3):
        for j in range(3):
            mat_a[:, :, i, j] = (4 * np.cos(phi[i] - thmzp[j]) ** 2 - 1) / 3

    return mat_a


def resolve_polarization(data_list: List[PUNCHData]) -> List[PUNCHData]:
    """
    Takes a set of input data in the camera MZP frame and converts to the solar MZP frame.

    Parameters
    ----------
    data_list : List[PUNCHData]
        List of PUNCHData objects on which to resolve polarization

    Returns
    -------
    List[PUNCHData]
        modified version of the input with polarization resolved

    """

    # # Unpack input data - originally in MZP w/r/t camera coordinates
    # data_mzp_camera = np.zeros([*data_list[0].data.shape, 3, 1])
    # for i, data in enumerate(data_list):
    #     data_mzp_camera[:, :, i, 0] = data.data
    #
    # # Generate the required A-matrix (MZP' (camera coords) = A x MZP (solar coords))
    # # This folds in the IMAX effect
    # mat_a = define_amatrix(data_list[0].data.shape)
    #
    # # Invert a-matrix
    # mat_a_inv = np.linalg.inv(mat_a)
    #
    # # Transform polarization from camera coordinates to solar coordinates
    # data_mzp_solar = np.matmul(mat_a_inv, data_mzp_camera)
    #
    # # Repackage data
    # # TODO - update WCS as well, or make new data objects?
    # for i, data in enumerate(data_list):
    #     data.duplicate_with_updates(data=data_mzp_solar[:, :, i, 0])

    # TODO - Rather than all this above... might want to just repackage data and pass to and from solpolpy?

    # Unpack data into a NDCollection object
    data_dictionary = dict((key, data) for key, data in zip(['Bm', 'Bz', 'Bp'], data_list))
    data_collection = NDCollection(data_dictionary)

    # Resolve polarization (test parameters for now)
    resolved_data_collection = resolve(data_collection, 'MZP')

    # Repack data
    # Create new data objects for each resolved polarization?
    data_list = []
    for key in resolved_data_collection.keys():
        data_list.append(resolved_data_collection[key])

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

    # Resolve polarization
    data_list = resolve_polarization(data_list)

    logger.info("resolve_polarization ended")

    for data_object in data_list:
        data_object.meta.history.add_now("LEVEL2-resolve_polarization", "polarization resolved")
    return data_list
