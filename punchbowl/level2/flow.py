import numpy as np
import solpolpy
from ndcube import NDCollection, NDCube
from prefect import flow, get_run_logger

from punchbowl.data import get_base_file_name, load_trefoil_wcs
from punchbowl.data.wcs import load_quickpunch_mosaic_wcs, load_quickpunch_nfi_wcs
from punchbowl.exceptions import IncorrectFileCountError
from punchbowl.level2.merge import merge_many_task
from punchbowl.level2.polarization import resolve_polarization_task
from punchbowl.level2.resample import reproject_many_flow
from punchbowl.util import load_image_task, output_image_task

ORDER = ["PM1", "PZ1", "PP1",
         "PM2", "PZ2", "PP2",
         "PM3", "PZ3", "PP3",
         "PM4", "PZ4", "PP4"]


@flow(validate_parameters=False)
def level2_core_flow(data_list: list[str] | list[NDCube],
                     # voter_filenames: list[list[str]],
                     output_filename: str | None = None) -> list[NDCube]:
    """Level 2 core flow."""
    logger = get_run_logger()
    logger.info("beginning level 2 core flow")
    if len(data_list) != 12:
        msg = f"Received {len(data_list)} files when need 12."
        raise IncorrectFileCountError(msg)

    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]

    # order the data list so it can be processed properly
    ordered_data_list = []
    for order_element in ORDER:
        found = False
        for data_element in data_list:
            typecode = data_element.meta["TYPECODE"].value
            obscode = data_element.meta["OBSCODE"].value
            if typecode == order_element[:2] and obscode == order_element[2]:
                ordered_data_list.append(data_element)
                found = True
        if not found:
            msg = f"Did not receive {order_element} file."
            raise IncorrectFileCountError(msg)
    logger.info(f"Ordered files are {[get_base_file_name(cube) for cube in ordered_data_list]}")

    trefoil_wcs, trefoil_shape = load_trefoil_wcs()

    # TODO make more robust
    data_list = (resolve_polarization_task(ordered_data_list[:3]) + resolve_polarization_task(ordered_data_list[3:6])
                 + resolve_polarization_task(ordered_data_list[6:9]) + resolve_polarization_task(ordered_data_list[9:]))
    data_list = reproject_many_flow(data_list, trefoil_wcs, trefoil_shape)
    # data_list = [identify_bright_structures_task(cube, voter_filenames)
    #              for cube, voter_filenames in zip(data_list, voter_filenames)]
    # TODO: merge only similar polarizations together
    output_data = merge_many_task(data_list, trefoil_wcs)

    if output_filename is not None:
        output_image_task(output_data, output_filename)

    logger.info("ending level 2 core flow")
    return [output_data]


def remix_polarization(input_data: NDCube) -> NDCube:
    """Remix polarization from (M,Z,P) to (B, pB) using solpolpy."""
    # Unpack data into a NDCollection object
    data_collection = NDCollection(
        [("M", NDCube(data=input_data.data[0], wcs=input_data.wcs)),
         ("Z", NDCube(data=input_data.data[1], wcs=input_data.wcs)),
         ("P", NDCube(data=input_data.data[2], wcs=input_data.wcs))],
        aligned_axes="all")

    resolved_data_collection = solpolpy.resolve(data_collection, "BpB", imax_effect=False)

    # Repack data
    data_list = []
    wcs_list = []
    uncertainty_list = []
    for key in resolved_data_collection:
        data_list.append(resolved_data_collection[key].data)
        wcs_list.append(resolved_data_collection[key].wcs)
        uncertainty_list.append(resolved_data_collection[key].uncertainty)

    # Remove alpha channel
    data_list.pop()
    wcs_list.pop()
    uncertainty_list.pop()

    # Repack into a PUNCHData object
    new_data = np.stack(data_list, axis=0)
    new_uncertainty = np.stack(uncertainty_list, axis=0) if uncertainty_list[0] is not None else None

    new_wcs = input_data.wcs.copy()

    return NDCube(data=new_data, wcs=new_wcs, uncertainty=new_uncertainty, meta=input_data.meta)


# TODO - Split this into multiple flows?
# TODO - Generate quickPUNCH f-corona products?
@flow(validate_parameters=False)
def levelq_core_flow(data_list: list[str] | list[NDCube],
                     output_filename: str | None = None) -> list[NDCube]:
    """Level quickPUNCH core flow."""
    logger = get_run_logger()
    logger.info("beginning level quickPUNCH core flow")
    # TODO - Check file input number for quickpunch products
    if len(data_list) != 12:
        msg = f"Received {len(data_list)} files when need 12."
        raise IncorrectFileCountError(msg)

    data_list = [load_image_task(d) if isinstance(d, str) else d for d in data_list]

    # order the data list so it can be processed properly
    ordered_data_list = []
    for order_element in ORDER:
        found = False
        for data_element in data_list:
            typecode = data_element.meta["TYPECODE"].value
            obscode = data_element.meta["OBSCODE"].value
            if typecode == order_element[:2] and obscode == order_element[2]:
                ordered_data_list.append(data_element)
                found = True
        if not found:
            msg = f"Did not receive {order_element} file."
            raise IncorrectFileCountError(msg)
    logger.info(f"Ordered files are {[get_base_file_name(cube) for cube in ordered_data_list]}")

    quickpunch_mosaic_wcs, quickpunch_mosaic_shape = load_quickpunch_mosaic_wcs()
    quickpunch_nfi_wcs, quickpunch_nfi_shape = load_quickpunch_nfi_wcs()

    data_list_mosaic = reproject_many_flow(data_list, quickpunch_mosaic_wcs, quickpunch_mosaic_shape)
    data_list_nfi = reproject_many_flow(data_list, quickpunch_nfi_wcs, quickpunch_nfi_shape)

    # TODO - Skip merge for NFI?
    output_data_mosaic = merge_many_task(data_list_mosaic, quickpunch_mosaic_wcs)
    output_data_nfi = merge_many_task(data_list_nfi, quickpunch_nfi_wcs)

    output_clear_mosaic = remix_polarization(output_data_mosaic)[0]
    output_clear_nfi = remix_polarization(output_data_nfi)[0]

    # TODO - Separate filenames needed / split into two flows?
    # TODO - Generate filename automatically for correct format?
    if output_filename is not None:
        output_image_task(output_clear_mosaic, output_filename)
        output_image_task(output_clear_nfi, output_filename)

    logger.info("ending level quickPUNCH core flow")
    return [output_data_mosaic, output_data_nfi]


if __name__ == "__main__":
    import os
    import glob

    filenames = sorted(glob.glob("/d0/punchsoc/gamera_data/forward_l1/*.fits"),
                       key=lambda s: os.path.basename(s).split("_")[3])


    # TODO - Running with punchpipe should fix all this?
    times = {filename.split("_")[-2] for filename in filenames}

    for time in times:
        input_files = [filename for filename in filenames if time in filename]

    levelq_core_flow(filenames,
                     output_filename="/d0/punchsoc/gamera_data/forward_lq/file.fits")
