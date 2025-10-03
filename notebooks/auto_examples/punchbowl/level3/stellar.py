from math import floor
from datetime import UTC, datetime

import astropy.units as u
import numpy as np
import remove_starfield
from astropy.wcs import WCS
from dateutil.parser import parse as parse_datetime_str
from ndcube import NDCollection, NDCube
from prefect import get_run_logger
from remove_starfield import ImageHolder, ImageProcessor, Starfield
from remove_starfield.reducers import PercentileReducer
from solpolpy import resolve

from punchbowl.data import NormalizedMetadata, load_ndcube_from_fits
from punchbowl.data.wcs import calculate_helio_wcs_from_celestial, get_p_angle
from punchbowl.prefect import punch_flow, punch_task


def to_celestial(input_data: NDCube) -> NDCube:
    """
    Convert polarization from mzpsolar to Celestial frame.

    All images need their polarization converted to Celestial frame
    to generate the background starfield model.
    """
    # Create a data collection for M, Z, P components
    mzp_angles = [-60, 0, 60]*u.degree

    # Compute new angles for celestial frame
    cel_north_offset = get_p_angle(time=input_data[0].meta["DATE-OBS"].value)
    new_angles = mzp_angles - cel_north_offset

    collection_contents = [
        (label,
         NDCube(data=input_data[i].data,
                wcs=input_data.wcs.dropaxis(2),
                meta={"POLAR": angle}))
        for label, i, angle in zip(["M", "Z", "P"], [0, 1, 2], mzp_angles, strict=False)
    ]
    data_collection = NDCollection(collection_contents, aligned_axes="all")

    # Resolve data to celestial frame
    celestial_data_collection = resolve(data_collection, "npol", out_angles=new_angles, imax_effect=False)

    valid_keys = [key for key in celestial_data_collection if key != "alpha"]
    new_data = [celestial_data_collection[key].data for key in valid_keys]
    new_wcs = input_data.wcs.copy()

    output_meta = NormalizedMetadata.load_template("PTM", "3")
    output_meta["DATE-OBS"] = input_data.meta["DATE-OBS"].value

    output = NDCube(data=new_data, wcs=new_wcs, meta=output_meta)
    output.meta.history.add_now("LEVEL3-convert2celestial", "Convert mzpsolar to Celestial")

    return output


def from_celestial(input_data: NDCube) -> NDCube:
    """
    Convert polarization from Celestial frame to mzpsolar.

    All images need their polarization converted back to Solar frame
    after removing the stellar polarization.
    """
    # Create a data collection for M, Z, P components
    mzp_angles = [-60, 0, 60]*u.degree
    # Compute new angles for celestial frame
    cel_north_offset = get_p_angle(time=input_data[0].meta["DATE-OBS"].value)
    new_angles = mzp_angles - cel_north_offset
    collection_contents = [
        (f"{angle.value} deg",
         NDCube(data=input_data[i].data,
                wcs=input_data.wcs.dropaxis(2),
                meta={"POLAR": angle}))
        for i, angle in enumerate(new_angles)
    ]
    data_collection = NDCollection(collection_contents, aligned_axes="all")

    # Resolve data to mzpsolar frame
    solar_data_collection = resolve(data_collection, "mzpsolar", imax_effect=False)

    valid_keys = [key for key in solar_data_collection if key != "alpha"]
    new_data = [solar_data_collection[key].data for key in valid_keys]
    new_wcs = input_data.wcs.copy()

    output_meta = NormalizedMetadata.load_template("PTM", "3")
    output_meta["DATE-OBS"] = input_data.meta["DATE-OBS"].value

    output = NDCube(data=new_data, wcs=new_wcs, meta=output_meta, uncertainty=input_data.uncertainty)
    output.meta.history.add_now("LEVEL3-convert2mzpsolar", "Convert Celestial to mzpsolar")

    return output


class PUNCHImageProcessor(ImageProcessor):
    """Special loader for PUNCH data."""

    def __init__(self, layer: int | None, apply_mask: bool = True, key: str = " ") -> None:
        """Create PUNCHImageProcessor."""
        self.layer: int | None = layer
        self.apply_mask = apply_mask
        self.key = key

    def load_image(self, filename: str) -> ImageHolder:
        """Load an image."""
        cube = load_ndcube_from_fits(filename, key=self.key)

        if self.layer is None:  # it's a clear image
            data = cube.data
        else:  # it's polarized
            cube = to_celestial(cube)
            data = cube.data[self.layer]

        if self.apply_mask:
            data[data==0] = np.nan
        return ImageHolder(data, cube.wcs.celestial, cube.meta)


@punch_flow(log_prints=True, timeout_seconds=21_600)
def generate_starfield_background(
        filenames: list[str],
        map_scale: float = 0.01,
        target_mem_usage: float = 1000,
        n_procs: int | None = None,
        reference_time: datetime | None = None,
        is_polarized: bool = False) -> [NDCube, NDCube]:
    """Create a background starfield_bg map from a series of PUNCH images over a long period of time."""
    logger = get_run_logger()

    if reference_time is None:
        reference_time = datetime.now(UTC)
    elif isinstance(reference_time, str):
        reference_time = parse_datetime_str(reference_time)

    logger.info("construct_starfield_background started")

    # create an empty array to fill with data
    #   open the first file in the list to ge the shape of the file
    if len(filenames) == 0:
        msg = "filenames cannot be empty"
        raise ValueError(msg)

    shape = [floor(132 / map_scale), floor(360 / map_scale)]
    starfield_wcs = WCS(naxis=2)
    # n.b. it seems the RA wrap point is chosen so there's 180 degrees
    # included on either side of crpix
    crpix = [shape[1] / 2 + .5, shape[0] / 2 + .5]
    starfield_wcs.wcs.crpix = crpix
    starfield_wcs.wcs.crval = 270, -23.5
    starfield_wcs.wcs.cdelt = map_scale, map_scale
    starfield_wcs.wcs.ctype = "RA---CAR", "DEC--CAR"
    starfield_wcs.wcs.cunit = "deg", "deg"
    starfield_wcs.array_shape = shape

    meta = NormalizedMetadata.load_template("PSM" if is_polarized else "CSM", "3")
    meta["DATE-OBS"] = reference_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    meta["DATE-BEG"] = reference_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    meta["DATE-END"] = reference_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    meta["DATE-AVG"] = reference_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    if is_polarized:
        logger.info("Starting m starfield")
        starfield_m = remove_starfield.build_starfield_estimate(
            filenames,
            attribution=False,
            frame_count=False,
            reducer=PercentileReducer(10),
            starfield_wcs=starfield_wcs,
            n_procs=n_procs,
            processor=PUNCHImageProcessor(0, apply_mask=True, key="A"),
            target_mem_usage=target_mem_usage)
        logger.info("Ending m starfield")

        logger.info("Starting z starfield")
        starfield_z = remove_starfield.build_starfield_estimate(
            filenames,
            attribution=False,
            frame_count=False,
            reducer=PercentileReducer(10),
            starfield_wcs=starfield_wcs,
            n_procs=n_procs,
            processor=PUNCHImageProcessor(1, apply_mask=True, key="A"),
            target_mem_usage=target_mem_usage)
        logger.info("Ending z starfield")

        logger.info("Starting p starfield")
        starfield_p = remove_starfield.build_starfield_estimate(
            filenames,
            attribution=False,
            frame_count=False,
            reducer=PercentileReducer(10),
            starfield_wcs=starfield_wcs,
            n_procs=n_procs,
            processor=PUNCHImageProcessor(2, apply_mask=True, key="A"),
            target_mem_usage=target_mem_usage)
        logger.info("Ending p starfield")

        out_data = np.stack([starfield_m.starfield, starfield_z.starfield, starfield_p.starfield], axis=0)
        out_wcs, _ = calculate_helio_wcs_from_celestial(starfield_m.wcs, meta.astropy_time, starfield_m.starfield.shape)
    else:
        logger.info("Starting clear starfield")
        starfield_clear = remove_starfield.build_starfield_estimate(
            filenames,
            attribution=False,
            frame_count=False,
            reducer=PercentileReducer(10),
            starfield_wcs=starfield_wcs,
            n_procs=n_procs,
            processor=PUNCHImageProcessor(None, apply_mask=True, key="A"),
            target_mem_usage=target_mem_usage)
        logger.info("Ending clear starfield")
        out_data = starfield_clear.starfield
        out_wcs, _ = calculate_helio_wcs_from_celestial(starfield_clear.wcs,
                                                        meta.astropy_time,
                                                        starfield_clear.starfield.shape)

    output = NDCube(data=out_data, wcs=out_wcs, meta=meta)
    output.meta.history.add_now("LEVEL3-starfield_background", "constructed starfield_bg model")

    logger.info("construct_starfield_background finished")
    return [output]


@punch_task
def subtract_starfield_background_task(data_object: NDCube,
                                       starfield_background_path: str | None,
                                       is_polarized: bool = False) -> NDCube:
    """
    Subtracts a background starfield from an input data frame.

    checks the dimensions of input data frame and background starfield match and
    subtracts the background starfield from the data frame of interest.

    Parameters
    ----------
    data_object : NDCube
        A NDCube data frame to be background subtracted

    starfield_background_path : str
        path to a NDCube background starfield map

    Returns
    -------
    NDCube
        A background starfield subtracted data frame

    """
    logger = get_run_logger()
    logger.info("subtract_starfield_background started")

    if starfield_background_path is None:
        output = data_object
        output.meta.history.add_now("LEVEL3-subtract_starfield_background",
                                           "starfield subtraction skipped since path is empty")
    else:
        star_datacube = load_ndcube_from_fits(starfield_background_path, key="A")
        starfield_model = Starfield(np.stack((star_datacube.data, star_datacube.uncertainty.array)), star_datacube.wcs)

        subtracted = starfield_model.subtract_from_image(
            NDCube(data=np.stack((data_object.data, data_object.uncertainty.array)),
                   wcs=data_object.wcs.celestial,
                   meta=data_object.meta),
            processor=PUNCHImageProcessor(0, key="A"))

        data_object.data[...] = subtracted.subtracted[0]
        data_object.uncertainty.array[...] -= subtracted.subtracted[1]
        data_object.meta.history.add_now("LEVEL3-subtract_starfield_background", "subtracted starfield background")
        if is_polarized:
            output = from_celestial(data_object)
        else:
            output = data_object
    logger.info("subtract_starfield_background finished")

    return output


def create_empty_starfield_background(data_object: NDCube) -> np.ndarray:
    """Create an empty starfield background map."""
    return np.zeros_like(data_object.data)
