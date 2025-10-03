import os
import pathlib
from datetime import UTC, datetime
from collections.abc import Generator

import astropy.units as u
import numpy as np
from ndcube import NDCube
from prefect import get_run_logger
from regularizepsf import ArrayPSFBuilder, ArrayPSFTransform, simple_functional_psf
from regularizepsf.util import calculate_covering

from punchbowl.data import NormalizedMetadata
from punchbowl.data.units import calculate_image_pixel_area, dn_to_msb
from punchbowl.level1.alignment import align_task
from punchbowl.level1.deficient_pixel import remove_deficient_pixels_task
from punchbowl.level1.despike import despike_task
from punchbowl.level1.destreak import destreak_task
from punchbowl.level1.initial_uncertainty import update_initial_uncertainty_task
from punchbowl.level1.psf import correct_psf_task
from punchbowl.level1.quartic_fit import perform_quartic_fit_task
from punchbowl.level1.sqrt import decode_sqrt_data
from punchbowl.level1.stray_light import remove_stray_light_task
from punchbowl.level1.vignette import correct_vignetting_task
from punchbowl.prefect import punch_flow
from punchbowl.util import DataLoader, load_image_task, output_image_task

KEYS_TO_NOT_COPY = ["BUNIT", "DESCRPTN", "FILENAME", "ISSQRT", "LEVEL", "PIPEVRSN", "TITLE", "TYPECODE", "FILEVRSN"]


@punch_flow
def generate_psf_model_core_flow(input_filepaths: list[str],
                                 masks: list[pathlib.Path | str] | np.ndarray | Generator = None,
                                 alpha: float = 2.0,
                                 epsilon: float = 0.3,
                                 image_shape: tuple[int, int] = (2048, 2048),
                                 psf_size: int = 32,
                                 target_fwhm: float = 3.25) -> ArrayPSFTransform:
    """Generate PSF model."""
    # Define the target PSF as a symmetric Gaussian
    center = psf_size / 2
    sigma = target_fwhm / 2.355
    @simple_functional_psf
    def target(row, col, x0=center, y0=center, sigma_x=sigma, sigma_y=sigma) -> np.ndarray:  # noqa: ANN001
        return np.exp(-(np.square(row - x0) / (2 * np.square(sigma_x))
                        + np.square(col - y0) / (2 * np.square(sigma_y))))

    image_psf, _ = ArrayPSFBuilder(psf_size).build(input_filepaths, hdu_choice=1, star_masks=masks)
    coords = calculate_covering(image_shape, psf_size)
    return ArrayPSFTransform.construct(image_psf, target.as_array_psf(coords, psf_size), alpha, epsilon)


@punch_flow
def level1_early_core_flow(  # noqa: C901
    input_data: list[str] | list[NDCube],
    gain_bottom: float = 4.9,
    gain_top: float = 4.9,
    dark_level: float = 55.81,
    read_noise_level: float = 17,
    bitrate_signal: int = 16,
    quartic_coefficient_path: str | pathlib.Path | DataLoader | None = None,
    despike_sigclip: float = 50,
    despike_sigfrac: float = 0.25,
    despike_objlim: float = 160.0,
    despike_niter: int = 10,
    despike_cleantype: str = "meanmask",
    exposure_time: float = 49 * 1000,
    readout_line_time: float = 163/2148,
    reset_line_time: float = 163/2148,
    vignetting_function_path: str | DataLoader | None = None,
    deficient_pixel_map_path: str | None = None,
    deficient_pixel_method: str = "median",
    deficient_pixel_required_good_count: int = 3,
    deficient_pixel_max_window_size: int = 10,
    output_filename: list[str] | None = None,
    max_workers: int | None = None,
    mask_path: str | None = None,
) -> list[NDCube]:
    """Core flow for level 1, doing preliminary stray light subtraction."""
    logger = get_run_logger()

    logger.info("beginning level 1 early core flow")

    output_data = []
    for i, this_data in enumerate(input_data):
        data = load_image_task(this_data) if isinstance(this_data, str) else this_data

        if data.meta["ISSQRT"].value:
            data = decode_sqrt_data(data)
        saturated_pixels = data.data >= data.meta["DSATVAL"].value
        data = perform_quartic_fit_task(data, quartic_coefficient_path)
        data = update_initial_uncertainty_task(data,
                                               dark_level=dark_level,
                                               gain_bottom=gain_bottom,
                                               gain_top=gain_top,
                                               read_noise_level=read_noise_level,
                                               bitrate_signal=bitrate_signal,
                                               saturated_pixels=saturated_pixels,
                                               )

        if data.meta["OBSCODE"].value == "4":
            scaling = {"gain_bottom": data.meta["GAINBTM"].value * u.photon / u.DN,
                       "gain_top": data.meta["GAINTOP"].value * u.photon / u.DN,
                       "wavelength": 530. * u.nm,
                       "exposure": data.meta["EXPTIME"].value * u.s,
                       "aperture": 49.57 * u.mm ** 2}
        else:
            scaling = {"gain_bottom": data.meta["GAINBTM"].value * u.photon / u.DN,
                       "gain_top": data.meta["GAINTOP"].value * u.photon / u.DN,
                       "wavelength": 530. * u.nm,
                       "exposure": data.meta["EXPTIME"].value * u.s,
                       "aperture": 34 * u.mm ** 2}
        pixel_scale = calculate_image_pixel_area(data.wcs, data.data.shape).to(u.sr) / u.pixel
        scaling["pixel_scale"] = pixel_scale

        # TODO - In dealing with converting the DSATVAL to MSB...
        # subtract bias, work with MSB conversion
        # watch out for linearization blowing up these values
        dsatval_msb = np.clip(dn_to_msb(np.zeros_like(data.data)+data.meta["DSATVAL"].value,
                                        data.wcs, **scaling), a_min=0, a_max=None)
        data.meta["DSATVAL"] = np.nanmin(dsatval_msb)

        data.data[:, :] = np.clip(dn_to_msb(data.data[:, :], data.wcs, **scaling), a_min=0, a_max=None)
        data.uncertainty.array[:, :] = dn_to_msb(data.uncertainty.array[:, :], data.wcs, **scaling)

        data = despike_task(data,
                            despike_sigclip,
                            despike_sigfrac,
                            despike_objlim,
                            despike_niter,
                            gain_bottom,  # TODO: despiking should handle the gain more completely
                            read_noise_level,
                            despike_cleantype,
                            max_workers=max_workers)
        data = destreak_task(data,
                             exposure_time=exposure_time,
                             reset_line_time=reset_line_time,
                             readout_line_time=readout_line_time,
                             max_workers=max_workers)
        data = correct_vignetting_task(data, vignetting_function_path)
        data = remove_deficient_pixels_task(data,
                                            deficient_pixel_map_path,
                                            required_good_count=deficient_pixel_required_good_count,
                                            max_window_size=deficient_pixel_max_window_size,
                                            method=deficient_pixel_method)

        if mask_path:
            with open(mask_path, "rb") as f:
                b = f.read()
            mask = np.unpackbits(np.frombuffer(b, dtype=np.uint8)).reshape(2048, 2048).T
            data.data *= mask
            data.uncertainty.array[mask==0] = np.inf

        # Repackage data with proper metadata
        product_code = data.meta["TYPECODE"].value + data.meta["OBSCODE"].value
        product_code = "X" + product_code[1:]
        new_meta = NormalizedMetadata.load_template(product_code, "1")
        new_meta["TITLE"] = new_meta["TITLE"].value + " with Stray Light"
        # copy over the existing values
        for key in data.meta.keys(): # noqa: SIM118
            if key in KEYS_TO_NOT_COPY:
                continue
            if key in new_meta.keys(): # noqa: SIM118
                new_meta[key] = data.meta[key].value
        new_meta.history = data.meta.history
        new_meta["DATE-OBS"] = data.meta["DATE-OBS"].value
        new_meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

        if isinstance(vignetting_function_path, DataLoader):
            vignetting_function_path = vignetting_function_path.src_repr()
        new_meta["CALVI"] = os.path.basename(vignetting_function_path) if vignetting_function_path else ""

        if isinstance(quartic_coefficient_path, DataLoader):
            quartic_coefficient_path = quartic_coefficient_path.src_repr()
        new_meta["CALCF"] = os.path.basename(quartic_coefficient_path) if quartic_coefficient_path else ""
        new_meta["FILEVRSN"] = data.meta["FILEVRSN"].value

        filename = data.meta.get("FILENAME").value
        if filename:
            new_meta.provenance = [filename]

        data = NDCube(data=data.data, meta=new_meta, wcs=data.wcs, unit=data.unit, uncertainty=data.uncertainty)

        if output_filename is not None and i < len(output_filename) and output_filename[i] is not None:
            output_image_task(data, output_filename[i])

        output_data.append(data)

    logger.info("ending level 1 early core flow")
    return output_data


@punch_flow
def level1_late_core_flow( # noqa: C901
    input_data: list[str] | list[NDCube],
    stray_light_before_path: str | None = None,
    stray_light_after_path: str | None = None,
    psf_model_path: str | DataLoader | None = None,
    distortion_path: str | None = None,
    mask_path: str | None = None,
    do_align: bool = True,
    output_as_Q_file: bool = False, # noqa: N803
    output_filename: list[str] | None = None,
    max_workers: int | None = None,
) -> list[NDCube]:
    """Core flow for level 1, applying final stray light subtraction."""
    logger = get_run_logger()

    logger.info("beginning level 1 late core flow")

    output_data = []
    for i, this_data in enumerate(input_data):
        data = load_image_task(this_data) if isinstance(this_data, str) else this_data
        data = remove_stray_light_task(data, stray_light_before_path, stray_light_after_path)

        if mask_path:
            with open(mask_path, "rb") as f:
                b = f.read()
            mask = np.unpackbits(np.frombuffer(b, dtype=np.uint8)).reshape(2048, 2048).T
            data.data[~mask] = 0
            data.uncertainty.array[~mask] = np.inf
        else:
            mask = None

        data = correct_psf_task(data, psf_model_path, max_workers=max_workers)
        if do_align:
            data = align_task(data, distortion_path)

        if mask is not None:
            data.data[~mask] = 0
            data.uncertainty.array[~mask] = np.inf

        # Repackage data with proper metadata
        product_code = data.meta["TYPECODE"].value + data.meta["OBSCODE"].value
        if output_as_Q_file:
            product_code = "Q" + product_code[1:]
        else:
            product_code = ("C" if product_code[1] == "R" else "P") + product_code[1:]
        new_meta = NormalizedMetadata.load_template(product_code, "1")
        # copy over the existing values
        for key in data.meta.keys():  # noqa: SIM118
            if key in KEYS_TO_NOT_COPY:
                continue
            if key in new_meta.keys():  # noqa: SIM118
                new_meta[key] = data.meta[key].value
        new_meta.history = data.meta.history
        new_meta["DATE-OBS"] = data.meta["DATE-OBS"].value
        new_meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

        if isinstance(psf_model_path, DataLoader):
            psf_model_path = psf_model_path.src_repr()
        new_meta["CALPSF"] = os.path.basename(psf_model_path) if psf_model_path else ""

        new_meta["CALSL0"] = os.path.basename(stray_light_before_path) if stray_light_before_path else ""
        new_meta["CALSL1"] = os.path.basename(stray_light_after_path) if stray_light_after_path else ""

        new_meta["FILEVRSN"] = data.meta["FILEVRSN"].value

        data = NDCube(data=data.data, meta=new_meta, wcs=data.wcs, unit=data.unit, uncertainty=data.uncertainty)

        if output_filename is not None and i < len(output_filename) and output_filename[i] is not None:
            output_image_task(data, output_filename[i])

        output_data.append(data)
    logger.info("ending level 1 late core flow")
    return output_data


@punch_flow
def levelh_core_flow(
    input_data: list[str] | list[NDCube],
    gain_bottom: float = 4.9,
    gain_top: float = 4.9,
    bias_level: float = 100,
    dark_level: float = 55.81,
    read_noise_level: float = 17,
    bitrate_signal: int = 16,
    psf_model_path: str | None = None,
    distortion_path: str | None = None,
    output_filename: str | None = None,
) -> list[NDCube]:
    """Core flow for level 0.5 also known as level H."""
    logger = get_run_logger()

    logger.info("beginning level H core flow")

    output_data = []
    for i, this_data in enumerate(input_data):
        data = load_image_task(this_data) if isinstance(this_data, str) else this_data
        data = decode_sqrt_data(data)
        data = update_initial_uncertainty_task(data,
                                               bias_level=bias_level,
                                               dark_level=dark_level,
                                               gain_bottom=gain_bottom,
                                               gain_top=gain_top,
                                               read_noise_level=read_noise_level,
                                               bitrate_signal=bitrate_signal,
                                               )

        data = correct_psf_task(data, psf_model_path)
        data = align_task(data, distortion_path)

        # Repackage data with proper metadata
        product_code = data.meta["TYPECODE"].value + data.meta["OBSCODE"].value
        new_meta = NormalizedMetadata.load_template(product_code, "H")
        new_meta["DATE-OBS"] = data.meta["DATE-OBS"].value
        new_meta["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

        output_header = new_meta.to_fits_header(data.wcs)
        for key in output_header:
            if (key in data.meta.keys()) and output_header[key] == "" and (key != "COMMENT") and (key != "HISTORY"): # noqa: SIM118
                new_meta[key].value = data.meta[key].value
        new_meta["FILEVRSN"] = data.meta["FILEVRSN"].value
        data = NDCube(data=data.data, meta=new_meta, wcs=data.wcs, unit=data.unit, uncertainty=data.uncertainty)

        if output_filename is not None and i < len(output_filename) and output_filename[i] is not None:
            output_image_task(data, output_filename[i])
        output_data.append(data)
        logger.info("ending level H core flow")
    return output_data
