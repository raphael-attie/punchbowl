import astropy.wcs.utils
import numpy as np
import reproject
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube
from prefect import get_run_logger
from scipy.ndimage import distance_transform_edt

from punchbowl.data.wcs import calculate_celestial_wcs_from_helio
from punchbowl.prefect import punch_flow, punch_task


@punch_task(tags=["reproject"])
def reproject_cube(input_cube: NDCube, output_wcs: WCS, output_shape: tuple[int, int]) -> np.ndarray:
    """
    Core reprojection function.

    Core reprojection function of the PUNCH mosaic generation module.
        With an input data array and corresponding WCS object, the function
        performs a reprojection into the output WCS object system, along with
        a specified pixel size for the output array. This utilizes the adaptive
        reprojection routine implemented in the reprojection astropy package.

    Parameters
    ----------
    input_cube: NDCube
        input cube to be reprojected
    output_wcs
        astropy WCS object describing the coordinate system to transform to
    output_shape
        pixel shape of the reprojected output array

    Returns
    -------
    np.ndarray
        output array after reprojection of the input array


    Example Call
    ------------
    >>> reprojected_arrays = reproject_cube(input_cube, output_wcs, output_shape)

    """
    input_wcs = input_cube.wcs
    time = input_cube.meta.astropy_time

    celestial_source = calculate_celestial_wcs_from_helio(input_wcs, time, output_shape)
    celestial_target = calculate_celestial_wcs_from_helio(output_wcs, time, output_shape)

    # When we build mosaics, each input image fills only a portion (less than half) of the output frame. When we
    # reproject, we don't want it spending time looping over all those empty pixels, calculating coordinates,
    # etc. So here we find a bounding box around the input in the output frame and crop to that before reprojecting.
    # To start, here we make a grid of points along the edges of the input image.
    xs = np.linspace(-1, input_cube.data.shape[-1], 60)
    ys = np.linspace(-1, input_cube.data.shape[-2], 60)
    edgex = np.concatenate((xs, # bottom edge
                            np.full(len(ys), xs[-1]), # right edge
                            xs, # top edge
                            np.full(len(ys), xs[0]))) # left edge
    edgey = np.concatenate((np.full(len(xs),ys[0]), # bottom edge
                            ys, # right edge
                            np.full(len(xs), ys[-1]), # top edge
                            ys)) # left edge

    # Now we transform them to the output frame
    xs, ys = astropy.wcs.utils.pixel_to_pixel(celestial_source, celestial_target, edgex, edgey)

    if np.any(np.isnan(xs)) or np.any(np.isnan(ys)):
        # If the input data is far enough outside the output frame that its coordinates aren't defined in the output
        # projection, we'll get nans. In that case, fall back to reprojecting into the entire output frame. We'll get a
        # lot of nothing, but at least we won't crash.
        logger = get_run_logger()
        logger.warning(f"For {input_cube.meta['FILENAME']}, got NaNs when finding input image's extent in output frame")
        xmin, ymin = 0, 0
        ymax, xmax = output_shape
    else:
        # And we find that bounding box
        xmin, xmax = int(np.floor(xs.min())), int(np.ceil(xs.max()))
        ymin, ymax = int(np.floor(ys.min())), int(np.ceil(ys.max()))
        xmin = np.max((xmin, 0))
        ymin = np.max((ymin, 0))
        xmax = np.min((xmax, output_shape[1]))
        ymax = np.min((ymax, output_shape[0]))

    output_array = np.full((2, *output_shape), np.nan)

    # We will roll off the uncertainty by the inverse of the distance to the edge of the mask.
    # This allows pixels closer to the center to be weighted more than those on the edge.
    # Note. We add 1 to the distance to edge to avoid division by zero errors.
    image_mask = input_cube.data == 0
    distance_to_edge = distance_transform_edt(~image_mask, return_indices=False)
    input_data = np.stack([input_cube.data, input_cube.uncertainty.array / (distance_to_edge + 1)])

    # Reproject will complain if the input and output arrays have different dtypes
    input_data = np.asarray(input_data, dtype=float)

    reproject.reproject_adaptive(
        (input_data, celestial_source),
        celestial_target[ymin:ymax, xmin:xmax],
        shape_out=(2, np.max((ymax-ymin, 0)), np.max((xmax-xmin, 0))),
        roundtrip_coords=False, return_footprint=False,
        output_array=output_array[..., ymin:ymax, xmin:xmax],
        conserve_flux=True,
    )

    return output_array


@punch_flow
def reproject_many_flow(data: list[NDCube | None], trefoil_wcs: WCS, trefoil_shape: np.ndarray) -> list[NDCube | None]:
    """Reproject many flow."""
    out_layers = [reproject_cube.submit(d, trefoil_wcs, trefoil_shape) if d is not None else None for d in data]

    return [NDCube(data=out_layers[i].result()[0],
                   uncertainty=StdDevUncertainty(out_layers[i].result()[1]),
                   wcs=trefoil_wcs,
                   meta=d.meta) if d is not None else None for i, d in enumerate(data)]
