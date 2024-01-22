import numpy as np
from prefect import get_run_logger, task
from scipy.signal import convolve2d, medfilt2d

from punchbowl.data import PUNCHData
from punchbowl.level1.deficient_pixel import cell_neighbors


def radial_array(shape, center=None):
    if len(shape) != 2:
        raise ValueError(f"Shape must be 2D, received {shape} with {len(shape)} dimensions")

    x = np.arange(shape[0])
    y = np.arange(shape[1])
    X, Y = np.meshgrid(x, y)

    center = [s//2 for s in shape] if center is None else center
    distance = np.floor(np.sqrt(np.square(X - center[0]) + np.square(Y - center[1])))

    return distance


def spikejones(image: np.ndarray,
               unsharp_size: int = 3,
               method: str = 'convolve',
               alpha: float = 1,
               dilation: int = 0) -> np.ndarray:
    kernel_size = unsharp_size * 2 + 1
    smoothing_size = kernel_size * 2 + 1
    smoothing_kernel = radial_array((smoothing_size, smoothing_size)) <= (smoothing_size / 2)

    if method == "median":
        normalized_image = image / medfilt2d(image, kernel_size=smoothing_size)
        # unsharp_kernel = radial_array((kernel_size, kernel_size)) <= (kernel_size / 2)
        unsharped_image = normalized_image - medfilt2d(normalized_image, kernel_size=kernel_size) > alpha
    elif method == "convolve":
        normalized_image = convolve2d(image, smoothing_kernel / np.sum(smoothing_kernel))[2*smoothing_size+1:-(2*smoothing_size+1)]
        unsharp_kernel = np.ones((kernel_size, kernel_size)) / kernel_size / kernel_size
        unsharp_kernel[kernel_size//2, kernel_size//2] += 0.75
        unsharp_kernel[kernel_size//2 - 1: kernel_size//2 +1, kernel_size//2 - 1: kernel_size//2 + 2] += 0.25 / 9
        unsharped_image = convolve2d(normalized_image, unsharp_kernel, mode='same') > alpha
    else:
        raise NotImplementedError(f"Unsupported method. Method must be 'median' or 'convolve' but received {method}")

    if dilation != 0:
        dilation_size = 2 * dilation + 1
        unsharped_image = convolve2d(unsharped_image, np.ones(dilation_size, dilation_size), mode='same') != 0

    spikes = np.where(unsharped_image != 0)
    output = np.copy(image)
    image[spikes] = np.nan
    for x, y in zip(*spikes):
        output[x, y] = np.nanmean(cell_neighbors(image, x, y, kernel_size-1))

    return output



@task
def despike_task(data_object: PUNCHData) -> PUNCHData:
    """Prefect task to perform despiking

    Parameters
    ----------
    data_object : PUNCHData
        data to operate on

    Returns
    -------
    PUNCHData
        a modified version of the input with spikes removed
    """
    logger = get_run_logger()
    logger.info("despike started")
    # TODO: do despiking in here
    logger.info("despike finished")
    data_object.meta.history.add_now("LEVEL1-despike", "image despiked")
    return data_object
