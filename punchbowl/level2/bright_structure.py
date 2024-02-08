from typing import List
import typing as t
from prefect import get_run_logger, task
from skimage.morphology import binary_dilation
from punchbowl.data import PUNCHData
import numpy as np

# typehint formatting
def find_spikes(
                data: np.ndarray,
                uncertainty: np.ndarray,
                threshold: float = 10,
                required_yes: int = 6,
                veto_limit: int = 2,
                diff_method: str = 'abs',
                dilation: int = 0,
                index_of_interest: t.Optional[int] = None):
    """Module to identify bright structures 
    Given a time sequence of images, identify "spikes" that exceed a
    threshold in a single frame.

    Diffuse bright structures over1he background F-corona are 
    identified and marked in the data using in-band "bad value" marking
    (which is supported by the FITS standard). Data marking follows the 
    existing ZSPIKE temporal despiking algorithm to identify auroral 
    transients.

    This is a voting algorithm based on my ZSPIKE.PRO from the 1990s,
    which is available in the solarsoft distribution from Lockheed Martin.
    
    ZSPIKE was originally used to identify cosmic rays, and 
    was adopted on STEREO for on-board despiking during exposure accumulation.
    For this application ZSPIKE is ideal because it does not rely on the 
    spatial structure of spikes, only their temporal structure. Both cosmic 
    rays and, if present, high-altitude aurora are transient and easily 
    detected with ZSPIKE. 
    
    The algorithm assembles "votes" from the images 
    surrounding each one in a stream, to determine whether a particular pixel 
    is a good candidate for a temporal spike. If the pixel is sufficiently 
    bright compared to its neighbors in time, it is marked bad. "Bad values"
    are stored in the DRP for file quality marking outlined in Module Quality 
    Marking.

    There are two methods of identifying if a pixel is above a given threshold, 
    and therefore considered a spike. diff_method='abs' or 'sigma'.

    diff_method 'abs' represents the absolute difference, and is the default. 
    If set, this is an absolute difference, in DN, required for a pixel to 
    'vote' its central value.  If the central value is this much higher than a 
    given voting value, then the central value is voted to be a spike.  If 
    it's this much lower, the veto count is incremented.

    If diff_method 'sigma' is set if, then each pixel is treated as a
    time series and the calculated sigma (RMS variation from the mean) of 
    the timeseries is used to calculate a difference threshold at each 
    location.

    The threshold is the value over which a pixel is voted as a spike. The threshold 
    should be different depending diff_method.

    Parameters
    ----------
    data_object : np.ndarray
        data to operate on - this is expected to have an odd number of frames, or 
        a frame_of_interest should be inserted.

    uncertainty: np.ndarray
        data to operate on - this is expected to have an odd number of frames, or 
        a frame_of_interest should be inserted.

    index_of_interest : int
        if you have an even number of frames, or you don't want to find spikes
        in the center frame, the frame_of_interest will be used. index_of_interest
        starts from 0, therefore the center frame from 21 frames will be 10.

    threshold : float
        This is the threshold over which a pixel is voted as a spike.
        
    nvotes: int=3
        (default is 3) - number of 'voting' frames on either side of the
        central frame; actual number of votes is twice this.

    required_yes: int=4
        (default is 4) - number of 'voting' frames that must vote the central 
        value is a spike, for it to be marked as such.
        
    veto_limit: int=2
        (default is 2) - number of 'voting' frames that must vote NO to veto
        marking the central value as a spike.
        
    diff_method: str='abs'
        This is the method by which the threshold is set. 'abs' treats each pixel 
        independently and finds the absolute diffference, and 'sigma' treats each 
        corresponding pixel as a time series and the calculated RMS variation 
        from the mean of the timeseries is used to calculate a difference at each
        location.

    Returns
    -------
    np.ndarray
        a frame matching the dimensions of the frame of interest with True flagging
        bad pixels, and False flagging good data points

    TO DO/COMMENTS
    -----

    consider what happens with rotation, and gaps, need to extract same region. 
    Are the regions without data flagged as NaN


    """

    # when using PUNCHdata object here we extract uncertainty and data.
    #data=PUNCHdata.data
    #uncertainty=PUNCHdata.uncertainty

    # test if 3-D data cube is inserted
    if len(data.shape) != 3:
        raise ValueError("`data` must be a 3-D array")

    # test if odd number of frames, or if a frame of interest has been included
    z_shape=np.shape(data[:,0,0])
    if z_shape[0] % 2 == 0:
        if index_of_interest is None:
            raise ValueError("The number of frames in `data` is expected to be odd, with frame of interest the center. Add frame_of_interest to specify a center frame.") 
    else:
        if index_of_interest is None:
            index_of_interest = np.round(z_shape[0]/2)

    # extract frame of interest
    frame_of_interest = data[index_of_interest,:,:]
    frame_of_interest_uncertainty = uncertainty[index_of_interest,:,:]

    # extract background frames to perform voting
    voters_array = np.delete(data, index_of_interest, 0)
    voters_array_uncertainty = np.delete(uncertainty, index_of_interest, 0)
    
    # create a reference cube the same dimensions as the voter cube, filled with frame of interest
    ref_cube = np.stack([frame_of_interest for _ in range(voters_array.shape[0])], axis=0)

    # calculate abs difference between the reference cube and the voters
    difference_array = np.abs(ref_cube - voters_array)


    # calculate abs difference between the reference cube and the voters
    if diff_method == 'abs':

        # create a threshold array of the same dimensions as the voter array where every value is the threshold
        threshold_array = np.zeros_like(voters_array) + threshold

    # calculate sigma difference between the reference cube and the voters
    elif diff_method == 'sigma':
        threshold_array = threshold*np.nanstd(voters_array, axis=0)

    else:
        raise ValueError("input an appropriate string")

    # calculate yes votes
    yes_vote_array = np.sum(difference_array > threshold_array, axis=0)

    # calculate no votes
    no_vote_array = np.zeros_like(frame_of_interest) + voters_array.shape[0] - yes_vote_array #np.sum(difference <= threshold, axis=0) 

    # create flagged_features_array
    flagged_features_array = yes_vote_array > required_yes
        
    # if the number of no votes exceeds a veto limit, then veto
    veto_mask = no_vote_array > veto_limit
    
    # if veto'd create a False flag
    flagged_features_array[veto_mask] = False

    # if input data already has an uncertainty of 1, create a True flag
    flagged_features_array = np.where(frame_of_interest_uncertainty >= 1.0, True, flagged_features_array)
    #flagged_features_array = np.where((1-frame_of_interest_uncertainty), flagged_features_array, True)

    for _ in range(dilation):
        binary_dilation(flagged_features_array, out=flagged_features_array)
    
    return flagged_features_array


@task
def identify_bright_structures_task(data_list: List[PUNCHData]) -> List[PUNCHData]:
    """Prefect task to perform bright structure identification

    Parameters
    ----------
    data_object : PUNCHData
        data to operate on

    Returns
    -------
    PUNCHData
        modified version of the input data with the bright structures identified
    """
    logger = get_run_logger()
    logger.info("identify_bright_structures_task started")
    # TODO: actually do the identification
    logger.info("identify_bright_structures_task ended")
    for data_object in data_list:
        data_object.meta.history.add_now("LEVEL2-bright_structures",
                                         "bright structure identification completed")
    print("test")
    return data_list
