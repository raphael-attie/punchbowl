from typing import List

from prefect import get_run_logger, task

from punchbowl.data import PUNCHData


@task
def quality_flag_task(data_list: List[PUNCHData]) -> List[PUNCHData]:
    """Prefect task to perform quality flagging

    Parameters
    ----------
    data_object : PUNCHData
        data to operate on

    Returns
    -------
    PUNCHData
        modified version of the input data with the quality of pixels flagged
    """
    logger = get_run_logger()
    logger.info("quality_flag started")
    # TODO: actually do the quality flagging
    logger.info("quality_flag ended")
    for data_object in data_list:
        data_object.meta.history.add_now("LEVEL2-quality_flag", "quality flagging completed")
    return data_list
