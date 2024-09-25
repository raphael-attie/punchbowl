from datetime import datetime

def generate_datetime_list(start_datetime: datetime | None = None,
                           end_datetime: datetime | None = None,
                           cadence: int = 60) -> list:
    """
    Create a list of times for searching the database for files.

    Creates a list of files based between a start date/time (start_datetime)
    and an end date/time (end_datetime) for a specified polarizer and
    PUNCH_product. The start and end times can both be input explicitly,
    individually, or derived from a mid time.

    Parameters
    ----------
    start_datetime : datetime, optional
        start date for file list, if not provided it will be derived.

    end_datetime : datetime, optional
        end date for file list, if not provided it will be derived.

    cadence : int
        time interval in minutes for generating datetime list.

    Returns
    -------
    datetime_list : list
        list of datetime objects over the specified period based on cadence.

    """
    # Create a list to hold the datetime objects
    datetime_list = []

    # Initialize the current time with the start time
    current_time = start_datetime

    # Generate datetime objects from start to end time with the given frequency
    while current_time <= end_datetime:
        datetime_list.append(current_time)
        current_time += datetime.timedelta(minutes=cadence)

    return datetime_list
