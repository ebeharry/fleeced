from datetime import datetime


def _now():
    """
    Get the current local timestamp as a formatted string.

    :return: the current timestamp in "%Y-%m-%d %H:%M:%S" format
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
