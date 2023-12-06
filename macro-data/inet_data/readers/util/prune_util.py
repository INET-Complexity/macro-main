import warnings
from datetime import datetime
from typing import Optional

import pandas as pd


class DataFilterWarning(Warning):
    pass


def filter_columns_by_date(
    columns: pd.Index, date: str | datetime | int, dataset_name: Optional[str] = None, date_format: str = "%Y-%m-%d"
) -> list[str]:
    """
    Returns
     1. All the columns that cannot be parsed as a date,
     2. The columns that can be parsed as a date and are greater than or equal to the date provided
    """

    # if date is str, convert to datetime
    if isinstance(date, str):
        date = pd.to_datetime(date, format="%Y-%m-%d")
    elif isinstance(date, int):
        date = pd.to_datetime(str(date), format="%Y")

    # identify indices of columns that can be parsed as a date
    as_datetime = pd.to_datetime(columns, errors="coerce")

    non_date_cols = list(columns[as_datetime.isna()])
    selected_date_cols = list(columns[as_datetime >= date])

    return non_date_cols + selected_date_cols
