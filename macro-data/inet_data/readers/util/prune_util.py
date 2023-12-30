import warnings
from datetime import datetime, date
from typing import Optional

import pandas as pd


class DataFilterWarning(Warning):
    pass


def prune_index(
    index: pd.Index,
    prune_date: date,
    dataset_name: Optional[str] = None,
    date_format: str = "%Y-%m-%d",
) -> list[str]:
    """
    Filters columns/index so datetime values are greater than or equal to the prune_date.

    Returns a list with
     1. All the values that cannot be parsed as a date,
     2. The values that can be parsed as a date and are greater than or equal to the date provided
    """

    # identify indices of columns that can be parsed as a date
    as_datetime = pd.to_datetime(index, errors="coerce")

    non_date_cols = list(index[as_datetime.isna()])
    selected_date_cols = list(index[as_datetime >= pd.to_datetime(prune_date)])

    return non_date_cols + selected_date_cols
