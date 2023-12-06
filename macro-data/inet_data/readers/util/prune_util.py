import warnings
from typing import Optional

import pandas as pd


class DataFilterWarning(Warning):
    pass


def filter_columns_by_date(columns: list[str], date: str, dataset_name: Optional[str] = None) -> list[str]:
    """
    Returns
     1. All the columns that cannot be parsed as a date,
     2. The columns that can be parsed as a date and are greater than or equal to the date provided
    """
    # Identify non-date columns
    non_date_cols = [col for col in columns if pd.to_datetime(col, errors="coerce") is pd.NaT]
    # Identify date columns that are greater than or equal to x
    date_cols_to_keep = [
        col for col in columns if pd.to_datetime(col, errors="coerce") is not pd.NaT and col >= f"{date}"
    ]
    if not date_cols_to_keep:
        warnings.warn(
            f"{dataset_name}: No columns were kept for date {date}.",
            DataFilterWarning,
        )
    return non_date_cols + date_cols_to_keep
