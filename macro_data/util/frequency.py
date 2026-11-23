from __future__ import annotations

from numbers import Real

import pandas as pd


def periods_per_year(time_unit: int) -> int:
    if time_unit <= 0 or 12 % time_unit != 0:
        raise ValueError("time_unit must be a positive divisor of 12.")
    return 12 // time_unit


def annual_to_period(
    value: Real | pd.Series | pd.DataFrame,
    time_unit: int,
    column: str | None = None,
):
    factor = periods_per_year(time_unit)

    if isinstance(value, pd.DataFrame):
        converted = value.copy()
        if column is None:
            return converted / factor
        converted[column] = converted[column].astype(float) / factor
        return converted

    if isinstance(value, pd.Series):
        return value.astype(float) / factor

    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value) / factor

    raise TypeError(f"Unsupported value type for annual-to-period conversion: {type(value)!r}")
