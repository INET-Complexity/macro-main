"""Time series management for goods market metrics.

This module provides functionality for creating and managing time series data
for the goods market. It tracks key market metrics including total supply
and demand across industries over time, enabling analysis of market dynamics
and equilibrium conditions.
"""

import numpy as np

from macromodel.timeseries import TimeSeries


def create_goods_market_timeseries(n_industries: int) -> TimeSeries:
    """Initialize time series for tracking goods market metrics.

    Creates a new TimeSeries object to track industry-level supply and demand
    over time. The time series maintains separate arrays for total supply
    and total demand across all industries.

    Args:
        n_industries (int): Number of industries in the economy

    Returns:
        TimeSeries: Initialized time series object with zero-filled arrays for:
            - total_industry_supply: Total supply by industry
            - total_industry_demand: Total demand by industry
    """
    return TimeSeries(
        total_industry_supply=np.zeros(n_industries),
        total_industry_demand=np.zeros(n_industries),
    )
