"""Rest of the World time series module.

This module provides functionality for creating and managing time series data
for the Rest of the World (ROW) component of the model. It tracks:

1. Trade Flows:
   - Real exports and imports
   - Desired trade volumes
   - Total trade aggregates

2. Price Series:
   - USD and LCU prices
   - Offered prices
   - Price adjustments

The time series maintain historical data for both actual and desired
quantities, facilitating analysis of trade patterns and price dynamics.
"""

import numpy as np
import pandas as pd

from macromodel.timeseries import TimeSeries


def create_rest_of_the_world_timeseries(
    data: pd.DataFrame,
    n_industries: int,
) -> TimeSeries:
    """Create time series for Rest of the World data.

    Initializes a TimeSeries object with trade and price data for the
    Rest of the World component. Includes both real and nominal values,
    actual and desired quantities, and prices in different currencies.

    Args:
        data (pd.DataFrame): Initial ROW data including exports, imports, and prices
        n_industries (int): Number of industrial sectors

    Returns:
        TimeSeries: Initialized time series with ROW data including:
            - exports_real: Real export volumes
            - total_exports: Aggregate export values
            - desired_exports_real: Target export volumes
            - imports_in_usd/lcu: Import values in USD and local currency
            - total_imports: Aggregate import values
            - desired_imports: Target import volumes
            - price_in_usd/lcu: Prices in USD and local currency
            - price_offered: Current offered prices
    """
    return TimeSeries(
        exports_real=data["Exports"].values,
        total_exports=[data["Exports"].values.sum()],
        desired_exports_real=data["Exports"].values,
        #
        imports_in_usd=data["Imports in USD"].values,
        imports_in_lcu=data["Imports in LCU"].values,
        total_imports=[data["Imports in USD"].values.sum()],
        desired_imports_in_usd=data["Imports in USD"].values,
        desired_imports_in_lcu=data["Imports in LCU"].values,
        #
        price_in_usd=data["Price in USD"].values,
        price_in_lcu=data["Price in LCU"].values,
        price_offered=np.full(n_industries, 1.0),
    )
