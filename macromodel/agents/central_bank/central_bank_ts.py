"""Time series management for Central Bank agent.

This module handles the creation and management of time series data
for the central bank agent, including:
- Policy interest rates
- Monetary policy indicators
- Historical rate decisions

The time series provide historical tracking of:
- Interest rate path
- Policy implementation
- Monetary conditions
"""

import pandas as pd

from macromodel.timeseries import TimeSeries


def create_central_bank_timeseries(data: pd.DataFrame) -> TimeSeries:
    """Create time series objects for central bank variables.

    Initializes time series tracking for:
    - Policy interest rates
    - Monetary policy indicators
    - Historical rate decisions

    Args:
        data (pd.DataFrame): Initial central bank data including
            historical values for policy rates and other variables

    Returns:
        TimeSeries: Initialized time series containing central bank
            variables with their initial values
    """
    return TimeSeries(policy_rate=[data["policy_rate"].values[0]])
