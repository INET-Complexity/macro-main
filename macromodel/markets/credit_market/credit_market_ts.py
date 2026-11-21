"""Credit market time series tracking module.

This module provides functionality for creating and managing time series data for the credit market.
It tracks various loan-related metrics including:
- Newly granted loans (short-term, long-term, consumption, mortgages)
- Outstanding loan balances by type
- Historical loan performance and trends

The time series data is essential for:
1. Monitoring credit market health
2. Analyzing lending patterns
3. Tracking loan performance over time
4. Supporting policy decisions
"""

import numpy as np

from macromodel.timeseries import TimeSeries


def create_credit_market_timeseries(
    total_short_term_loans: float,
    total_long_term_loans: float,
    total_consumption_expansion_loans: float,
    total_mortgage_loans: float,
) -> TimeSeries:
    """Create a new credit market time series object.

    Initializes a TimeSeries object to track various credit market metrics over time.
    The time series tracks both new loan originations and outstanding loan balances
    across different loan types.

    Args:
        total_short_term_loans (float): Initial total of outstanding short-term loans
        total_long_term_loans (float): Initial total of outstanding long-term loans
        total_consumption_expansion_loans (float): Initial total of outstanding consumer loans
        total_mortgage_loans (float): Initial total of outstanding mortgage loans

    Returns:
        TimeSeries: Initialized time series object with the following metrics:
            - Newly granted loans by type (initialized as NaN)
            - Outstanding loan balances by type (initialized with provided values)

    Example:
        >>> ts = create_credit_market_timeseries(
        ...     total_short_term_loans=1000000,
        ...     total_long_term_loans=5000000,
        ...     total_consumption_expansion_loans=2000000,
        ...     total_mortgage_loans=10000000
        ... )
        >>> ts.total_outstanding_loans_granted_mortgages[0]
        10000000
    """
    return TimeSeries(
        total_newly_loans_granted_firms_short_term=[np.nan],
        total_newly_loans_granted_firms_long_term=[np.nan],
        total_newly_loans_granted_households_consumption=[np.nan],
        total_newly_loans_granted_mortgages=[np.nan],
        #
        total_outstanding_loans_granted_firms_short_term=[total_short_term_loans],
        total_outstanding_loans_granted_firms_long_term=[total_long_term_loans],
        total_outstanding_loans_granted_households_consumption=[total_consumption_expansion_loans],
        total_outstanding_loans_granted_mortgages=[total_mortgage_loans],
    )
