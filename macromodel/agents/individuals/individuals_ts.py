"""Individual time series data management.

This module handles the creation and tracking of time series data for individuals
in the macroeconomic model. It captures:
- Income streams and distributions
- Labor market participation
- Wage expectations
- Unemployment benefits

The time series track both individual-level data and aggregate distributions
to analyze:
- Income inequality
- Labor market dynamics
- Social welfare outcomes
- Economic mobility
"""

import pandas as pd

from macromodel.timeseries import TimeSeries
from macromodel.util.get_histogram import get_histogram


def create_individuals_timeseries(data: pd.DataFrame, scale: int) -> TimeSeries:
    """Create time series tracking for individual economic variables.

    This function initializes time series objects to track individual-level
    economic outcomes and their distributions over time. It captures:
    - Employment income
    - Unemployment benefits
    - Total income
    - Labor market participation
    - Wage expectations

    The function creates both individual-level series and histograms
    for distributional analysis.

    Args:
        data (pd.DataFrame): Individual-level data containing economic variables
        scale (int): Scale factor for histogram binning

    Returns:
        TimeSeries: Time series object containing:
            - n_individuals: Total number of individuals
            - employee_income: Employment-based income by individual
            - employee_income_histogram: Distribution of employment income
            - income_from_unemployment_benefits: Unemployment benefits by individual
            - income: Total income by individual
            - expected_income: Expected future income by individual
            - income_histogram: Distribution of total income
            - labour_inputs: Labor market participation by individual
            - reservation_wages: Minimum acceptable wages by individual
    """
    return TimeSeries(
        n_individuals=len(data),
        #
        employee_income=data["Employee Income"].values,
        employee_income_histogram=get_histogram(data["Employee Income"].values, scale),
        income_from_unemployment_benefits=data["Income from Unemployment Benefits"].values,
        income=data["Income"].values,
        expected_income=data["Income"].values,
        income_histogram=get_histogram(data["Income"].values, scale),
        #
        labour_inputs=data["Labour Inputs"].values,
        reservation_wages=data["Employee Income"].values + data["Income from Unemployment Benefits"].values,
    )
