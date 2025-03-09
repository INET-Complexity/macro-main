"""Labour market time series tracking and management.

This module provides functionality for creating and managing time series data
for labour market metrics. It tracks various employment indicators and market
dynamics over time.

The time series include:
1. Employment Metrics:
   - Total employment
   - Employment by industry
   - New hires
   - Job separations

2. Market Dynamics:
   - Voluntary quits
   - Involuntary separations
   - Random terminations
   - Industry transitions

3. Market Analysis:
   - Employment trends
   - Turnover rates
   - Industry distribution
   - Market stability

The module integrates with the broader TimeSeries framework to provide
consistent data storage and retrieval across the simulation.
"""

import numpy as np

from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.timeseries import TimeSeries


def create_labour_market_timeseries(
    initial_individual_activity: np.ndarray,
    initial_individual_employment_industry: np.ndarray,
    n_industries: int,
) -> TimeSeries:
    """Create a new time series object for labour market tracking.

    This function initializes a TimeSeries object with various metrics
    for tracking labour market evolution. It sets up initial values
    and prepares containers for historical data.

    Args:
        initial_individual_activity: Initial employment status array
        initial_individual_employment_industry: Initial industry
            assignments array
        n_industries: Number of industries in the economy

    Returns:
        TimeSeries: Object containing:
            - num_employed_individuals_before_clearing: Pre-clearing
              employment count
            - num_individuals_newly_joining: New hires count
            - num_individuals_newly_randomly_fired: Random terminations
            - num_individuals_newly_randomly_quit: Voluntary quits
            - num_individuals_newly_fired: Involuntary separations
            - num_individuals_newly_leaving: Total separations
            - num_employed_individuals_by_sector: Industry-specific
              employment counts

    Note:
        The time series tracks both aggregate employment metrics and
        detailed breakdowns by industry and separation type.
    """
    num_employed = np.zeros(n_industries)
    for g in range(n_industries):
        num_employed[g] = np.sum(
            np.logical_and(
                initial_individual_employment_industry == g,
                initial_individual_activity == ActivityStatus.EMPLOYED,
            )
        )
    return TimeSeries(
        num_employed_individuals_before_clearing=[np.sum(initial_individual_activity == ActivityStatus.EMPLOYED)],
        num_individuals_newly_joining=[np.nan],
        num_individuals_newly_randomly_fired=[np.nan],
        num_individuals_newly_randomly_quit=[np.nan],
        num_individuals_newly_fired=[np.nan],
        num_individuals_newly_leaving=[np.nan],
        num_employed_individuals_by_sector=num_employed,
        #
    )
