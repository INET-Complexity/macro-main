"""Housing market time series tracking and management.

This module provides functionality for creating and managing time series data
for housing market metrics. It tracks various market indicators over time,
including transaction volumes, property values, and market composition.

The time series include:
1. Transaction Metrics:
   - Number of properties sold
   - Number of new rental agreements
   - Price-to-value ratios
   - Rent-to-value ratios

2. Property Values:
   - Current values
   - Value distributions
   - Historical trends
   - Market-wide statistics

3. Market Composition:
   - Number of rented properties
   - Number of owner-occupied properties
   - Number of vacant properties
   - Occupancy rates

The module integrates with the broader TimeSeries framework to provide
consistent data storage and retrieval across the simulation.
"""

import numpy as np
import pandas as pd

from macromodel.timeseries import TimeSeries
from macromodel.util.get_histogram import get_histogram


def create_housing_market_timeseries(
    data: pd.DataFrame,
    initial_observed_fraction_value_price: np.ndarray,
    initial_observed_fraction_rent_value: np.ndarray,
    scale: int,
) -> TimeSeries:
    """Create a new time series object for housing market tracking.

    This function initializes a TimeSeries object with various metrics
    for tracking housing market evolution. It sets up initial values
    and prepares containers for historical data.

    Args:
        data: DataFrame containing initial property data
        initial_observed_fraction_value_price: Initial price/value ratio
            coefficients from linear regression
        initial_observed_fraction_rent_value: Initial rent/value ratio
            coefficients from linear regression
        scale: Scale factor for histogram binning

    Returns:
        TimeSeries: Object containing:
            - total_number_of_bought_houses: Properties sold
            - total_number_of_newly_rented_houses: New rentals
            - observed_fraction_value_price: Price/value ratios
            - price_value_histogram: Distribution of price/value ratios
            - observed_fraction_rent_value: Rent/value ratios
            - rent_value_histogram: Distribution of rent/value ratios
            - property_values: Current property values
            - property_values_histogram: Distribution of values
            - total_number_of_houses_rented: Rented properties
            - total_number_of_houses_owner_occupied: Owner-occupied
            - total_number_of_houses_unoccupied: Vacant properties
            - households_hoping_to_move: Households hoping to move

    Note:
        The histograms provide distributions of various metrics,
        useful for analyzing market segmentation and trends.
    """
    return TimeSeries(
        total_number_of_bought_houses=[np.nan],
        total_number_of_newly_rented_houses=[np.nan],
        #
        observed_fraction_value_price=initial_observed_fraction_value_price,
        price_value_histogram=get_histogram(np.array([]), None),
        observed_fraction_rent_value=initial_observed_fraction_rent_value,
        rent_value_histogram=get_histogram(np.array([]), None),
        #
        property_values=data["Value"].values,
        property_values_histogram=get_histogram(data["Value"].values, scale),
        #
        total_number_of_houses_rented=[
            np.sum(
                np.logical_and(
                    data["Corresponding Inhabitant Household ID"] != -1,
                    data["Corresponding Inhabitant Household ID"] != data["Corresponding Owner Household ID"],
                )
            )
        ],
        total_number_of_houses_owner_occupied=[
            np.sum(data["Corresponding Inhabitant Household ID"] == data["Corresponding Owner Household ID"])
        ],
        total_number_of_houses_unoccupied=[np.sum(data["Corresponding Inhabitant Household ID"] == -1)]
    )
