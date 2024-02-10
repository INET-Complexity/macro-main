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
        total_number_of_houses_unoccupied=[np.sum(data["Corresponding Inhabitant Household ID"] == -1)],
    )
