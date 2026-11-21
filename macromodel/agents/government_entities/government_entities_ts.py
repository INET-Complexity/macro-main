"""Time series management for government entities.

This module handles time series data for government entities, tracking:
- Consumption patterns in multiple currencies
- Desired consumption targets
- Total consumption across entities
- Optional emissions from consumption

The time series provide historical tracking of:
- Actual vs desired consumption
- Currency conversions
- Aggregate spending patterns
- Environmental impact metrics
"""

from typing import Optional

import numpy as np
import pandas as pd

from macromodel.timeseries import TimeSeries


def create_government_entities_timeseries(
    data: pd.DataFrame,
    n_government_entities: int,
    add_emissions: bool = False,
    emission_factors_lcu: Optional[np.ndarray] = None,
    emitting_indices: Optional[np.ndarray] = None,
) -> TimeSeries:
    """Create time series for government entities.

    Initializes time series tracking for:
    - Consumption in USD and local currency
    - Desired consumption targets
    - Total consumption across entities
    - Optional emissions by type

    The time series include:
    - Actual consumption values
    - Consumption targets
    - Currency conversions
    - Environmental metrics

    Args:
        data (pd.DataFrame): Initial data containing consumption values
            in both USD and local currency
        n_government_entities (int): Number of government entities
        add_emissions (bool, optional): Whether to track emissions
        emission_factors_lcu (np.ndarray, optional): Emission factors
            per unit of consumption in local currency
        emitting_indices (np.ndarray, optional): Indices of goods
            that generate emissions

    Returns:
        TimeSeries: Initialized time series containing:
            - n_government_entities: Number of entities
            - consumption_in_usd: Consumption in USD
            - consumption_in_lcu: Consumption in local currency
            - total_consumption: Aggregate consumption
            - desired_consumption_in_usd: Target consumption in USD
            - desired_consumption_in_lcu: Target consumption in local currency
            - emissions: Optional total emissions
            - coal_emissions: Optional coal-related emissions
            - gas_emissions: Optional gas-related emissions
            - oil_emissions: Optional oil-related emissions
            - refined_products_emissions: Optional refined product emissions
    """
    if add_emissions:
        emissions = np.sum(data["Consumption in LCU"].values[emitting_indices] * emission_factors_lcu)
        emissions_dict = {
            "coal_emissions": np.sum(data["Consumption in LCU"].values[emitting_indices] * emission_factors_lcu[0]),
            "gas_emissions": np.sum(data["Consumption in LCU"].values[emitting_indices] * emission_factors_lcu[1]),
            "oil_emissions": np.sum(data["Consumption in LCU"].values[emitting_indices] * emission_factors_lcu[2]),
            "refined_products_emissions": np.sum(
                data["Consumption in LCU"].values[emitting_indices] * emission_factors_lcu[3]
            ),
        }
    else:
        emissions = None
        emissions_dict = {}

    return TimeSeries(
        n_government_entities=n_government_entities,
        consumption_in_usd=data["Consumption in USD"].values,
        consumption_in_lcu=data["Consumption in LCU"].values,
        total_consumption=[data["Consumption in LCU"].values.sum()],
        desired_consumption_in_usd=data["Consumption in USD"].values,
        desired_consumption_in_lcu=data["Consumption in LCU"].values,
        emissions=emissions,
        **emissions_dict,
    )
