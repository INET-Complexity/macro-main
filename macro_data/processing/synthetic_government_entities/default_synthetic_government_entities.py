"""Module for preprocessing default synthetic government entity data.

This module provides a default implementation for preprocessing government entity data
using standard data sources such as OECD and national accounts. It handles the
organization of government consumption and investment data using commonly available
economic statistics.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from macro_data.configuration.countries import Country
from macro_data.processing.synthetic_government_entities.synthetic_government_entities import (
    SyntheticGovernmentEntities,
)
from macro_data.readers.default_readers import DataReaders
from macro_data.readers.emissions.emissions_reader import EmissionsData
from macro_data.readers.exogenous_data import ExogenousCountryData


class DefaultSyntheticGovernmentEntities(SyntheticGovernmentEntities):
    """Default implementation for preprocessing government entity data.

    This class provides a standard implementation for processing government entity
    data using common data sources like OECD statistics and national accounts. It
    organizes data about government consumption, investment, and environmental
    impact using widely available economic indicators.

    The preprocessing workflow includes:
    1. Data Collection:
       - OECD business demography statistics
       - National accounts government consumption data
       - Industry-level consumption patterns
       - Environmental impact factors

    2. Entity Structure:
       - Entity count calculation based on economic size
       - Optional single-entity consolidation
       - Consumption allocation by industry
       - Size-based distribution

    3. Growth Model:
       - Historical consumption pattern analysis
       - Growth rate calculation
       - Linear regression model estimation
       - Future consumption projection parameters

    4. Environmental Impact:
       - Fuel-specific consumption tracking
       - Emission factor application
       - Industry-specific impact calculation
       - Total emissions aggregation

    Args:
        country_name (Country): Country identifier for data collection
        year (int): Base year for preprocessing
        number_of_entities (int): Number of government entities
        gov_entity_data (pd.DataFrame): Preprocessed entity data
        government_consumption_model (Optional[LinearRegression]): Model for
            projecting consumption growth patterns

    Note:
        This implementation uses default data sources and standard preprocessing
        methods. For specialized preprocessing needs, create a new implementation
        of the base SyntheticGovernmentEntities class.
    """

    def __init__(
        self,
        country_name: Country,
        year: int,
        number_of_entities: int,
        gov_entity_data: pd.DataFrame,
        government_consumption_model: Optional[LinearRegression] = None,
    ):
        super().__init__(
            country_name,
            year,
            number_of_entities,
            gov_entity_data,
            government_consumption_model,
        )

    @classmethod
    def from_readers(
        cls,
        readers: DataReaders,
        country_name: Country,
        year: int,
        quarter: int,
        exogenous_country_data: ExogenousCountryData,
        industry_data: dict[str, pd.DataFrame],
        single_government_entity: bool,
        create_model: bool = False,
        emission_factors: Optional[EmissionsData] = None,
    ):
        """Create preprocessed government entity data from standard data sources.

        This method processes government entity data using default data sources:
        1. Extracts historical consumption from national accounts
        2. Calculates growth rates and estimates model parameters
        3. Determines entity count based on economic indicators
        4. Processes industry-specific consumption patterns
        5. Applies emission factors if environmental tracking is enabled

        Args:
            readers (DataReaders): Access to standard data sources
            country_name (Country): Country to process data for
            year (int): Base year for preprocessing
            quarter (int): Base quarter for preprocessing
            exogenous_country_data (ExogenousCountryData): External economic data
            industry_data (dict[str, pd.DataFrame]): Industry-level statistics
            single_government_entity (bool): Whether to consolidate into one entity
            create_model (bool): Whether to estimate growth model. Defaults to False
            emission_factors (Optional[EmissionsData]): Environmental impact data

        Returns:
            DefaultSyntheticGovernmentEntities: Container with preprocessed data
        """
        if exogenous_country_data:
            total_gov_consumption = exogenous_country_data.national_accounts["Real Government Consumption (Value)"]
            total_gov_consumption = total_gov_consumption.loc[total_gov_consumption.index < f"{year}-Q{quarter}"]
            growth_series = 1 + total_gov_consumption.pct_change()
            total_gov_consumption_growth = growth_series.values
            if growth_series.dropna().shape[0] > 0:
                create_model = True
        else:
            total_gov_consumption_growth = None

        govt_consumption_in_usd = industry_data["industry_vectors"]["Government Consumption in USD"].values
        govt_consumption_in_lcu = industry_data["industry_vectors"]["Government Consumption in LCU"].values
        total_va_lcu = industry_data["industry_vectors"]["Value Added in LCU"].sum()
        total_number_of_firms = int(
            readers.oecd_econ.read_business_demography(
                country=country_name,
                output=pd.Series(industry_data["industry_vectors"]["Output in USD"].values),
                year=year,
            ).sum()
        )

        n_entities = int(
            max(
                1,
                total_number_of_firms * govt_consumption_in_lcu.sum() / total_va_lcu,
            )
        )
        n_entities = 1 if single_government_entity else n_entities

        gov_entity_data = pd.DataFrame(
            {
                "Consumption in USD": govt_consumption_in_usd,
                "Consumption in LCU": govt_consumption_in_lcu,
            }
        )

        if create_model:
            government_consumption_model = LinearRegression().fit(
                [[0], [1]], [np.nanmean(total_gov_consumption_growth), np.nanmean(total_gov_consumption_growth)]
            )
        else:
            government_consumption_model = None

        if emission_factors is not None:
            array = emission_factors.emissions_array
            emitting_consumption = industry_data["industry_vectors"]["Government Consumption in LCU"].loc[
                ["B05a", "B05b", "B05c", "C19"]
            ]
            emissions = emitting_consumption.values @ array
            gov_entity_data["Consumption Emissions"] = emissions
            for i, name in enumerate(["Coal", "Gas", "Oil", "Refined Products"]):
                gov_entity_data[f"{name} Consumption Emissions"] = emitting_consumption.values[i] * array[i]

        return cls(
            country_name,
            year,
            n_entities,
            gov_entity_data,
            government_consumption_model,
        )
