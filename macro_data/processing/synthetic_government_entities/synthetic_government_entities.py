"""
Module for preprocessing synthetic government entity data.

This module provides a framework for preprocessing and organizing data about government
entities that consume and invest in goods and services produced by the economy. Key
preprocessing tasks include:

1. Government Consumption Data:
   - Sectoral consumption patterns
   - Consumption in local currency and USD
   - Historical consumption growth trends
   - Consumption model parameter estimation

2. Entity Structure:
   - Number of government entities determination
   - Entity size distribution
   - Consumption allocation across entities
   - Entity-industry relationships

3. Environmental Impact:
   - Emissions from government consumption
   - Fuel-specific emission tracking
   - Environmental policy parameters

Note:
    This module is NOT used for simulating government behavior. It only handles
    the preprocessing and organization of government entity data that will later be
    used to initialize behavioral models in the simulation package.
"""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from sklearn.linear_model import LinearRegression


class SyntheticGovernmentEntities(ABC):
    """
    Container for preprocessed government entity data.

    This class organizes data about government entities that consume and invest in
    goods and services produced in the economy. It processes and structures data
    about consumption patterns, entity relationships, and environmental impact.
    It does NOT implement any government behavior - it only handles data preprocessing.

    The preprocessing workflow includes:
    1. Consumption Data Processing:
       - Historical consumption patterns by sector
       - Currency conversion and normalization
       - Growth trend analysis
       - Consumption model estimation

    2. Entity Structure Organization:
       - Entity count determination based on economic size
       - Consumption allocation across entities
       - Industry relationship mapping
       - Size distribution calculation

    3. Environmental Impact Processing:
       - Emissions from consumption activities
       - Fuel-type specific tracking
       - Environmental policy parameters
       - Impact distribution across entities

    The government entity data is stored in a pandas DataFrame with columns:
        - Consumption in LCU: Entity consumption in local currency
        - Consumption in USD: Entity consumption in US dollars
        - Consumption Emissions: CO2 emissions from consumption (if tracked)
        - {Fuel} Consumption Emissions: Fuel-specific emissions (if tracked)

    Attributes:
        country_name (str): Country identifier for data collection
        year (int): Base year for data preprocessing
        number_of_entities (int): Number of government entities
        gov_entity_data (pd.DataFrame): Preprocessed entity data
        government_consumption_model (Optional[LinearRegression]): Model for
            projecting consumption growth patterns
    """

    @abstractmethod
    def __init__(
        self,
        country_name: str,
        year: int,
        number_of_entities: int,
        gov_entity_data: pd.DataFrame,
        government_consumption_model: Optional[LinearRegression],
    ):
        self.country_name = country_name
        self.year = year

        # Government entity data
        self.number_of_entities = number_of_entities
        self.gov_entity_data = gov_entity_data

        # Consumption model
        self.government_consumption_model = government_consumption_model

    @property
    def total_emissions(self):
        """
        Get total CO2 emissions from government consumption activities.

        This property calculates the total emissions across all government entities
        based on their consumption patterns and the associated emission factors.

        Returns:
            pd.Series: Total emissions by government entity, with index matching
                the entity identifiers in gov_entity_data.
        """
        return self.gov_entity_data["Consumption Emissions"]
