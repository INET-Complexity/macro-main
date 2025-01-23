from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from sklearn.linear_model import LinearRegression


class SyntheticGovernmentEntities(ABC):
    """
    Represents a collection of synthetic government entities. These entities are used to represent government consumption.

    The government entity data is stored in a pandas DataFrame with the following columns:
        - Consumption in LCU: The consumption of the government entity (in LCU).
        - Consumption in USD: The consumption of the government entity (in USD).

    Parameters:
    - country_name (str): The name of the country.
    - year (int): The year of the data.
    - number_of_entities (int): The number of government entities.
    - gov_entity_data (pd.DataFrame): The data for the government entities.
    - government_consumption_model (Optional[LinearRegression]): The consumption model for the government (a linear
    regression model to extrapolate government consumption growth).

    Attributes:
    - country_name (str): The name of the country.
    - year (int): The year of the data.
    - number_of_entities (int): The number of government entities.
    - gov_entity_data (pd.DataFrame): The data for the government entities.
    - government_consumption_model (Optional[LinearRegression]): The consumption model for the government.
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
        Returns the total emissions of the government entities.

        Returns:
            pd.Series: The total emissions of the government entities.
        """
        return self.gov_entity_data["Consumption Emissions"]
