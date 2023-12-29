from abc import ABC, abstractmethod
from typing import Optional

from sklearn.linear_model import LinearRegression

import pandas as pd


class SyntheticGovernmentEntities(ABC):
    """
    Represents a collection of synthetic government entities. These entities are used to represent government consumption.

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
