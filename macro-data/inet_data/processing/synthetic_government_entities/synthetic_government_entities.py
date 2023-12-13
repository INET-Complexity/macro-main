from abc import ABC, abstractmethod
from typing import Optional

from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd


class SyntheticGovernmentEntities(ABC):
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
