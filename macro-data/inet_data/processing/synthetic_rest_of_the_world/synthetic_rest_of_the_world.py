from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from sklearn.linear_model import LinearRegression


class SyntheticRestOfTheWorld(ABC):
    @abstractmethod
    def __init__(
        self,
        year: int,
        row_data: pd.DataFrame,
        exports_model: Optional[LinearRegression],
        imports_model: Optional[LinearRegression],
    ):
        self.country_name = "ROW"
        self.year = year

        # Rest of the World data
        self.row_data = row_data

        # Models
        self.exports_model = exports_model
        self.imports_model = imports_model
