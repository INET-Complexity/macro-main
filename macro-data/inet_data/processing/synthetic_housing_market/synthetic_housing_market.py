from abc import ABC, abstractmethod

import pandas as pd


class SyntheticHousingMarket(ABC):
    def __init__(
        self,
        country_name: str,
        year: int,
    ):
        self.country_name = country_name
        self.year = year

        # Housing market data
        self.housing_market_data = pd.DataFrame()

    @abstractmethod
    def create(self) -> None:
        pass

    @abstractmethod
    def set_initial_conditions(self) -> None:
        pass
