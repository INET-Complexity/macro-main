from abc import ABC, abstractmethod

import pandas as pd


class SyntheticHousingMarket(ABC):
    def __init__(
        self,
        country_name: str,
        year: int,
        housing_market_data: pd.DataFrame,
    ):
        self.country_name = country_name
        self.year = year

        # Housing market data
        self.housing_market_data = housing_market_data
