import pandas as pd

from macro_data.processing.synthetic_housing_market.synthetic_housing_market import (
    SyntheticHousingMarket,
)


class DefaultSyntheticHousingMarket(SyntheticHousingMarket):
    @classmethod
    def init_from_datadict(cls, country_name: str, housing_data_dict: dict):
        housing_market_data = pd.DataFrame(housing_data_dict)
        housing_market_data["Newly on the Rental Market"] = False
        return cls(country_name, housing_market_data)
