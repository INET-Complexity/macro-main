from typing import Optional

import pandas as pd
from sklearn.linear_model import LinearRegression

from macro_data.configuration.countries import Country
from macro_data.readers import DataReaders
from macro_data.readers.exogenous_data import ExogenousCountryData
from macro_data.util.regressions import fit_linear


class SyntheticGoodsMarket:
    def __init__(self, country_name: str | Country, exchange_rates_model: Optional[LinearRegression]):
        """
        Represents a synthetic goods market.

        Attributes:
            country_name (str): The name of the country.
            exchange_rates_model (Optional[LinearRegression]): The model for exchange rates (optional).
        """

        self.country_name = country_name
        self.exchange_rates_model = exchange_rates_model

    @classmethod
    def from_readers(
        cls,
        country_name: Country | str,
        year: int,
        quarter: int,
        readers: DataReaders,
        exogenous_data: ExogenousCountryData,
        max_timeframe: float = 40,
    ) -> "SyntheticGoodsMarket":
        rates = readers.exchange_rates.df.loc[country_name]
        inflation = exogenous_data.inflation["PPI Inflation"]
        growth = exogenous_data.national_accounts["Gross Output (Growth)"]

        # merge the three dataframes on index
        merged = pd.merge_asof(rates, inflation, left_index=True, right_index=True)
        merged = pd.merge_asof(merged, growth, left_index=True, right_index=True)

        # dropnans
        merged = merged.dropna()

        # select only data up to the current quarter
        merged = merged.loc[:f"{year}-Q{quarter}"]

        # select last max_timeframe rows
        merged = merged.iloc[-max_timeframe:]

        model = LinearRegression()

        fit_linear(
            model=model,
            dependent="Exchange Rates",
            independents=["PPI Inflation", "Gross Output (Growth)"],
            data=merged,
        )

        return cls(country_name, model)
