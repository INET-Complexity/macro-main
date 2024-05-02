from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from macromodel.configurations import ExchangeRatesConfiguration


ModelDict = dict[str, LinearRegression]


class ExchangeRates:
    def __init__(
        self,
        exchange_rate_type: str,
        initial_year: int,
        country_names: list[str],
        historic_exchange_rate_data: pd.DataFrame,
        exchange_rates_model: Optional[ModelDict] = None,
    ):
        self.exchange_rate_type = exchange_rate_type
        self.initial_year = initial_year
        self.country_names = country_names
        self.historic_exchange_rate_data = historic_exchange_rate_data
        self.exchange_rates_model = exchange_rates_model

    @classmethod
    def from_data(
        cls,
        exchange_rates_data: pd.DataFrame,
        exchange_rate_config: ExchangeRatesConfiguration,
        initial_year: int,
        country_names: list[str],
        exchange_rates_model: Optional[ModelDict] = None,
    ):
        return cls(
            exchange_rate_type=exchange_rate_config.exchange_rate_type,
            initial_year=initial_year,
            country_names=country_names,
            historic_exchange_rate_data=exchange_rates_data,
            exchange_rates_model=exchange_rates_model,
        )

    def reset(
        self,
    ): ...

    def get_current_exchange_rates_from_usd_to_lcu(
        self,
        country_name: str,
        current_year: int,
        prev_inflation: float,
        prev_growth: float,
    ) -> list[float]:
        match self.exchange_rate_type:
            case "constant":
                return self.historic_exchange_rate_data.loc[country_name, str(self.initial_year)]
            case "exogenous":
                return self.historic_exchange_rate_data.loc[country_name, str(current_year)]
            case "model":
                if self.exchange_rates_model[country_name] is None:
                    raise ValueError("Exchange rates model is not provided")
                return self.exchange_rates_model[country_name].predict(np.array([prev_inflation, prev_growth]))
