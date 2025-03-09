"""Exchange rate determination and management implementation.

This module provides mechanisms for managing exchange rates between different
currencies in the multi-country macroeconomic model. It supports multiple
exchange rate regimes:

1. Constant: Fixed exchange rates throughout simulation
2. Exogenous: Externally provided exchange rate paths
3. Model-based: Rates determined by economic fundamentals

The implementation handles bilateral rates relative to USD as the base
currency, with conversion to local currency units (LCU) based on the
chosen regime and economic conditions.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from macromodel.configurations import ExchangeRatesConfiguration

ModelDict = dict[str, LinearRegression]


class ExchangeRates:
    """Exchange rate management system.

    This class handles exchange rate determination and conversion between
    currencies based on different exchange rate regimes. It supports:
    - Constant exchange rates
    - Exogenous rate paths
    - Model-based rates using economic fundamentals

    Exchange rates are managed relative to USD as the base currency, with
    conversions to local currency units (LCU) based on the specified regime
    and current economic conditions.

    Attributes:
        exchange_rate_type (str): Type of exchange rate regime
        initial_year (int): Starting year for exchange rate data
        country_names (list[str]): List of countries in the model
        historic_exchange_rate_data (pd.DataFrame): Historical exchange rates
        exchange_rates_model (Optional[ModelDict]): Models for rate prediction
    """

    def __init__(
        self,
        exchange_rate_type: str,
        initial_year: int,
        country_names: list[str],
        historic_exchange_rate_data: pd.DataFrame,
        exchange_rates_model: Optional[ModelDict] = None,
    ):
        """Initialize exchange rate system.

        Args:
            exchange_rate_type (str): Type of exchange rate regime
            initial_year (int): Starting year for exchange rate data
            country_names (list[str]): List of countries in the model
            historic_exchange_rate_data (pd.DataFrame): Historical exchange rates
            exchange_rates_model (Optional[ModelDict], optional): Rate prediction models.
                Defaults to None.
        """
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
        """Create exchange rate system from data and configuration.

        Factory method that constructs an ExchangeRates instance using
        provided data and configuration settings.

        Args:
            exchange_rates_data (pd.DataFrame): Historical exchange rate data
            exchange_rate_config (ExchangeRatesConfiguration): Configuration settings
            initial_year (int): Starting year for exchange rate data
            country_names (list[str]): List of countries in the model
            exchange_rates_model (Optional[ModelDict], optional): Rate prediction models.
                Defaults to None.

        Returns:
            ExchangeRates: Configured exchange rate system
        """
        return cls(
            exchange_rate_type=exchange_rate_config.exchange_rate_type,
            initial_year=initial_year,
            country_names=country_names,
            historic_exchange_rate_data=exchange_rates_data,
            exchange_rates_model=exchange_rates_model,
        )

    def reset(self) -> None:
        """Reset exchange rate system state.

        Currently a placeholder for potential future reset functionality.
        """
        ...

    def get_current_exchange_rates_from_usd_to_lcu(
        self,
        country_name: str,
        current_year: int,
        prev_inflation: float,
        prev_growth: float,
    ) -> float:
        """Get current USD to LCU exchange rate for a country.

        Computes the exchange rate from USD to local currency units (LCU)
        based on the specified exchange rate regime:
        - Constant: Returns initial year's rate
        - Exogenous: Returns rate from historical data
        - Model: Predicts rate using economic fundamentals

        Args:
            country_name (str): Target country name
            current_year (int): Current simulation year
            prev_inflation (float): Previous period's inflation rate
            prev_growth (float): Previous period's growth rate

        Returns:
            float: Exchange rate from USD to LCU

        Raises:
            ValueError: If exchange rate type is unknown or model is missing
        """
        match self.exchange_rate_type:
            case "constant":
                return self.historic_exchange_rate_data.loc[country_name, str(self.initial_year)]
            case "exogenous":
                return self.historic_exchange_rate_data.loc[country_name, str(current_year)]
            case "model":
                if self.exchange_rates_model[country_name] is None:
                    raise ValueError("Exchange rates model is not provided")
                return self.exchange_rates_model[country_name].predict(np.array([prev_inflation, prev_growth]))
            case _:
                raise ValueError("Unknown Exchange Rates Type")
