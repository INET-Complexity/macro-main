"""Module for preprocessing synthetic goods market data.

This module provides a framework for preprocessing and organizing goods market data
that will be used to initialize behavioral models. Key preprocessing includes:

1. Exchange Rate Processing:
   - Historical exchange rate data
   - Inflation rate relationships
   - Growth rate correlations
   - Rate prediction model estimation

2. Market Data Organization:
   - Buyer-seller relationships
   - Trade flow patterns
   - Price level initialization
   - Market clearing conditions

3. Parameter Processing:
   - Exchange rate model parameters
   - Price adjustment factors
   - Trade flow coefficients
   - Growth-inflation relationships

Note:
    This module is NOT used for simulating goods market behavior. It only handles
    the preprocessing and organization of goods market data that will later be used
    to initialize behavioral models in the simulation package. The actual market
    matching, price setting, and trade flow dynamics are implemented in the
    simulation package.
"""

from typing import Optional

import pandas as pd
from sklearn.linear_model import LinearRegression

from macro_data.configuration.countries import Country
from macro_data.configuration.region import Region
from macro_data.readers import DataReaders
from macro_data.readers.exogenous_data import ExogenousCountryData
from macro_data.util.regressions import fit_linear


class SyntheticGoodsMarket:
    """Container for preprocessed goods market data.

    This class organizes goods market data for initializing behavioral models. It
    processes and structures data about market relationships, trade patterns, and
    exchange rates. It does NOT implement any market behavior - it only handles
    data preprocessing.

    The preprocessing workflow includes:
    1. Exchange Rate Model:
       - Historical rate collection
       - Inflation data integration
       - Growth rate correlation
       - Model parameter estimation

    2. Market Structure:
       - Buyer identification
       - Seller categorization
       - Trade relationship mapping
       - Initial price levels

    3. Trade Flow Data:
       - Historical patterns
       - Volume relationships
       - Price dependencies
       - Growth correlations

    Note:
        This is a data container class. The actual goods market behavior
        (matching, price setting, trade flows, etc.) is implemented in the
        simulation package, which uses this preprocessed data for initialization.

    Attributes:
        country_name (str | Country): Country identifier for data collection
        exchange_rates_model (Optional[LinearRegression]): Preprocessed exchange
            rate model for initializing price dynamics. The model relates exchange
            rates to inflation and growth patterns.
    """

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
        country_name: Country | str | Region,
        year: int,
        quarter: int,
        readers: DataReaders,
        exogenous_data: ExogenousCountryData,
        max_timeframe: float = 40,
    ) -> "SyntheticGoodsMarket":
        """Create a preprocessed goods market data container from data sources.

        This method processes goods market data from various sources to prepare:
        1. Exchange rate relationships with inflation and growth
        2. Historical trade patterns and price levels
        3. Market structure initialization data

        The preprocessing steps:
        1. Collect historical exchange rates
        2. Match with inflation and growth data
        3. Clean and align time series
        4. Estimate exchange rate model parameters

        Args:
            country_name (Country | str): Country to process data for
            year (int): Base year for preprocessing
            quarter (int): Base quarter for preprocessing
            readers (DataReaders): Data source access
            exogenous_data (ExogenousCountryData): External economic data
            max_timeframe (float, optional): Maximum historical periods.
                Defaults to 40.

        Returns:
            SyntheticGoodsMarket: Container with preprocessed market data
        """
        rates_country = country_name
        if isinstance(country_name, Region):
            rates_country = country_name.parent_country
        rates = readers.exchange_rates.df.loc[rates_country].copy()
        inflation = exogenous_data.inflation["PPI Inflation"]
        growth = exogenous_data.national_accounts["Gross Output (Growth)"]

        rates.index = pd.to_datetime(rates.index, format="%Y")
        # merge the three dataframes on index
        merged = pd.merge_asof(rates, inflation, left_index=True, right_index=True)
        merged = pd.merge_asof(merged, growth, left_index=True, right_index=True)

        # dropnans
        merged = merged.dropna()

        # select only data up to the current quarter
        merged = merged.loc[:f"{year}-Q{quarter}"]
        # drop current quarter observation (ie last row)
        merged = merged.iloc[:-1]

        # select last max_timeframe rows
        merged = merged.iloc[-max_timeframe:]

        merged.columns = ["Exchange Rates", "PPI Inflation", "Growth"]

        model = LinearRegression()

        fit_linear(
            model=model,
            dependent="Exchange Rates",
            independents=["PPI Inflation", "Growth"],
            data=merged,
        )

        return cls(country_name, model)
