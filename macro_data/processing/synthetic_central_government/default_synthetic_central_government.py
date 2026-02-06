"""Module for preprocessing synthetic central government data.

This module provides a concrete implementation for preprocessing central government data
that will be used to initialize behavioral models. Key preprocessing includes:

1. Data Collection and Processing:
   - Historical benefits data preparation
   - Tax revenue data organization
   - Initial state calculations

2. Parameter Estimation:
   - Benefits model estimation
   - Tax revenue estimation
   - Data validation and consistency checks

3. Data Organization:
   - Standard data preprocessing
   - Exogenous data preprocessing
   - Historical data processing

Note:
    This module is NOT used for simulating government behavior. It preprocesses
    data that will be used to initialize behavioral models in the simulation package.
    The actual government decisions and operations are implemented elsewhere.
"""

from typing import Optional

import pandas as pd
from sklearn.linear_model import LinearRegression

from macro_data.configuration.countries import Country
from macro_data.processing.synthetic_central_government.synthetic_central_government import (
    SyntheticCentralGovernment,
)
from macro_data.readers.default_readers import DataReaders


class DefaultSyntheticCGovernment(SyntheticCentralGovernment):
    """Default implementation for preprocessing central government data.

    This class preprocesses and organizes central government data by collecting historical
    data and estimating parameters. These will be used to initialize behavioral models,
    but this class does NOT implement any behavioral logic.

    The class provides two preprocessing paths:
    1. Standard preprocessing using historical data
    2. Exogenous data preprocessing when available

    The preprocessing includes:
    - Benefits data organization
    - Tax revenue parameter estimation
    - Initial state calculations
    - Model parameter estimation

    Note:
        This is a data container class. The actual government behavior (spending,
        tax policy, etc.) is implemented in the simulation package, which uses
        these preprocessed parameters.

    Attributes:
        country_name (str): Country identifier for data collection
        year (int): Reference year for preprocessing
        central_gov_data (pd.DataFrame): Preprocessed government data
        other_benefits_model (Optional[LinearRegression]): Model for estimating other benefits
        unemployment_benefits_model (Optional[LinearRegression]): Model for estimating unemployment benefits
    """

    def __init__(
        self,
        country_name: str,
        year: int,
        central_gov_data: pd.DataFrame,
        other_benefits_model: Optional[LinearRegression],
        unemployment_benefits_model: Optional[LinearRegression],
    ):
        """Initialize the central government data container.

        Args:
            country_name (str): Country identifier for data collection
            year (int): Reference year for preprocessing
            central_gov_data (pd.DataFrame): Initial data to preprocess
            other_benefits_model (Optional[LinearRegression]): Model for estimating other benefits
            unemployment_benefits_model (Optional[LinearRegression]): Model for estimating unemployment benefits
        """
        super().__init__(
            country_name,
            year,
            central_gov_data,
            other_benefits_model,
            unemployment_benefits_model,
        )

    @classmethod
    def from_readers(
        cls,
        readers: DataReaders,
        country_name: Country,
        year: int,
        year_range: int = 10,
        regression_window: int = 48,
        equity_injection: float = 0.0,
    ) -> SyntheticCentralGovernment:
        """Create a preprocessed central government data container using standard data sources.

        This method preprocesses data using historical sources to prepare:
        1. Benefits data (unemployment and other benefits)
        2. Initial debt levels
        3. Parameter estimates for benefits models

        The preprocessing steps:
        1. Check for exogenous data availability
        2. If available, estimate benefits models using historical data
        3. If not available, use direct historical values
        4. Prepare initial state data

        Args:
            readers (DataReaders): Data source readers
            country_name (Country): Country to preprocess data for
            year (int): Reference year for preprocessing
            year_range (int, optional): Years of historical data to use. Defaults to 10.
            regression_window (int, optional): Months of data for model estimation. Defaults to 48.
            equity_injection (float, optional): Initial bank support level. Defaults to 0.0.

        Returns:
            SyntheticCentralGovernment: Container with preprocessed government data
        """
        country_exogenous_data = readers.get_exogenous_data(country_name)
        if country_exogenous_data is not None:
            # if exogenous data is available, use it to fit the benefits models
            benefits_inflation_data = readers.get_benefits_inflation_data(
                country_name, year_min=year - year_range, year_max=year, exogenous_data=country_exogenous_data
            )
            unemployment_benefits_model = build_unemployment_model(
                benefits_inflation_data, regression_window=regression_window
            )
            other_benefits_model = build_other_benefits_model(
                benefits_inflation_data, regression_window=regression_window
            )
            last_observation = (
                benefits_inflation_data[["Real CPI Inflation", "Unemployment Rate"]].iloc[-1].values.reshape(1, -1)
            )

            if unemployment_benefits_model:
                current_unemployment_benefits = (
                    unemployment_benefits_model.predict(last_observation)[0]
                    * benefits_inflation_data["Unemployment Benefits"].iloc[-1]
                )
            else:
                current_unemployment_benefits = benefits_inflation_data["Unemployment Benefits"].iloc[-1]

            if other_benefits_model:
                current_other_benefits = (
                    other_benefits_model.predict(last_observation)[0]
                    * benefits_inflation_data["Other Total Benefits"].iloc[-1]
                )
            else:
                current_other_benefits = benefits_inflation_data["Other Total Benefits"].iloc[-1]
        else:
            # if exogenous data is not available, set the benefits models to None
            unemployment_benefits_model = None
            other_benefits_model = None
            current_unemployment_benefits = readers.get_total_unemployment_benefits_lcu(country_name, year)
            current_other_benefits = readers.get_total_benefits_lcu(country_name, year) - current_unemployment_benefits

        # TODO: debt in USD or in local currency?

        debt = readers.world_bank.get_central_gov_debt(country_name, year)

        central_gov_data = pd.DataFrame(
            data={
                "Total Unemployment Benefits": [current_unemployment_benefits],
                "Other Social Benefits": [current_other_benefits],
                "Debt": [debt],
                "Bank Equity Injection": [equity_injection],
            }
        )

        central_gov = DefaultSyntheticCGovernment(
            country_name=country_name,
            year=year,
            central_gov_data=central_gov_data,
            other_benefits_model=other_benefits_model,
            unemployment_benefits_model=unemployment_benefits_model,
        )
        return central_gov


def build_unemployment_model(benefits_inflation_data: pd.DataFrame, regression_window: int = 48):
    """Estimate a model for preprocessing unemployment benefits data.

    This function estimates parameters for preprocessing unemployment benefits by:
    1. Computing historical growth ratios
    2. Fitting a model based on inflation and unemployment
    3. Preparing parameters for initialization

    Args:
        benefits_inflation_data (pd.DataFrame): Historical benefits and inflation data
        regression_window (int, optional): Months of data for estimation. Defaults to 48.

    Returns:
        LinearRegression: Estimated model for preprocessing, or None if insufficient data
    """
    benefits_inflation_data["Unemployment benefits growth ratio"] = (
        1 + benefits_inflation_data["Unemployment Benefits"].pct_change()
    )
    # Select last N months of data using loc with date filtering
    end_date = benefits_inflation_data.index.max()
    start_date = end_date - pd.DateOffset(months=regression_window)
    selection = benefits_inflation_data.loc[start_date:end_date].dropna()
    if selection.shape[0] > 0:
        model = LinearRegression()
        model.fit(
            selection[["Real CPI Inflation", "Unemployment Rate"]],
            selection["Unemployment benefits growth ratio"],
        )
        return model
    else:
        return None


def build_other_benefits_model(benefits_inflation_data: pd.DataFrame, regression_window: int = 48):
    """Estimate a model for preprocessing other benefits data.

    This function estimates parameters for preprocessing non-unemployment benefits by:
    1. Computing historical growth ratios
    2. Fitting a model based on inflation and unemployment
    3. Preparing parameters for initialization

    Args:
        benefits_inflation_data (pd.DataFrame): Historical benefits and inflation data
        regression_window (int, optional): Months of data for estimation. Defaults to 48.

    Returns:
        LinearRegression: Estimated model for preprocessing, or None if insufficient data
    """
    benefits_inflation_data["Other benefits growth ratio"] = (
        1 + benefits_inflation_data["Other Total Benefits"].pct_change()
    )
    selection = benefits_inflation_data.iloc[-regression_window:].dropna()
    if selection.shape[0] > 0:
        model = LinearRegression()
        model.fit(
            selection[["Real CPI Inflation", "Unemployment Rate"]],
            selection["Other benefits growth ratio"],
        )
        return model
    else:
        return None
