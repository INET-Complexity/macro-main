from typing import Optional

import pandas as pd
from sklearn.linear_model import LinearRegression

from inet_data.processing.synthetic_banks.synthetic_banks import SyntheticBanks
from inet_data.processing.synthetic_central_government.synthetic_central_government import (
    SyntheticCentralGovernment,
)
from inet_data.processing.synthetic_firms.synthetic_firms import SyntheticFirms
from inet_data.processing.synthetic_population.synthetic_population import SyntheticPopulation
from inet_data.readers.default_readers import DataReaders


class SyntheticDefaultCentralGovernment(SyntheticCentralGovernment):
    def __init__(
        self,
        country_name: str,
        year: int,
        central_gov_data: pd.DataFrame,
        other_benefits_model: Optional[LinearRegression],
        unemployment_benefits_model: Optional[LinearRegression],
    ):
        """
        Represents a synthetic central government.

        Attributes:
            country_name (str): The name of the country.
            year (int): The year.
            central_gov_data (pd.DataFrame): The central government data.
            other_benefits_model (Optional[LinearRegression]): The model for other benefits (optional).
            unemployment_benefits_model (Optional[LinearRegression]): A linear regression model to determine unemployment benefits,
                                                                    based on e.g. unemployment rate and inflation (optional).
        """
        super().__init__(
            country_name,
            year,
            central_gov_data,
            other_benefits_model,
            unemployment_benefits_model,
        )

    @classmethod
    def create_from_readers(
        cls,
        readers: DataReaders,
        country_name: str,
        year: int,
        year_range: int = 10,
        regression_window: int = 48,
        equity_injection: float = 0.0,
    ) -> SyntheticCentralGovernment:
        """
        Create a synthetic central government object from a DataReaders object.

        This first checks if exogenous data is available for the country and year. If so, it uses this data to fit a model
        for unemployment benefits and other benefits. If not, it sets the models to None.

        Then it returns a SyntheticCentralGovernment object with the fitted models and the current values of the benefits.

        Arguments:
            readers (DataReaders): DataReaders object.
            country_name (str): Name of the country.
            year (int): Year.
            year_range (int, optional): Number of years to use for fitting the benefits models. Defaults to 10.
            regression_window (int, optional): Number of months to use for fitting the benefits models. Defaults to 48.
            equity_injection (float, optional): Amount of equity injection. Defaults to 0.0.

        Returns:
            SyntheticCentralGovernment: Synthetic central government object.
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
                current_unemployment_benefits = readers.get_total_unemployment_benefits(country_name, year)

            if other_benefits_model:
                current_other_benefits = (
                    other_benefits_model.predict(last_observation)[0]
                    * benefits_inflation_data["Other Total Benefits"].iloc[-1]
                )
            else:
                current_other_benefits = readers.get_total_benefits(country_name, year) - current_unemployment_benefits
        else:
            # if exogenous data is not available, set the benefits models to None
            unemployment_benefits_model = None
            other_benefits_model = None
            current_unemployment_benefits = readers.get_total_unemployment_benefits(country_name, year)
            current_other_benefits = readers.get_total_benefits(country_name, year) - current_unemployment_benefits

        debt = readers.oecd_econ.general_gov_debt(country_name, year)

        central_gov_data = pd.DataFrame(
            data={
                "Total Unemployment Benefits": [current_unemployment_benefits],
                "Other Social Benefits": [current_other_benefits],
                "Debt": [debt],
                "Bank Equity Injection": [equity_injection],
            }
        )

        central_gov = SyntheticDefaultCentralGovernment(
            country_name=country_name,
            year=year,
            central_gov_data=central_gov_data,
            other_benefits_model=other_benefits_model,
            unemployment_benefits_model=unemployment_benefits_model,
        )
        return central_gov


def build_unemployment_model(benefits_inflation_data: pd.DataFrame, regression_window: int = 48):
    """
    Build a linear regression model to predict the growth ratio of unemployment benefits based on
    the real CPI inflation and the unemployment rate.

    Parameters:
    - benefits_inflation_data (pd.DataFrame): DataFrame containing the benefits inflation data.
    - regression_window (int): Number of months to consider for the regression window.

    Returns:
    - model (LinearRegression): The trained linear regression model if there is enough data, otherwise None.
    """
    benefits_inflation_data["Unemployment benefits growth ratio"] = (
        1 + benefits_inflation_data["Unemployment Benefits"].pct_change()
    )
    selection = benefits_inflation_data.last(f"{regression_window}M").dropna()
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
    """
    Build a linear regression model to predict the growth ratio of other benefits (i.e. benefits that are not
    unemployment benefits) based on real CPI inflation and unemployment rate.

    Parameters:
        benefits_inflation_data (pd.DataFrame): DataFrame containing the benefits inflation data.
        regression_window (int, optional): Number of months to consider for the regression window. Defaults to 48.

    Returns:
        model (LinearRegression): Linear regression model trained on the selected data, or None if no data is available.
    """
    # select a regression window with a span of a given amount of months
    benefits_inflation_data["Other benefits growth ratio"] = (
        1 + benefits_inflation_data["Other Total Benefits"].pct_change()
    )
    selection = benefits_inflation_data.last(f"{regression_window}M").dropna()
    if selection.shape[0] > 0:
        model = LinearRegression()
        model.fit(
            selection[["Real CPI Inflation", "Unemployment Rate"]],
            selection["Other benefits growth ratio"],
        )
        return model
    else:
        return None
