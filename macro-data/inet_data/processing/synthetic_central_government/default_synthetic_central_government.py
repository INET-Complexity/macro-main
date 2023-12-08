from typing import Optional

import pandas as pd
from sklearn.linear_model import LinearRegression

from inet_data.processing.synthetic_central_government.synthetic_central_government import (
    SyntheticCentralGovernment,
)
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
            current_unemployment_benefits = (
                unemployment_benefits_model.predict(last_observation)[0]
                * benefits_inflation_data["Unemployment Benefits"].iloc[-1]
            )
            current_other_benefits = (
                other_benefits_model.predict(last_observation)[0]
                * benefits_inflation_data["Other Total Benefits"].iloc[-1]
            )
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
    # select a regression window with a span of a given amount of months
    benefits_inflation_data["Unemployment benefits growth ratio"] = (
        1 + benefits_inflation_data["Unemployment Benefits"].pct_change()
    )
    selection = benefits_inflation_data.last(f"{regression_window}M").dropna()
    model = LinearRegression()
    model.fit(
        selection[["Real CPI Inflation", "Unemployment Rate"]],
        selection["Unemployment benefits growth ratio"],
    )
    return model


def build_other_benefits_model(benefits_inflation_data: pd.DataFrame, regression_window: int = 48):
    # select a regression window with a span of a given amount of months
    benefits_inflation_data["Other benefits growth ratio"] = (
        1 + benefits_inflation_data["Other Total Benefits"].pct_change()
    )
    selection = benefits_inflation_data.last(f"{regression_window}M").dropna()
    model = LinearRegression()
    model.fit(
        selection[["Real CPI Inflation", "Unemployment Rate"]],
        selection["Other benefits growth ratio"],
    )
    return model
