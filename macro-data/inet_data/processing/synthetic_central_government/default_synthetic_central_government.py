import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from inet_data.processing.synthetic_central_government.synthetic_central_government import (
    SyntheticCentralGovernment,
)

from typing import Any


class SyntheticDefaultCentralGovernment(SyntheticCentralGovernment):
    def __init__(
        self,
        country_name: str,
        year: int,
    ):
        super().__init__(
            country_name,
            year,
        )

    def set_central_government_debt(self, central_gov_debt: float) -> None:
        self.central_gov_data["Debt"] = [central_gov_debt]

    def set_total_unemployment_benefits(
        self,
        benefits_data: pd.DataFrame,
        exogenous_data: dict[str, Any],
        regression_window: int = 48,
    ) -> None:
        if exogenous_data is None:
            self.central_gov_data["Total Unemployment Benefits"] = [benefits_data["Unemployment Benefits"].values[-1]]
            return

        # Benefits
        benefits_data = benefits_data["Unemployment Benefits"].astype(float).resample("M").interpolate("linear").copy()
        benefits_data.index = pd.DatetimeIndex([pd.Timestamp(d.year, d.month, 1) for d in benefits_data.index])
        benefits_data_log = benefits_data / benefits_data.shift(1)
        curr_benefits = benefits_data_log.iloc[-regression_window:].values

        # Inflation
        log_inflation = exogenous_data["log_inflation"]["Real CPI Inflation"].copy()
        log_inflation.index = pd.DatetimeIndex(
            [pd.Timestamp(year=int(ind[0:4]), month=int(ind[5:]), day=1) for ind in log_inflation.index.values]
        )
        log_inflation = log_inflation.loc[log_inflation.index <= max(benefits_data.index)]
        curr_inflation = log_inflation.iloc[-regression_window:].values

        # Unemployment
        unemployment_rate = exogenous_data["unemployment_rate"]["Unemployment Rate"].copy()
        unemployment_rate.index = pd.DatetimeIndex(
            [pd.Timestamp(year=int(ind[0:4]), month=int(ind[5:]), day=1) for ind in unemployment_rate.index.values]
        )
        unemployment_rate = unemployment_rate.loc[unemployment_rate.index <= max(benefits_data.index)]
        curr_unemployment_rate = unemployment_rate.iloc[-regression_window:].values

        # Fit
        x = np.stack([curr_inflation, curr_unemployment_rate], axis=1)
        nan_ind = np.isnan(x).any(axis=1)
        self.unemployment_benefits_model = LinearRegression().fit(x[~nan_ind], curr_benefits[~nan_ind])
        pred = self.unemployment_benefits_model.predict(np.array([[log_inflation[-1], unemployment_rate[-1]]]))[  # noqa
            0
        ]

        self.central_gov_data["Total Unemployment Benefits"] = [pred * benefits_data[-1]]

    def set_other_social_benefits(
        self,
        benefits_data: pd.DataFrame,
        exogenous_data: dict[str, Any],
        regression_window: int = 48,
    ) -> None:
        if exogenous_data is None:
            self.central_gov_data["Other Social Benefits"] = [benefits_data["Other Total Benefits"].values[-1]]
            return

        # Benefits
        benefits_data = benefits_data["Other Total Benefits"].astype(float).resample("M").interpolate("linear").copy()
        benefits_data.index = pd.DatetimeIndex([pd.Timestamp(d.year, d.month, 1) for d in benefits_data.index])
        benefits_data_log = benefits_data / benefits_data.shift(1)
        curr_benefits = benefits_data_log.iloc[-regression_window:].values

        # Inflation
        log_inflation = exogenous_data["log_inflation"]["Real CPI Inflation"].copy()
        log_inflation.index = pd.DatetimeIndex(
            [pd.Timestamp(year=int(ind[0:4]), month=int(ind[5:]), day=1) for ind in log_inflation.index.values]
        )
        log_inflation = log_inflation.loc[log_inflation.index <= max(benefits_data.index)]
        curr_inflation = log_inflation.iloc[-regression_window:].values

        # Unemployment
        unemployment_rate = exogenous_data["unemployment_rate"]["Unemployment Rate"].copy()
        unemployment_rate.index = pd.DatetimeIndex(
            [pd.Timestamp(year=int(ind[0:4]), month=int(ind[5:]), day=1) for ind in unemployment_rate.index.values]
        )
        unemployment_rate = unemployment_rate.loc[unemployment_rate.index <= max(benefits_data.index)]
        curr_unemployment_rate = unemployment_rate.iloc[-regression_window:].values

        # Fit
        x = np.stack([curr_inflation, curr_unemployment_rate], axis=1)
        nan_ind = np.isnan(x).any(axis=1)
        self.other_benefits_model = LinearRegression().fit(x[~nan_ind], curr_benefits[~nan_ind])
        pred = self.other_benefits_model.predict(np.array([[log_inflation[-1], unemployment_rate[-1]]]))[0]  # noqa

        self.central_gov_data["Other Social Benefits"] = [pred * benefits_data[-1]]

    def set_initial_bank_equity_injection(self) -> None:
        self.central_gov_data["Bank Equity Injection"] = [0.0]
