import warnings
from pathlib import Path
from warnings import simplefilter

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer  # noqa

from macro_data.configuration.countries import Country
from macro_data.readers.economic_data.exchange_rates import ExchangeRatesReader

var_mapping = {
    "curcdq": "Currency Code",
    "emp": "Number of Employees",
    "atq": "Assets",
    "ceqq": "Equity",
    "dlttq": "Debt",
    "dptbq": "Deposits",
    "invtq": "Inventory",
    "ltq": "Liabilities",
    "revtq": "Revenue",
    "gpq": "Profits",
    "gsector": "Sector",
    "loc": "Country",
}
var_numerical = [
    "Assets",
    "Equity",
    "Debt",
    "Deposits",
    "Inventory",
    "Liabilities",
    "Revenue",
    "Profits",
]
var_keeping = [
    "Number of Employees",
    "Assets",
    "Equity",
    "Debt",
    "Deposits",
    "Inventory",
    "Liabilities",
    "Revenue",
    "Profits",
    "Sector",
    "Country",
    "Currency Code",
]

simplefilter("ignore", category=ConvergenceWarning)


class CompustatFirmsReader:
    def __init__(
        self,
        data: pd.DataFrame,
    ):
        self.data = data

    @classmethod
    def from_raw_data(
        cls,
        year: int,
        quarter: int,
        raw_annual_path: Path | str,
        raw_quarterly_path: Path | str,
        countries: list[str | Country],
    ):
        raw_annual_data = pd.read_csv(
            raw_annual_path,
            encoding="unicode_escape",
            engine="pyarrow",
        )
        raw_quarterly_data = pd.read_csv(
            raw_quarterly_path,
            encoding="unicode_escape",
            engine="pyarrow",
        )
        raw_quarterly_data = raw_quarterly_data[
            np.logical_and(
                raw_quarterly_data["fyearq"] == year,
                raw_quarterly_data["fqtr"] == quarter,
            )
        ]

        # Pick
        annual_data = raw_annual_data.dropna(axis=0, how="all").dropna(axis=1, how="all")
        quarterly_data = raw_quarterly_data.dropna(axis=0, how="all").dropna(axis=1, how="all")

        # select only the countries we want
        annual_data = annual_data[annual_data["loc"].isin(countries)]
        quarterly_data = quarterly_data[quarterly_data["loc"].isin(countries)]

        # Merge
        data = pd.merge(
            quarterly_data,
            annual_data,
            on="conm",
        )

        # drop loc_y and keep loc_x
        data.drop(columns=["loc_y"], inplace=True)
        data.rename(columns={"loc_x": "loc"}, inplace=True)

        data = data.rename(columns=var_mapping)

        data = data[var_keeping]

        # Impute
        # select all columns except Country and Currency Code
        column_selection = [col for col in data.columns if col not in ["Country", "Currency Code"]]
        data.loc[:, column_selection] = IterativeImputer().fit_transform(data[column_selection].values)

        data.set_index("Country", inplace=True)

        return cls(data)

    @property
    def numerical_columns(self):
        return var_numerical

    def get_firm_data(self, country: str | Country) -> pd.DataFrame:
        if isinstance(country, Country):
            country = country.value
        return self.data.loc[country]

    def get_proxied_firm_data(self, proxy_country: str | Country, exchange_rate: float) -> pd.DataFrame:
        if isinstance(proxy_country, Country):
            proxy_country = proxy_country.value
        proxied = self.data.loc[proxy_country, self.numerical_columns].copy()
        proxied = proxied * exchange_rate
        return proxied
