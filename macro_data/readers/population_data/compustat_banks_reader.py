from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer  # noqa

from macro_data.configuration.countries import Country
from macro_data.readers.economic_data.exchange_rates import ExchangeRatesReader

var_mapping = {
    "fqtr": "Quarter",
    "fyearq": "Year",
    "loc": "Country",
    "curcdq": "Currency Code",
    "atq": "Assets",
    "ciq": "Income",
    "dlttq": "Debt",
    "dptcq": "Deposits",
    "ltq": "Liabilities",
    "teqq": "Equity",
    "dltisy": "Long-term Debt Issuance",
    "dltry": "Long-term Debt Reduction",
}
var_numerical = [
    "Assets",
    "Income",
    "Debt",
    "Deposits",
    "Liabilities",
    "Equity",
    "Long-term Debt Issuance",
    "Long-term Debt Reduction",
]
var_keeping = ["Assets", "Debt", "Deposits", "Liabilities", "Equity"]


class CompustatBanksReader:
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
        raw_quarterly_path: Path | str,
        countries: list[str | Country],
        proxy_with_us: bool = True,
    ):
        raw_data = pd.read_csv(raw_quarterly_path, encoding="unicode_escape", engine="pyarrow")

        # pick
        data = raw_data[np.logical_and(raw_data["fyearq"] == year, raw_data["fqtr"] == quarter)]

        if proxy_with_us:
            countries += [Country("USA")]

        # pick countries in the list
        data = data[data["loc"].isin(countries)]

        # rename loc to country and set it as index
        data.rename(columns={"loc": "Country"}, inplace=True)
        data.set_index("Country", inplace=True)

        # pick columns
        data = data[[col for col in var_mapping.keys() if col in data.columns]]
        # rename columns
        data.rename(columns=var_mapping, inplace=True)
        # keep only the columns we want
        data = data[var_keeping]

        # impute missing values

        for c in data.index.get_level_values(0).unique():
            data_values = data.loc[c].values
            if len(data_values) == 1:
                data_values = data_values.reshape(1, -1)
            data.loc[c] = IterativeImputer().fit_transform(data_values)

        return cls(data)

    @property
    def numerical_columns(self):
        # list of numerical columns, ie var_numerical and var_keeping
        return [col for col in var_numerical if col in self.data.columns]

    def get_country_data(self, country: str, exchange_rate: float) -> pd.DataFrame:
        if country == "USA":
            return self.data.loc[country]
        else:
            proxied = self.data.loc["USA"].copy()
            proxied[self.numerical_columns] *= exchange_rate
            return proxied

    def get_proxied_country_data(self, proxy_country: str | Country, exchange_rate: float):
        if isinstance(proxy_country, Country):
            proxy_country = proxy_country.value
        proxied = self.data.loc[proxy_country, self.numerical_columns].copy()
        proxied = proxied * exchange_rate
        return proxied
