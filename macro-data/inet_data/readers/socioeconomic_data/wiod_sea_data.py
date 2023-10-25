import json
from pathlib import Path

import numpy as np
import pandas as pd

from inet_data.readers.economic_data.exchange_rates import WorldBankRatesReader


class WIODSEAReader:
    def __init__(
        self,
        df: pd.DataFrame,
        year: int,
        industries: list[str],
        exchange_rates: WorldBankRatesReader,
    ):
        self.df = df
        self.year = year
        self.industries = industries
        self.exchange_rates = exchange_rates

        self.clean_sea()

    @classmethod
    def agg_from_csv(
        cls,
        path: Path | str,
        aggregation_path: Path,
        year: int,
        country_names: list[str],
        industries: list,
        exchange_rates: WorldBankRatesReader,
    ):
        # Aggregate industries
        raw_df = pd.read_csv(path, thousands=",", index_col=[0, 1, 2, 3])
        aggregation = json.load(open(aggregation_path))
        agg_dict_full = {}
        for key, values in aggregation.items():
            for value in values:
                agg_dict_full[value] = key
        stacked = raw_df[str(year)].reset_index()
        stacked.rename(columns={str(year): "Value"}, inplace=True)

        # Don't include indices or employment info
        stacked = stacked[stacked["variable"].isin(["VA", "LAB", "CAP", "K"])]

        # Convert to USD
        stacked["Value"] /= stacked["country"].map(exchange_rates.exchange_rates_dict(year))
        stacked["Value"] *= 1e6

        # Aggregate
        stacked["new_code"] = stacked["code"].map(agg_dict_full)

        # Unstack things
        sea = stacked.groupby(["country", "new_code", "variable"])["Value"].sum().unstack()

        # Cosmetics
        sea = sea.loc[sea.index.get_level_values(0).isin(country_names)]
        sea = sea.loc[sea.index.get_level_values(1).isin(industries)]
        sea.index.names = ["Country", "Industry"]
        sea.columns.name = "Field"
        sea.rename(
            {
                "VA": "Value Added",
                "LAB": "Labour Compensation",
                "CAP": "Capital Compensation",
                "K": "Capital Stock",
            },
            axis=1,
            inplace=True,
        )

        return cls(
            df=sea,
            year=year,
            industries=industries,
            exchange_rates=exchange_rates,
        )

    def clean_sea(self) -> None:
        # Overwrite negative capital compensation
        self.df.loc[:, "Capital Compensation"] = np.maximum(0.0, self.df.loc[:, "Capital Compensation"])

    def get_values_in_usd(self, country: str, field: str) -> np.ndarray:
        return self.df.loc[country].loc[self.industries, field].values

    def get_values_in_lcu(self, country: str, field: str) -> np.ndarray:
        return self.get_values_in_usd(country, field) * self.exchange_rates.from_usd_to_lcu(country, self.year)
