from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from macro_data.configuration.countries import Country


def country_code_switch(codes: Iterable[str]):
    return [Country.convert_two_letter_to_three(c) for c in codes]


def preprocess_df(df: pd.DataFrame, freq="QS") -> Optional[pd.Series]:
    # df = df.loc[:, df.columns != "TIME PERIOD"]
    df.drop(columns="TIME PERIOD", inplace=True)
    df.columns = [c[-26:-24] for c in df.columns]
    df.drop(columns="U2", inplace=True)
    df.columns = country_code_switch(df.columns)
    data = df.resample(freq).mean()
    data.freq = None
    return data


class ECBReader:
    def __init__(
        self,
        path: Path | str,
        proxy_country: Country = Country("DEU"),
    ):
        # For proxying
        self.proxy_country = proxy_country

        # Load data files
        self.data = {}
        for f in [
            "firm_loans",
            "household_loans_for_consumption",
            "household_loans_for_mortgages",
        ]:
            filepath = path / (f + ".csv")
            self.data[f] = preprocess_df(pd.read_csv(filepath, index_col="DATE", parse_dates=True))

    def get_firm_rates(self, country_name: str) -> Optional[pd.Series]:
        df = self.data["firm_loans"].copy()
        if country_name in df.columns:
            return df[country_name] / 100.0
        else:
            return None

    def get_household_consumption_rates(self, country_name: str) -> Optional[pd.Series]:
        df = self.data["household_loans_for_consumption"].copy()
        if country_name in df.columns:
            return df[country_name] / 100.0
        else:
            return None

    def get_household_mortgage_rates(self, country_name: str) -> Optional[pd.Series]:
        df = self.data["household_loans_for_mortgages"].copy()
        if country_name in df.columns:
            return df[country_name] / 100.0
        else:
            return None
