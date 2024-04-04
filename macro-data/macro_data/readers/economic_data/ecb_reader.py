import pandas as pd

from pathlib import Path

from typing import Optional, Iterable

from macro_data.configuration.countries import Country


def country_code_switch(codes: Iterable[str]):
    return [Country.convert_two_letter_to_three(c) for c in codes]


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
            self.data[f] = pd.read_csv(path / (f + ".csv")).set_index("DATE", drop=True)

    def get_firm_rates(self, country_name: str) -> Optional[pd.DataFrame]:
        df = self.data["firm_loans"].copy()
        df = df.loc[:, df.columns != "TIME PERIOD"]
        df.columns = [c[-26:-24] for c in df.columns]
        df = df.loc[:, df.columns != "U2"]
        df.columns = country_code_switch(df.columns)
        if country_name in df.columns:
            df = df[country_name]
        else:
            return None
        df = df.groupby(pd.PeriodIndex(df.index, freq="Q")).mean()
        df.index = df.index.to_timestamp()
        return df / 100.0

    def get_household_consumption_rates(self, country_name: str) -> Optional[pd.DataFrame]:
        df = self.data["household_loans_for_consumption"].copy()
        df = df.loc[:, df.columns != "TIME PERIOD"]
        df.columns = [c[-26:-24] for c in df.columns]
        df = df.loc[:, df.columns != "U2"]
        df.columns = country_code_switch(df.columns)
        if country_name in df.columns:
            df = df[country_name]
        else:
            return None
        df = df.groupby(pd.PeriodIndex(df.index, freq="Q")).mean()
        df.index = df.index.to_timestamp()
        return df / 100.0

    def get_household_mortgage_rates(self, country_name: str) -> Optional[pd.DataFrame]:
        df = self.data["household_loans_for_mortgages"].copy()
        df = df.loc[:, df.columns != "TIME PERIOD"]
        df.columns = [c[-26:-24] for c in df.columns]
        df = df.loc[:, df.columns != "U2"]
        df.columns = country_code_switch(df.columns)
        if country_name in df.columns:
            df = df[country_name]
        else:
            return None
        df = df.groupby(pd.PeriodIndex(df.index, freq="Q")).mean()
        df.index = df.index.to_timestamp()
        return df / 100.0
