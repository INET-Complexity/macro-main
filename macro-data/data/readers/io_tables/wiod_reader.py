import json
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

from data.readers.io_tables.util import aggregate_df


# all in current USD
class WIODReader:
    def __init__(self, df: pd.DataFrame, considered_countries: list[str], industries: list[str]):
        self.df = df
        self.industries = industries
        self.considered_countries = considered_countries

    @classmethod
    def from_csv(cls, path: Path | str) -> "WIODReader":
        df = cls.read_csv(path)
        countries = df.columns.get_level_values(0).unique()
        considered_countries = [c for c in countries if c not in ["ROW", "TOT"]]
        industries = list(df.loc[considered_countries[0]].index)
        return cls(df, considered_countries=considered_countries, industries=industries)

    @staticmethod
    def read_csv(path: Path | str) -> pd.DataFrame:
        df = pd.read_csv(path, skiprows=1, header=[1, 2, 3, 4], index_col=[0, 1, 2, 3], thousands=",")
        df.index = df.index.droplevel(1)
        df.index = df.index.droplevel(2)
        df.columns = df.columns.droplevel(1)
        df.columns = df.columns.droplevel(2)
        df.drop(index=["PURR", "PURNR"], level=0, inplace=True)
        df = df.swaplevel(0, 1, axis=1)
        df = df.swaplevel(0, 1, axis=0)
        df.index.names = ["CountryInd", "industryInd"]
        df.columns.names = ["CountryCol", "industryCol"]
        return df

    @classmethod
    def agg_from_csv(
        cls,
        path: Path | str,
        considered_countries: list[str],
        aggregation_path: Path,
    ) -> "WIODReader":
        df = cls.read_csv(path)
        aggregation = json.load(open(aggregation_path))
        df = cls.aggregate_io(considered_countries, df, aggregation)
        industries = list(df.loc[considered_countries[0]].index)
        return cls(df, considered_countries, industries)

    @staticmethod
    def aggregate_io(
        considered_countries: list[str],
        df: pd.DataFrame,
        aggregation: dict[str, list[str]],
    ) -> pd.DataFrame:
        """
        Take an input output table and aggregate it.
        Pairs of (country, industry) identifiers for every entry are aggregated,
        countries may be aggregated into "ROW", the rest-of-the-world super-category.
        industries are mapped according to an AGG_DICT dictionary, that has pairs like
        'A': ['A01', 'A02', 'A03']
        indicating that these three industries go into industry A.
        Parameters
        ----------
        considered_countries : list[str]
            list of countries considered for the aggregation
        df : pd.DataFrame
            Input output table
        aggregation: dict
            industryal aggregation dictionary

        Returns
        -------
        pd.DataFrame
        the aggregated io-table.
        """
        col_level_0 = df.columns.get_level_values(0).unique()
        # tracks the countries we want to keep
        keep_level_0 = considered_countries + ["ROW", "TOT"]
        # countries we want to discard
        discard_level_0 = [c for c in col_level_0 if c not in keep_level_0]
        # build the country aggregation dictionary,
        # ie mapping AUS to ROW if Australia is not in considered countries
        country_agg_dict = {c: "ROW" for c in discard_level_0}
        for c in keep_level_0:
            country_agg_dict[c] = c
        aggregated = aggregate_df(aggregation, country_agg_dict, df)
        aggregated.dropna(axis=1, inplace=True)
        aggregated.drop(columns="TOT", level=0, inplace=True)
        aggregated *= 1e6  # units in USD
        return aggregated

    def column_allc(self, country: str, symbol: str) -> pd.Series:
        considered_countries_row = self.considered_countries + ["ROW"]
        all_cols = [self.df.loc[col, (country, symbol)].loc[self.industries] for col in considered_countries_row]
        return reduce(lambda a, b: a + b, all_cols)

    def capital_formation(self, country: str) -> np.ndarray:
        return self.column_allc(country, "GFCF").values

    def capital_weights(self, country: str) -> np.ndarray:
        cap_form = self.capital_formation(country)
        return cap_form / cap_form.sum()

    def hh_consumption(self, country: str) -> np.ndarray:
        return self.column_allc(country, "CONS_h") + self.column_allc(country, "CONS_np")

    def hh_consumption_weights(self, country: str) -> np.ndarray:
        hh_cons = self.hh_consumption(country)
        return hh_cons / hh_cons.sum()

    def govt_consumption(self, country: str) -> np.ndarray:
        return self.column_allc(country, "CONS_g").values

    def govt_cons_weights(self, country: str) -> np.ndarray:
        gov_cons = self.govt_consumption(country)
        return gov_cons / gov_cons.sum()

    def intermediate_inputs(self, country: str) -> np.ndarray:
        return reduce(
            lambda a, b: a + b,
            [
                self.df.loc[c_prime, country].loc[self.industries, self.industries]
                # flow of goods is row->columns,
                # so we count all incoming goods
                for c_prime in self.considered_countries + ["ROW"]
            ],
        ).values

    def intermediate_input_weights(self, country: str) -> np.ndarray:
        ii_g = self.intermediate_inputs(country)
        # normalise so that columns sum to 1
        return ii_g / ii_g.sum(axis=0)

    def total_exports(self, country: str) -> np.ndarray:
        considered_countries_row = self.considered_countries + ["ROW"]
        exports = reduce(
            lambda a, b: a + b,
            (
                self.df.loc[country, c2].loc[self.industries].sum(axis=1).values
                for c2 in considered_countries_row
                if c2 != country
            ),
        )
        return exports

    def total_imports(self, country: str) -> np.ndarray:
        considered_countries_row = self.considered_countries + ["ROW"]
        imports = reduce(
            lambda a, b: a + b,
            (
                self.df.loc[c2, country].loc[self.industries].sum(axis=0)
                for c2 in considered_countries_row
                if c2 != country
            ),
        )

        return imports
