import os
import json
import numpy as np
import pandas as pd

from pathlib import Path
from functools import reduce

from data.readers.io_tables.util import aggregate_df
from data.readers.economic_data.exchange_rates import WorldBankRatesReader


class ICIOReader:
    def __init__(
        self,
        iot: pd.DataFrame,
        considered_countries: list[str],
        industries: list[str],
        imputed_rents: dict[str, float],
        year: int,
    ):
        self.iot = iot
        self.considered_countries = considered_countries
        self.industries = industries
        self.imputed_rents = imputed_rents
        self.year = year
        self.investment_matrices = {}

        # Normalisation
        self.normalise_iot()

    @classmethod
    def agg_from_csv(
        cls,
        path: Path,
        pivot_path: Path,
        considered_countries: list[str],
        aggregation_path: Path,
        industries: list[str],
        year: int,
        exchange_rates: WorldBankRatesReader,
        imputed_rent_fraction: dict[str, float],
    ):
        # This is quite slow, so adding the option of loading it
        if os.path.isfile(pivot_path):
            df = pd.read_csv(pivot_path, index_col=[0, 1], header=[0, 1])
        else:
            df = cls.read_df(path)
            df.to_csv(pivot_path)

        # Aggregate the IOT
        aggregation = json.load(open(aggregation_path))
        agg_df = cls.aggregate_io(considered_countries, df, aggregation)

        # Isolate-out imputed rents
        imputed_rents = {}
        for country_name in imputed_rent_fraction.keys():
            imputed_rents[country_name] = (
                (
                    imputed_rent_fraction[country_name]
                    * agg_df.at[(country_name, "L"), (country_name, "Household Consumption")]
                )
                / 12.0
                * exchange_rates.from_usd_to_lcu(country_name, year)
            )
            agg_df.at[(country_name, "L"), (country_name, "Household Consumption")] -= imputed_rents[country_name]

        return cls(
            iot=agg_df,
            considered_countries=considered_countries,
            industries=industries,
            imputed_rents=imputed_rents,
            year=year,
        )

    @staticmethod
    def read_df(path: Path | str) -> pd.DataFrame:
        df = pd.read_csv(path, index_col=0)
        df.index.name = "rows"
        df.columns.name = "columns"
        df = pd.melt(df.reset_index(), id_vars="rows")
        df["_inrow"] = df["rows"].str.contains("_")
        df["_incol"] = df["columns"].str.contains("_")
        sep_cols = df["columns"].str.split("_", expand=True)
        sep_rows = df["rows"].str.split("_", expand=True)
        df["col_1"] = np.where(df["_incol"], sep_cols[0], df["columns"])
        df["col_2"] = np.where(df["_incol"], sep_cols[1], df["columns"])
        df["rows_1"] = np.where(df["_inrow"], sep_rows[0], df["rows"])
        df["rows_2"] = np.where(df["_inrow"], sep_rows[1], df["rows"])
        df.drop(columns=["columns", "rows"], inplace=True)
        df = df.pivot(index=["rows_1", "rows_2"], columns=["col_1", "col_2"], values="value")
        df.index.names = ["CountryInd", "industryInd"]
        df.columns.names = ["CountryCol", "industryCol"]
        return df.sort_index(axis=0).sort_index(axis=1)

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
            industrial aggregation dictionary

        Returns
        -------
        pd.DataFrame
        the aggregated io-table.
        """

        # Build the aggregation dictionary
        col_level_0 = df.columns.get_level_values(0).unique()
        keep_level_0 = considered_countries + ["ROW", "TOTAL"]
        discard_level_0 = [c for c in col_level_0 if c not in keep_level_0]
        country_agg_dict = {c: "ROW" for c in discard_level_0}
        for c in keep_level_0:
            country_agg_dict[c] = c
        country_agg_dict["OUTPUT"] = "TOTAL"
        country_agg_dict["VALU"] = "TOTAL"
        country_agg_dict["TAXSUB"] = "TOTAL"

        # Perform the aggregation
        aggregated = aggregate_df(aggregation, country_agg_dict, df)

        # Cosmetics
        aggregated *= 1e6
        aggregated.index.names = ["Country", "Industry"]
        aggregated.columns.names = ["Country", "Industry"]

        return aggregated

    def normalise_iot(self) -> None:
        """
        Normalises the IOT by adjusting value-added.
        """

        # Sums-up intermediate inputs into a new row
        self.iot.loc[
            ("TOTAL", "Intermediate Inputs"),
            self.iot.columns.get_level_values(1).isin(self.industries),
        ] = self.iot.loc[
            (self.iot.index != ("TOTAL", "Value Added"))
            & (self.iot.index.get_level_values(1) != "Taxes Less Subsidies"),
            self.iot.columns.get_level_values(1).isin(self.industries),
        ].sum(
            axis=0
        )

        # Sums-up taxes-less-subsidies into a new row
        self.iot.loc[
            ("TOTAL", "Taxes Less Subsidies"),
            self.iot.columns.get_level_values(1).isin(self.industries),
        ] = self.iot.loc[
            self.iot.index.get_level_values(1) == "Taxes Less Subsidies",
            self.iot.columns.get_level_values(1).isin(self.industries),
        ].sum(
            axis=0
        )
        self.iot = self.iot.loc[
            np.logical_not(
                (self.iot.index.get_level_values(0) != "TOTAL")
                & (self.iot.index.get_level_values(1) == "Taxes Less Subsidies")
            )
        ].copy()

        # Adds total output
        output = self.iot.loc[self.iot.index.get_level_values(1).isin(self.industries)].sum(axis=1)
        self.iot.loc[:, ("TOTAL", "Output")] = np.nan
        self.iot.loc[
            self.iot.index.get_level_values(1).isin(self.industries),
            ("TOTAL", "Output"),
        ] = output
        self.iot.loc[
            ("TOTAL", "Output"),
            self.iot.columns.get_level_values(1).isin(self.industries),
        ] = output

        # Adjust value-added
        self.iot.loc[("TOTAL", "Value Added")] = (
            self.iot.loc[("TOTAL", "Output")]
            - self.iot.loc[("TOTAL", "Intermediate Inputs")]
            - self.iot.loc[("TOTAL", "Taxes Less Subsidies")]
        )

    def column_allc(self, country_name: str, symbol: str) -> pd.Series:
        considered_countries_row = self.considered_countries + ["ROW"]
        all_cols = [self.iot.loc[col, (country_name, symbol)].loc[self.industries] for col in considered_countries_row]
        return reduce(lambda a, b: a + b, all_cols)

    def get_monthly_total_output(self, country_name: str) -> np.ndarray:
        return (self.iot[("TOTAL", "Output")].xs(country_name, axis=0, level=0).loc[self.industries]).values / 12.0

    def get_monthly_intermediate_inputs_use(self, country_name: str) -> np.ndarray:
        return (
            reduce(
                lambda a, b: a + b,
                [
                    self.iot.loc[c_prime, country_name].loc[self.industries, self.industries]
                    for c_prime in self.considered_countries + ["ROW"]
                ],
            )
            / 12.0
        )

    def get_monthly_intermediate_inputs_supply(self, country_name: str) -> np.ndarray:
        return (
            reduce(
                lambda a, b: a + b,
                [
                    self.iot.loc[country_name, c_prime].loc[self.industries, self.industries]
                    for c_prime in self.considered_countries + ["ROW"]
                ],
            )
            / 12.0
        )

    def get_monthly_intermediate_inputs_domestic(self, country_name: str) -> np.ndarray:
        c_iot = self.iot.xs(country_name, axis=1, level=0)
        return c_iot.loc[country_name, c_iot.columns.isin(self.industries)] / 12.0

    def get_monthly_capital_inputs(self, country_name: str) -> np.ndarray:
        return self.column_allc(country_name, "Fixed Capital Formation").values / 12.0

    def get_gfcf_column(self, country_name: str) -> np.ndarray:
        return (
            self.iot.loc[
                self.iot.index.get_level_values(1).isin(self.industries),
                (country_name, "Fixed Capital Formation"),
            ].values
            / 12.0
        )

    def get_monthly_capital_inputs_domestic(self, country_name: str) -> np.ndarray:
        return self.iot.loc[country_name, country_name]["Fixed Capital Formation"].values / 12.0

    def get_monthly_value_added(self, country_name: str) -> np.ndarray:
        return self.iot.xs(country_name, axis=1, level=0).loc[("TOTAL", "Value Added"), self.industries].values / 12.0

    def get_monthly_taxes_less_subsidies(self, country_name: str) -> np.ndarray:
        return (
            self.iot.xs(country_name, axis=1, level=0).loc[("TOTAL", "Taxes Less Subsidies"), self.industries].values
        ) / 12.0

    def get_taxes_less_subsidies_rates(self, country_name: str) -> np.ndarray:
        return self.get_monthly_taxes_less_subsidies(country_name) / self.get_monthly_total_output(country_name)

    def get_monthly_hh_consumption(self, country_name: str) -> np.ndarray:
        return self.column_allc(country_name, "Household Consumption").values / 12.0

    def get_monthly_hh_consumption_domestic(self, country_name: str) -> np.ndarray:
        return self.iot.loc[country_name, (country_name, "Household Consumption")].values / 12.0

    def get_hh_consumption_weights(self, country_name: str) -> np.ndarray:
        hh_cons = self.get_monthly_hh_consumption(country_name)
        return hh_cons / hh_cons.sum()

    def get_monthly_govt_consumption(self, country_name: str) -> np.ndarray:
        return self.column_allc(country_name, "Government Consumption").values / 12.0

    def get_monthly_govt_consumption_domestic(self, country_name: str) -> np.ndarray:
        return self.iot.loc[country_name, (country_name, "Government Consumption")].values / 12.0

    def govt_consumption_weights(self, country_name: str) -> np.ndarray:
        gov_cons = self.get_monthly_govt_consumption(country_name)
        return gov_cons / gov_cons.sum()

    def get_imports(self, country_name: str) -> pd.Series:
        considered_countries_row = self.considered_countries + ["ROW"]
        imports = reduce(
            lambda a, b: a + b,
            (self.iot.loc[c2, country_name].sum(axis=1) for c2 in considered_countries_row if c2 != country_name),
        )
        return imports.loc[self.industries] / 12.0

    def get_exports(self, country_name: str) -> pd.Series:
        considered_countries_row = self.considered_countries + ["ROW"]
        exports = reduce(
            lambda a, b: a + b,
            (self.iot.loc[country_name, c2].sum(axis=1) for c2 in considered_countries_row if c2 != country_name),
        )
        return exports.loc[self.industries] / 12.0

    def get_trade(self, start_country: str, end_country: str) -> pd.Series:
        return self.iot.loc[start_country, end_country].sum(axis=1).loc[self.industries] / 12.0

    def get_intermediate_inputs_matrix(self, country_name: str) -> pd.DataFrame:
        total_output = self.get_monthly_total_output(country_name)
        total_monthly_intermediate_inputs = self.get_monthly_intermediate_inputs_use(country_name)
        return total_output[None, :] / total_monthly_intermediate_inputs

    def get_capital_inputs_matrix(
        self,
        country_name: str,
        capital_stock: np.ndarray,
    ) -> pd.DataFrame:
        norm_investment_matrix = self.investment_matrices[country_name].copy()
        norm_investment_matrix /= norm_investment_matrix.sum(axis=0)
        cap_inputs_matrix = (self.get_monthly_total_output(country_name) / capital_stock) / norm_investment_matrix
        return cap_inputs_matrix.xs(country_name, axis=0, level=0).xs(country_name, axis=1, level=0).fillna(np.inf)

    def get_capital_inputs_depreciation(
        self,
        country_name: str,
        capital_compensation: np.ndarray,
    ) -> pd.DataFrame:
        total_output = self.get_monthly_total_output(country_name)
        gfcf = self.get_monthly_capital_inputs(country_name)
        investment_matrix = np.array([gfcf for _ in range(len(capital_compensation))]).T
        norm_investment_matrix = investment_matrix / investment_matrix.sum(axis=0)
        norm_investment_matrix *= (capital_compensation / total_output)[None, :]
        return (
            pd.DataFrame(
                data=norm_investment_matrix,
                index=pd.Index(self.industries, name="Industries"),
                columns=pd.Index(self.industries, name="Industries"),
            )
            / 12.0
        )
