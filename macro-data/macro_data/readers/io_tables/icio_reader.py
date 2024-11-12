import json
import os
from functools import reduce
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.readers.economic_data.exchange_rates import ExchangeRatesReader
from macro_data.readers.io_tables.mappings import ICIO_AGGREGATE, ICIO_ALL
from macro_data.readers.io_tables.util import aggregate_df


class ICIOReader:
    """
    A class for reading and manipulating Input-Output Tables (IOT) from the OECD Inter-Country Input Output Tables.

    Parameters
    ----------
    iot : pd.DataFrame
        The input-output table.
    considered_countries : list[str]
        List of countries considered for the aggregation.
    industries : list[str]
        List of industries.
    imputed_rents : dict[str, float]
        Dictionary of imputed rents for each country.
    year : int
        The year of the input-output table.

    Methods
    -------
    agg_from_csv(cls, path, pivot_path, considered_countries, aggregation_path, industries, year, exchange_rates, imputed_rent_fraction)
        Class method to aggregate the input-output table from CSV files (i.e. map unused countries to the ROW).
    read_df(path)
        Static method to read the input-output table from a CSV file.
    aggregate_io(considered_countries, df, aggregation)
        Static method to aggregate the input-output table.
    normalise_iot()
        Normalizes the input-output table by adjusting value-added.
    column_allc(country_name, symbol)
        Returns the sum of columns for a specific country and symbol.
    get_monthly_total_output(country_name)
        Returns the monthly total output for a specific country.
    get_monthly_intermediate_inputs_use(country_name)
        Returns the monthly intermediate inputs use for a specific country.
    get_monthly_intermediate_inputs_supply(country_name)
        Returns the monthly intermediate inputs supply for a specific country.
    get_monthly_intermediate_inputs_domestic(country_name)
        Returns the monthly domestic intermediate inputs for a specific country.
    get_monthly_capital_inputs(country_name)
        Returns the monthly capital inputs for a specific country.
    get_gfcf_column(country_name)
        Returns the Gross Fixed Capital Formation (GFCF) column for a specific country.
    get_monthly_capital_inputs_domestic(country_name)
        Returns the monthly domestic capital inputs for a specific country.
    get_monthly_value_added(country_name)
        Returns the monthly value added for a specific country.
    get_monthly_taxes_less_subsidies(country_name)
        Returns the monthly taxes less subsidies for a specific country.
    get_taxes_less_subsidies_rates(country_name)
        Returns the taxes less subsidies rates for a specific country.
    get_monthly_hh_consumption(country_name)
        Returns the monthly household consumption for a specific country.
    get_monthly_hh_consumption_domestic(country_name)
        Returns the monthly domestic household consumption for a specific country.
    get_hh_consumption_weights(country_name)
        Returns the household consumption weights for a specific country.
    get_monthly_govt_consumption(country_name)
        Returns the monthly government consumption for a specific country.
    get_monthly_govt_consumption_domestic(country_name)
        Returns the monthly domestic government consumption for a specific country.
    govt_consumption_weights(country_name)
        Returns the government consumption weights for a specific country.
    get_imports(country_name)
        Returns the imports for a specific country.
    """

    def __init__(
        self,
        iot: pd.DataFrame,
        considered_countries: list[str],
        industries: list[str],
        imputed_rents: dict[str, float],
        year: int,
        yearly_factor: float = 4.0,
    ):
        self.iot = iot
        self.considered_countries = considered_countries
        self.industries = industries
        self.imputed_rents = imputed_rents
        self.year = year
        self.investment_matrices = {}
        self.yearly_factor = yearly_factor

        # Normalisation
        # self.normalise_iot()

    @classmethod
    def agg_from_csv(
        cls,
        path: Path,
        pivot_path: Path,
        considered_countries: list[str] | list[Country | str],
        industries: list[str],
        year: int,
        exchange_rates: ExchangeRatesReader,
        imputed_rent_fraction: dict[str, float],
        investment_fractions: dict[Country | str, dict[str, float]],
        yearly_factor: float = 4.0,
        proxy_country_dict: Optional[dict[str | Country, str | Country]] = None,
        aggregation_path: Optional[Literal["All", "Aggregate"]] = None,
    ) -> "ICIOReader":
        if proxy_country_dict is None:
            proxy_country_dict = {}

        # considered_countries = [c.value if isinstance(c, Country) else c for c in considered_countries]

        # This is quite slow, so adding the option of loading it
        if os.path.isfile(pivot_path):
            df = pd.read_csv(pivot_path, index_col=[0, 1], header=[0, 1])
        else:
            df = cls.read_df(path)
            df.to_csv(pivot_path)

        # Get output and value added
        output_df = 1e6 * df.loc[("OUT", "OUT")]
        va_df = 1e6 * df.loc[("VA", "VA")]
        output, value_added = {}, {}
        for c in considered_countries:
            output[c] = max(0.0, output_df.xs(c).sum())
            value_added[c] = max(0.0, va_df.xs(c).sum())

        # Aggregate the IOT
        if aggregation_path is not None:
            aggregation = ICIO_AGGREGATE if aggregation_path == "Aggregate" else ICIO_ALL
        else:
            aggregation = None
        agg_df = cls.aggregate_io(considered_countries, df, aggregation)

        # Isolate-out imputed rents

        avg_imputed_rent_fraction = sum(imputed_rent_fraction.values()) / len(imputed_rent_fraction)

        new_rent_fraction = {}
        for c in considered_countries:
            if c in imputed_rent_fraction.keys():
                new_rent_fraction[c] = imputed_rent_fraction[c]
            else:
                new_rent_fraction[c] = imputed_rent_fraction.get(proxy_country_dict[c], avg_imputed_rent_fraction)

        imputed_rents = {}
        for country_name in considered_countries:
            if country_name in new_rent_fraction.keys():
                imputed_rents[country_name] = (
                    (
                        new_rent_fraction[country_name]
                        * agg_df.at[
                            (country_name, "L"),
                            (country_name, "Household Consumption"),
                        ]
                    )
                    / yearly_factor
                    * exchange_rates.from_usd_to_lcu(country_name, year)
                )
                agg_df.at[(country_name, "L"), (country_name, "Household Consumption")] -= (
                    new_rent_fraction[country_name]
                    * agg_df.at[
                        (country_name, "L"),
                        (country_name, "Household Consumption"),
                    ]
                )
            else:
                imputed_rents[country_name] = None

        agg_df = normalise_iot(
            agg_df,
            considered_countries=considered_countries,
            industries=industries,
            investment_fractions=investment_fractions,
        )

        return cls(
            iot=agg_df,
            considered_countries=considered_countries,
            industries=industries,
            imputed_rents=imputed_rents,
            year=year,
            yearly_factor=yearly_factor,
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
        aggregation: Optional[dict[str, list[str]]] = None,
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
        if aggregation is None:
            aggregation = default_no_agg_dict(df)
        col_level_0 = df.columns.get_level_values(0).unique()
        keep_level_0 = considered_countries + ["ROW", "TOTAL"]
        discard_level_0 = [c for c in col_level_0 if c not in keep_level_0]
        country_agg_dict = {c: "ROW" for c in discard_level_0}
        for c in keep_level_0:
            country_agg_dict[c] = c
        country_agg_dict["VA"] = "TOTAL"
        country_agg_dict["TLS"] = "TOTAL"
        country_agg_dict["OUT"] = "TOTAL"

        # Perform the aggregation
        aggregated = aggregate_df(aggregation, country_agg_dict, df)

        # Cosmetics
        aggregated *= 1e6
        aggregated.index.names = ["Country", "Industry"]
        aggregated.columns.names = ["Country", "Industry"]

        return aggregated

    # def normalise_iot(self) -> None:
    #     """
    #     Normalises the IOT by adjusting value-added.
    #     """
    #     # Remove aggregates
    #     self.iot = self.iot.loc[self.iot.index != ("TOTAL", "Gross Output")]
    #     self.iot = self.iot.loc[:, self.iot.columns.get_level_values(1) != "Gross Output"]
    #     self.iot = self.iot.loc[:, self.iot.columns.get_level_values(0) != "TOTAL"]
    #
    #     # Remove aggregates from non-industry columns
    #     self.iot.loc[
    #         self.iot.index.get_level_values(0) == "TOTAL",
    #         np.logical_not(self.iot.columns.get_level_values(1).isin(self.industries)),
    #     ] = np.nan
    #
    #     # Remove sectors with negative VA
    #     neg_va_sec = self.iot.columns[np.where(self.iot.loc[("TOTAL", "Value Added")] <= 0.0)].values
    #     neg_va_sec = np.array([list(i) for i in neg_va_sec if list(i)[1] in self.industries])
    #     self.iot.loc[neg_va_sec] = 0.0
    #     self.iot.loc[:, neg_va_sec] = 0.0
    #
    #     # Force positive values
    #     self.iot.loc[self.iot.index.get_level_values(1) != "Taxes Less Subsidies"] = np.maximum(
    #         0.0,
    #         self.iot.loc[self.iot.index.get_level_values(1) != "Taxes Less Subsidies"],
    #     )
    #
    #     # Sums-up intermediate inputs into a new row
    #     self.iot.loc[
    #         ("TOTAL", "Intermediate Inputs"),
    #         self.iot.columns.get_level_values(1).isin(self.industries),
    #     ] = self.iot.loc[
    #         (self.iot.index != ("TOTAL", "Value Added"))
    #         & (self.iot.index.get_level_values(1) != "Taxes Less Subsidies"),
    #         self.iot.columns.get_level_values(1).isin(self.industries),
    #     ].sum(
    #         axis=0
    #     )
    #
    #     # Sums-up taxes-less-subsidies into a new row
    #     self.iot.loc[
    #         ("TOTAL", "Taxes Less Subsidies"),
    #         self.iot.columns.get_level_values(1).isin(self.industries),
    #     ] = self.iot.loc[
    #         self.iot.index.get_level_values(1) == "Taxes Less Subsidies",
    #         self.iot.columns.get_level_values(1).isin(self.industries),
    #     ].sum(
    #         axis=0
    #     )
    #     self.iot = self.iot.loc[
    #         np.logical_not(
    #             (self.iot.index.get_level_values(0) != "TOTAL")
    #             & (self.iot.index.get_level_values(1) == "Taxes Less Subsidies")
    #         )
    #     ].copy()
    #
    #     # Adds total output
    #     output = self.iot.loc[self.iot.index.get_level_values(1).isin(self.industries)].sum(axis=1)
    #     self.iot.loc[:, ("TOTAL", "Output")] = np.nan
    #     self.iot.loc[
    #         self.iot.index.get_level_values(1).isin(self.industries),
    #         ("TOTAL", "Output"),
    #     ] = output
    #     self.iot.loc[
    #         ("TOTAL", "Output"),
    #         self.iot.columns.get_level_values(1).isin(self.industries),
    #     ] = output
    #
    #     # Adjust value-added
    #     self.iot.loc[("TOTAL", "Value Added")] = (
    #         self.iot.loc[("TOTAL", "Output")]
    #         - self.iot.loc[("TOTAL", "Intermediate Inputs")]
    #         - self.iot.loc[("TOTAL", "Taxes Less Subsidies")]
    #     )
    #     if not np.all(
    #         self.iot.loc[
    #             ("TOTAL", "Value Added"),
    #             self.iot.columns.get_level_values(1).isin(self.industries),
    #         ].values
    #         >= 0.0
    #     ):
    #         self.iot.loc[("TOTAL", "Value Added")].to_csv("va.csv")
    #         raise ValueError("Negative VA!")
    #
    #     # Split the total GFCF column
    #     for c in self.considered_countries:
    #         ind = self.iot.index.get_level_values(1).isin(self.industries)
    #         self.iot.loc[ind, (c, "Firm Fixed Capital Formation")] = (
    #             self.investment_fractions[c][0] * self.iot.loc[ind, (c, "Fixed Capital Formation")]
    #         )
    #         self.iot.loc[ind, (c, "Household Fixed Capital Formation")] = (
    #             self.investment_fractions[c][1] * self.iot.loc[ind, (c, "Fixed Capital Formation")]
    #         )
    #         self.iot.loc[ind, (c, "Government Consumption")] += (
    #             self.investment_fractions[c][2] * self.iot.loc[ind, (c, "Fixed Capital Formation")]
    #         )
    #         self.iot = self.iot.loc[
    #             :,
    #             np.logical_or(
    #                 self.iot.columns.get_level_values(1) != "Fixed Capital Formation",
    #                 self.iot.columns.get_level_values(0) != c,
    #             ),
    #         ]
    #     self.iot.sort_index(axis=0, inplace=True)
    #     self.iot.sort_index(axis=1, inplace=True)
    #
    #     # # Sums-up intermediate inputs into a new row
    #     # self.iot.loc[
    #     #     ("TOTAL", "Intermediate Inputs"),
    #     #     self.iot.columns.get_level_values(1).isin(self.industries),
    #     # ] = self.iot.loc[
    #     #     (self.iot.index != ("TOTAL", "Value Added"))
    #     #     & (self.iot.index.get_level_values(1) != "Taxes Less Subsidies"),
    #     #     self.iot.columns.get_level_values(1).isin(self.industries),
    #     # ].sum(
    #     #     axis=0
    #     # )
    #     #
    #     # # Sums-up taxes-less-subsidies into a new row
    #     # self.iot.loc[
    #     #     ("TOTAL", "Taxes Less Subsidies"),
    #     #     self.iot.columns.get_level_values(1).isin(self.industries),
    #     # ] = self.iot.loc[
    #     #     self.iot.index.get_level_values(1) == "Taxes Less Subsidies",
    #     #     self.iot.columns.get_level_values(1).isin(self.industries),
    #     # ].sum(
    #     #     axis=0
    #     # )
    #     # self.iot = self.iot.loc[
    #     #     np.logical_not(
    #     #         (self.iot.index.get_level_values(0) != "TOTAL")
    #     #         & (self.iot.index.get_level_values(1) == "Taxes Less Subsidies")
    #     #     )
    #     # ].copy()
    #     #
    #     # # Adds total output
    #     # output = self.iot.loc[self.iot.index.get_level_values(1).isin(self.industries)].sum(axis=1)
    #     # self.iot.loc[:, ("TOTAL", "Output")] = np.nan
    #     # self.iot.loc[
    #     #     self.iot.index.get_level_values(1).isin(self.industries),
    #     #     ("TOTAL", "Output"),
    #     # ] = output
    #     # self.iot.loc[
    #     #     ("TOTAL", "Output"),
    #     #     self.iot.columns.get_level_values(1).isin(self.industries),
    #     # ] = output
    #     #
    #     # # Adjust value-added
    #     # self.iot.loc[("TOTAL", "Value Added")] = (
    #     #     self.iot.loc[("TOTAL", "Output")]
    #     #     - self.iot.loc[("TOTAL", "Intermediate Inputs")]
    #     #     - self.iot.loc[("TOTAL", "Taxes Less Subsidies")]
    #     # )

    def column_allc(self, country_name: str, symbol: str) -> pd.Series:
        considered_countries_row = self.considered_countries + ["ROW"]
        all_cols = [self.iot.loc[col, (country_name, symbol)].loc[self.industries] for col in considered_countries_row]
        return reduce(lambda a, b: a + b, all_cols).fillna(0)

    def get_total_output(self, country_name: str) -> np.ndarray:
        return (self.iot[("TOTAL", "Output")].xs(country_name, axis=0, level=0).loc[self.industries]).fillna(
            0
        ).values / self.yearly_factor

    def get_intermediate_inputs_use(self, country_name: str) -> np.ndarray:
        return (
            reduce(
                lambda a, b: a + b,
                [
                    self.iot.loc[c_prime, country_name].loc[self.industries, self.industries]
                    for c_prime in self.considered_countries + ["ROW"]
                ],
            )
            / self.yearly_factor
        )

    def get_intermediate_inputs_supply(self, country_name: str) -> np.ndarray:
        return (
            reduce(
                lambda a, b: a + b,
                [
                    self.iot.loc[country_name, c_prime].loc[self.industries, self.industries]
                    for c_prime in self.considered_countries + ["ROW"]
                ],
            )
            / self.yearly_factor
        )

    def get_intermediate_inputs_domestic(self, country_name: str) -> np.ndarray:
        c_iot = self.iot.xs(country_name, axis=1, level=0)
        return c_iot.loc[country_name, c_iot.columns.isin(self.industries)] / self.yearly_factor

    def get_capital_inputs(self, country_name: str) -> np.ndarray:
        return self.column_allc(country_name, "Fixed Capital Formation").values / self.yearly_factor

    def get_firm_capital_inputs(self, country_name: str):
        return self.column_allc(country_name, "Firm Fixed Capital Formation").values / self.yearly_factor

    def get_household_capital_inputs(self, country_name: str) -> np.ndarray:
        return self.column_allc(country_name, "Household Fixed Capital Formation").values / self.yearly_factor

    def get_gfcf_column(self, country_name: str) -> np.ndarray:
        return (
            self.iot.loc[
                self.iot.index.get_level_values(1).isin(self.industries),
                (country_name, "Fixed Capital Formation"),
            ].values
            / self.yearly_factor
        )

    def get_capital_inputs_domestic(self, country_name: str) -> np.ndarray:
        return self.iot.loc[country_name, country_name]["Fixed Capital Formation"].values / self.yearly_factor

    def get_value_added(self, country_name: str) -> np.ndarray:
        return (
            self.iot.xs(country_name, axis=1, level=0).loc[("TOTAL", "Value Added"), self.industries].values
            / self.yearly_factor
        )

    def get_value_added_series(self, country_name: str) -> pd.Series:
        return (
            self.iot.xs(country_name, axis=1, level=0).loc[("TOTAL", "Value Added"), self.industries]
            / self.yearly_factor
        )

    def get_taxes_less_subsidies(self, country_name: str) -> np.ndarray:
        return (
            self.iot.xs(country_name, axis=1, level=0).loc[("TOTAL", "Taxes Less Subsidies"), self.industries].values
        ) / self.yearly_factor

    def get_taxes_less_subsidies_rates(self, country_name: str) -> np.ndarray:
        return self.get_taxes_less_subsidies(country_name) / self.get_total_output(country_name)

    def get_hh_consumption(self, country_name: str) -> np.ndarray:
        return self.column_allc(country_name, "Household Consumption").values / self.yearly_factor

    def get_hh_consumption_domestic(self, country_name: str) -> np.ndarray:
        return self.iot.loc[country_name, (country_name, "Household Consumption")].values / self.yearly_factor

    def get_hh_consumption_weights(self, country_name: str) -> np.ndarray:
        hh_cons = self.get_hh_consumption(country_name)
        return hh_cons / hh_cons.sum()

    def get_govt_consumption(self, country_name: str) -> np.ndarray:
        return self.column_allc(country_name, "Government Consumption").values / self.yearly_factor

    def get_govt_consumption_domestic(self, country_name: str) -> np.ndarray:
        return self.iot.loc[country_name, (country_name, "Government Consumption")].values / self.yearly_factor

    def govt_consumption_weights(self, country_name: str) -> np.ndarray:
        gov_cons = self.get_govt_consumption(country_name)
        return gov_cons / gov_cons.sum()

    def get_imports(self, country_name: str) -> pd.Series:
        considered_countries_row = self.considered_countries + ["ROW"]
        imports = reduce(
            lambda a, b: a + b,
            (self.iot.loc[c2, country_name].sum(axis=1) for c2 in considered_countries_row if c2 != country_name),
        )
        return imports.loc[self.industries] / self.yearly_factor

    def get_exports(self, country_name: str) -> pd.Series:
        considered_countries_row = self.considered_countries + ["ROW"]
        exports = reduce(
            lambda a, b: a + b,
            (self.iot.loc[country_name, c2].sum(axis=1) for c2 in considered_countries_row if c2 != country_name),
        )
        return exports.loc[self.industries] / self.yearly_factor

    def get_trade(self, start_country: str, end_country: str) -> pd.Series:
        return self.iot.loc[start_country, end_country].sum(axis=1).loc[self.industries] / self.yearly_factor

    def get_origin_trade_proportions(self) -> pd.DataFrame:
        trade_proportions = {
            "start_country": [],
            "end_country": [],
            "industry": [],
            "value": [],
        }
        for end_country in self.considered_countries + ["ROW"]:
            if end_country == "ROW":
                imports_total = self.get_imports(end_country)
            else:
                imports_total = self.get_imports(end_country) + self.get_trade(end_country, end_country)
            for start_country in self.considered_countries + ["ROW"]:
                trade_proportions["start_country"] += [start_country] * len(self.industries)
                trade_proportions["end_country"] += [end_country] * len(self.industries)
                trade_proportions["industry"] += list(range(len(self.industries)))
                if start_country == end_country == "ROW":
                    trade_proportions["value"] += list(np.zeros(len(self.industries)))
                else:
                    trade_proportions["value"] += list(
                        (self.get_trade(start_country, end_country) / imports_total).values
                    )
        return pd.DataFrame(trade_proportions).set_index(["start_country", "end_country", "industry"]).sort_index()

    def get_destination_trade_proportions(self) -> pd.DataFrame:
        trade_proportions = {
            "start_country": [],
            "end_country": [],
            "industry": [],
            "value": [],
        }
        for start_country in self.considered_countries + ["ROW"]:
            if start_country == "ROW":
                exports_total = self.get_exports(start_country)
            else:
                exports_total = self.get_exports(start_country) + self.get_trade(start_country, start_country)
            for end_country in self.considered_countries + ["ROW"]:
                if start_country == end_country == "ROW":
                    trade_proportions["value"] += list(np.zeros(len(self.industries)))
                else:
                    trade_proportions["value"] += list(
                        (self.get_trade(start_country, end_country) / exports_total).values
                    )
                trade_proportions["start_country"] += [start_country] * len(self.industries)
                trade_proportions["end_country"] += [end_country] * len(self.industries)
                trade_proportions["industry"] += list(range(len(self.industries)))
        return pd.DataFrame(trade_proportions).set_index(["start_country", "end_country", "industry"]).sort_index()

    # def get_trade_proportions(self) -> pd.DataFrame:
    #     trade_proportions = {"start_country": [], "end_country": [], "industry": [], "value": []}
    #     for end_country in self.considered_countries + ["ROW"]:
    #         if end_country == "ROW":
    #             imports_total = self.get_imports(end_country)
    #         else:
    #             imports_total = self.get_imports(end_country) + self.get_trade(end_country, end_country)
    #         for start_country in self.considered_countries + ["ROW"]:
    #             if start_country == end_country == "ROW":
    #                 continue
    #             trade_proportions["start_country"] += [start_country] * len(self.industries)
    #             trade_proportions["end_country"] += [end_country] * len(self.industries)
    #             trade_proportions["industry"] += list(range(len(self.industries)))
    #             trade_proportions["value"] += list((self.get_trade(start_country, end_country) / imports_total).values)
    #     return pd.DataFrame(trade_proportions).set_index(["start_country", "end_country", "industry"])

    def get_intermediate_inputs_matrix(self, country_name: str) -> pd.DataFrame:
        total_output = self.get_total_output(country_name)
        total_monthly_intermediate_inputs = self.get_intermediate_inputs_use(country_name)
        return total_output[None, :] / total_monthly_intermediate_inputs  # noqa

    def get_capital_inputs_matrix(
        self,
        country_name: str,
        capital_stock: np.ndarray,
    ) -> pd.DataFrame:
        norm_investment_matrix = self.investment_matrices[country_name].copy()
        norm_investment_matrix /= norm_investment_matrix.sum(axis=0)
        cap_inputs_matrix = (self.get_total_output(country_name) / capital_stock) / norm_investment_matrix
        return cap_inputs_matrix.xs(country_name, axis=0, level=0).xs(country_name, axis=1, level=0).fillna(np.inf)

    def get_capital_inputs_depreciation(
        self,
        country_name: str,
        capital_compensation: np.ndarray,
    ) -> pd.DataFrame:
        total_output = self.get_total_output(country_name)
        gfcf = self.get_firm_capital_inputs(country_name)
        investment_matrix = np.array([gfcf for _ in range(len(capital_compensation))]).T
        norm_investment_matrix = investment_matrix / investment_matrix.sum(axis=0)
        norm_investment_matrix *= (capital_compensation / total_output)[None, :]
        return (
            pd.DataFrame(
                data=norm_investment_matrix,
                index=pd.Index(self.industries, name="Industries"),
                columns=pd.Index(self.industries, name="Industries"),
            )
            / self.yearly_factor
        )


def normalise_iot(
    iot: pd.DataFrame,
    industries: list[str],
    considered_countries: list[Country] | list[str],
    investment_fractions: dict[str | Country, dict[str, float]],
) -> pd.DataFrame:
    """
    Normalises the IOT by adjusting value-added.
    """
    # Remove aggregates
    iot = iot.loc[iot.index != ("TOTAL", "Gross Output")]
    iot = iot.loc[:, iot.columns.get_level_values(1) != "Gross Output"]
    iot = iot.loc[:, iot.columns.get_level_values(0) != "TOTAL"]

    # Remove aggregates from non-industry columns
    iot.loc[
        iot.index.get_level_values(0) == "TOTAL",
        np.logical_not(iot.columns.get_level_values(1).isin(industries)),
    ] = np.nan

    # Remove sectors with negative VA
    neg_va_sec = iot.columns[np.where(iot.loc[("TOTAL", "Value Added")] <= 0.0)].values
    neg_va_sec = np.array([list(i) for i in neg_va_sec if list(i)[1] in industries])
    iot.loc[neg_va_sec] = 0.0
    iot.loc[:, neg_va_sec] = 0.0

    # Force positive values
    iot.loc[iot.index.get_level_values(1) != "Taxes Less Subsidies"] = np.maximum(
        0.0,
        iot.loc[iot.index.get_level_values(1) != "Taxes Less Subsidies"],
    )

    # Sums-up intermediate inputs into a new row
    iot.loc[
        ("TOTAL", "Intermediate Inputs"),
        iot.columns.get_level_values(1).isin(industries),
    ] = iot.loc[
        (iot.index != ("TOTAL", "Value Added")) & (iot.index.get_level_values(1) != "Taxes Less Subsidies"),
        iot.columns.get_level_values(1).isin(industries),
    ].sum(axis=0)

    # Sums-up taxes-less-subsidies into a new row
    iot.loc[
        ("TOTAL", "Taxes Less Subsidies"),
        iot.columns.get_level_values(1).isin(industries),
    ] = iot.loc[
        iot.index.get_level_values(1) == "Taxes Less Subsidies",
        iot.columns.get_level_values(1).isin(industries),
    ].sum(axis=0)
    iot = iot.loc[
        np.logical_not(
            (iot.index.get_level_values(0) != "TOTAL") & (iot.index.get_level_values(1) == "Taxes Less Subsidies")
        )
    ].copy()

    # Adds total output
    output = iot.loc[iot.index.get_level_values(1).isin(industries)].sum(axis=1)
    iot.loc[:, ("TOTAL", "Output")] = np.nan
    iot.loc[
        iot.index.get_level_values(1).isin(industries),
        ("TOTAL", "Output"),
    ] = output
    iot.loc[
        ("TOTAL", "Output"),
        iot.columns.get_level_values(1).isin(industries),
    ] = output

    # Adjust value-added
    iot.loc[("TOTAL", "Value Added")] = (
        iot.loc[("TOTAL", "Output")]
        - iot.loc[("TOTAL", "Intermediate Inputs")]
        - iot.loc[("TOTAL", "Taxes Less Subsidies")]
    )
    if not np.all(
        iot.loc[
            ("TOTAL", "Value Added"),
            iot.columns.get_level_values(1).isin(industries),
        ].values
        >= 0.0
    ):
        iot.loc[("TOTAL", "Value Added")].to_csv("va.csv")
        raise ValueError("Negative VA!")

    # Split the total GFCF column
    for c in considered_countries:
        ind = iot.index.get_level_values(1).isin(industries)
        iot.loc[ind, (c, "Firm Fixed Capital Formation")] = (
            investment_fractions[c]["Firm"] * iot.loc[ind, (c, "Fixed Capital Formation")]
        )
        iot.loc[ind, (c, "Household Fixed Capital Formation")] = (
            investment_fractions[c]["Household"] * iot.loc[ind, (c, "Fixed Capital Formation")]
        )
        iot.loc[ind, (c, "Government Consumption")] += (
            investment_fractions[c]["Government"] * iot.loc[ind, (c, "Fixed Capital Formation")]
        )
        iot = iot.loc[
            :,
            np.logical_or(
                iot.columns.get_level_values(1) != "Fixed Capital Formation",
                iot.columns.get_level_values(0) != c,
            ),
        ]
    iot.sort_index(axis=0, inplace=True)
    iot.sort_index(axis=1, inplace=True)

    return iot


def default_no_agg_dict(df: pd.DataFrame) -> dict[str, list[str]]:
    ind_cols = df.columns.get_level_values(1).unique()
    ind_rows = df.index.get_level_values(1).unique()

    names = set(ind_rows).union(set(ind_cols))
    return {c: [c] for c in names}
