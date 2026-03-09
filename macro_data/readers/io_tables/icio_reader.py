"""
This module provides functionality for reading and processing OECD Inter-Country Input
Output (ICIO) tables. It handles the complex task of managing multi-country,
multi-industry economic relationships and transforming them into a format suitable
for economic modeling.

The module centers around the ICIOReader class, which processes raw ICIO data and
provides methods to:
1. Read and aggregate input-output relationships
2. Handle country-specific data transformations
3. Calculate economic indicators and flows
4. Convert between different time frequencies

Key features:
- Support for multiple countries and industries
- Flexible aggregation of sectors and regions
- Exchange rate conversions
- Time frequency adjustments
- Trade flow calculations

Example:
    ```python
    from pathlib import Path
    from macro_data.readers.io_tables.icio_reader import ICIOReader

    # Initialize reader with raw data
    reader = ICIOReader.agg_from_csv(
        path=Path("icio_data.csv"),
        pivot_path=Path("pivoted_data.csv"),
        considered_countries=["FRA", "DEU"],
        industries=["C10T12", "C13T15"],
        year=2018,
        exchange_rates=exchange_rates_reader,
        imputed_rent_fraction={"FRA": 0.2, "DEU": 0.18},
        investment_fractions=investment_data
    )

    # Get processed data
    fra_output = reader.get_total_output("FRA")
    fra_imports = reader.get_imports("FRA")
    ```
"""

import os
import re
from functools import reduce
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.configuration.region import Region
from macro_data.readers.economic_data.exchange_rates import ExchangeRatesReader
from macro_data.readers.io_tables.industries import AGGREGATED_INDUSTRIES
from macro_data.readers.io_tables.mappings import ICIO_AGGREGATE, ICIO_ALL
from macro_data.readers.io_tables.util import aggregate_df


class ICIOReader:
    """Reader and processor for OECD Inter-Country Input Output Tables.

    This class reads and processes OECD ICIO tables, providing methods to:
    1. Aggregate and normalize input-output relationships
    2. Calculate trade flows and proportions
    3. Extract industry-specific metrics
    4. Convert annual values to sub-annual frequency

    The reader handles:
    - Multi-country input-output relationships
    - Industry aggregation and mapping
    - Trade flow calculations
    - Value-added computations
    - Capital formation and investment
    - Government and household consumption
    - Exchange rate conversions

    Attributes:
        iot (pd.DataFrame): The input-output table
        considered_countries (list[str]): Countries included in analysis
        industries (list[str]): Industries tracked in the model
        imputed_rents (dict[str, float]): Imputed rents by country
        year (int): Reference year for the data
        investment_matrices (dict): Investment allocation matrices
        yearly_factor (float): Factor to convert annual to sub-annual values

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
        """Initialize the ICIOReader.

        Args:
            iot (pd.DataFrame): Input-output table
            considered_countries (list[str]): Countries to include
            industries (list[str]): Industries to track
            imputed_rents (dict[str, float]): Imputed rents by country
            year (int): Reference year
            yearly_factor (float, optional): Factor to convert annual to sub-annual values.
                Defaults to 4.0 (quarterly).
        """
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
        aggregation_type: Optional[Literal["All", "Aggregate"]] = None,
    ) -> "ICIOReader":
        """Create an ICIOReader instance from CSV data with aggregation.

        This factory method reads raw ICIO data, performs necessary aggregations
        and normalizations, and returns a configured ICIOReader instance.

        Args:
            path (Path): Path to raw ICIO CSV file
            pivot_path (Path): Path to save/load pivoted data
            considered_countries (list[str | Country]): Countries to include
            industries (list[str]): Industries to track
            year (int): Reference year
            exchange_rates (ExchangeRatesReader): Exchange rate data
            imputed_rent_fraction (dict[str, float]): Rent fractions by country
            investment_fractions (dict[Country | str, dict[str, float]]): Investment splits
            yearly_factor (float, optional): Annual to sub-annual conversion.
                Defaults to 4.0 (quarterly).
            proxy_country_dict (Optional[dict]): Country mapping for missing data
            aggregation_type (Optional[Literal["All", "Aggregate"]]): Aggregation method

        Returns:
            ICIOReader: Configured reader instance
        """
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
        if aggregation_type is not None:
            aggregation = ICIO_AGGREGATE if aggregation_type == "Aggregate" else ICIO_ALL
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

    def column_allc(self, country_name: str, symbol: str) -> pd.Series:
        """Sum columns across all countries for a specific symbol.

        Aggregates values across all countries (including ROW) for a given
        country and symbol combination.

        Args:
            country_name (str): Target country
            symbol (str): Column identifier (e.g., "Household Consumption")

        Returns:
            pd.Series: Summed values across all countries
        """
        considered_countries_row = self.considered_countries + ["ROW"]
        all_cols = [self.iot.loc[col, (country_name, symbol)].loc[self.industries] for col in considered_countries_row]
        return reduce(lambda a, b: a + b, all_cols).fillna(0)

    def get_total_output(self, country_name: str) -> np.ndarray:
        """Get total output by industry for a country.

        Args:
            country_name (str): Country to get output for

        Returns:
            np.ndarray: Total output values by industry, converted to sub-annual frequency
        """
        return (
            (self.iot[("TOTAL", "Output")].xs(country_name, axis=0, level=0).loc[self.industries]).fillna(0).values
            / self.yearly_factor
        )

    def get_total_output_series(self, country_name: str) -> pd.Series:
        """Get total output by industry as a pandas Series.

        Args:
            country_name (str): Country to get output for

        Returns:
            pd.Series: Total output values by industry, converted to sub-annual frequency
        """
        return self.iot[("TOTAL", "Output")].xs(country_name, axis=0, level=0).loc[self.industries] / self.yearly_factor

    def get_output_shares_dict(self, country_name: str) -> dict[str, pd.Series]:
        """Calculate output shares for aggregated industry sectors.

        Computes the proportion of output each sub-sector contributes to its
        aggregated sector total.

        Args:
            country_name (str): Country to analyze

        Returns:
            dict[str, pd.Series]: Mapping of aggregate sectors to sub-sector shares
        """
        output_shares_dict = {}
        output_series = self.get_total_output_series(country_name)

        industry_dict = update_dictionary(self.industries, ICIO_AGGREGATE)

        for agg_sector in AGGREGATED_INDUSTRIES:
            sub_sectors = industry_dict[agg_sector]
            sub_sector_outputs = output_series.loc[output_series.index.intersection(sub_sectors)]
            total_output = sub_sector_outputs.sum()
            output_shares_dict[agg_sector] = sub_sector_outputs / total_output

        return output_shares_dict

    def get_consumption_shares_series(self, country_name: str) -> pd.Series:
        """Calculate consumption shares for each industry.

        Computes the proportion of total consumption each industry represents
        within its aggregate sector.

        Args:
            country_name (str): Country to analyze

        Returns:
            pd.Series: Industry-level consumption shares
        """
        hh_cons = self.get_hh_consumption_series(country_name)

        df = hh_cons.reset_index()
        df.columns = ["Industry", "Consumption"]

        # Step 3: Map each sub-sector to its aggregate sector
        inverse_dict = self.get_inverse_updated_dictionary()
        df["AggregateSector"] = df["Industry"].map(inverse_dict)

        # Step 4: Calculate total output per aggregate sector
        agg_cons = df.groupby("AggregateSector")["Consumption"].sum().reset_index()
        agg_cons.columns = ["AggregateSector", "AggregateConsumption"]

        # Step 5: Calculate the consumption share
        df = df.merge(agg_cons, on="AggregateSector", how="left")
        df["ConsumptionShare"] = df["Consumption"] / df["AggregateConsumption"]
        return df.set_index("Industry")["ConsumptionShare"]

    def get_intermediate_inputs_use(self, country_name: str) -> np.ndarray:
        """Get intermediate inputs used by each industry.

        Computes the total intermediate inputs used by each industry in the country,
        including both domestic and imported inputs.

        Args:
            country_name (str): Country to analyze

        Returns:
            np.ndarray: Matrix of intermediate input usage, converted to sub-annual frequency
        """
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
        """Get intermediate inputs supplied by each industry.

        Computes the total intermediate inputs supplied by each industry in the country
        to all other industries, both domestic and foreign.

        Args:
            country_name (str): Country to analyze

        Returns:
            np.ndarray: Matrix of intermediate input supply, converted to sub-annual frequency
        """
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
        """Get domestic intermediate inputs for each industry.

        Computes the intermediate inputs used within the same country,
        excluding imports.

        Args:
            country_name (str): Country to analyze

        Returns:
            np.ndarray: Matrix of domestic intermediate inputs, converted to sub-annual frequency
        """
        c_iot = self.iot.xs(country_name, axis=1, level=0)
        return c_iot.loc[country_name, c_iot.columns.isin(self.industries)] / self.yearly_factor

    def get_capital_inputs(self, country_name: str) -> np.ndarray:
        """Get total capital inputs by industry.

        Computes the total fixed capital formation (investment) from all sources
        for each industry.

        Args:
            country_name (str): Country to analyze

        Returns:
            np.ndarray: Capital input values by industry, converted to sub-annual frequency
        """
        return self.column_allc(country_name, "Fixed Capital Formation").values / self.yearly_factor

    def get_firm_capital_inputs(self, country_name: str) -> np.ndarray:
        """Get capital inputs used by firms.

        Computes the capital inputs used by firms in each industry,
        excluding household and government capital formation.

        Args:
            country_name (str): Country to analyze

        Returns:
            np.ndarray: Firm capital inputs by industry, converted to sub-annual frequency
        """
        return self.column_allc(country_name, "Firm Fixed Capital Formation").values / self.yearly_factor

    def get_household_capital_inputs(self, country_name: str) -> np.ndarray:
        """Get capital inputs used by households.

        Computes household fixed capital formation (e.g., housing investment)
        by industry.

        Args:
            country_name (str): Country to analyze

        Returns:
            np.ndarray: Household capital inputs by industry, converted to sub-annual frequency
        """
        return self.column_allc(country_name, "Household Fixed Capital Formation").values / self.yearly_factor

    def get_gfcf_column(self, country_name: str) -> np.ndarray:
        """Get Gross Fixed Capital Formation (GFCF) column.

        Retrieves the raw GFCF values for each industry before splitting
        between firms, households, and government.

        Args:
            country_name (str): Country to analyze

        Returns:
            np.ndarray: GFCF values by industry, converted to sub-annual frequency
        """
        return (
            self.iot.loc[
                self.iot.index.get_level_values(1).isin(self.industries),
                (country_name, "Fixed Capital Formation"),
            ].values
            / self.yearly_factor
        )

    def get_capital_inputs_domestic(self, country_name: str) -> np.ndarray:
        """Get domestic capital inputs by industry.

        Computes fixed capital formation using only domestically produced
        capital goods.

        Args:
            country_name (str): Country to analyze

        Returns:
            np.ndarray: Domestic capital inputs by industry, converted to sub-annual frequency
        """
        return self.iot.loc[country_name, country_name]["Fixed Capital Formation"].values / self.yearly_factor

    def get_value_added(self, country_name: str) -> np.ndarray:
        """Get value added by industry.

        Computes the value added (contribution to GDP) for each industry
        in the specified country.

        Args:
            country_name (str): Country to analyze

        Returns:
            np.ndarray: Value added by industry, converted to sub-annual frequency
        """
        return (
            self.iot.xs(country_name, axis=1, level=0).loc[("TOTAL", "Value Added"), self.industries].values
            / self.yearly_factor
        )

    def get_value_added_series(self, country_name: str) -> pd.Series:
        """Get value added by industry as a pandas Series.

        Computes the value added (contribution to GDP) for each industry,
        returned as a labeled Series.

        Args:
            country_name (str): Country to analyze

        Returns:
            pd.Series: Value added by industry, converted to sub-annual frequency
        """
        return (
            self.iot.xs(country_name, axis=1, level=0).loc[("TOTAL", "Value Added"), self.industries]
            / self.yearly_factor
        )

    def get_taxes_less_subsidies(self, country_name: str) -> np.ndarray:
        """Get net taxes (taxes less subsidies) by industry.

        Computes the net taxes (taxes minus subsidies) for each industry
        in the specified country.

        Args:
            country_name (str): Country to analyze

        Returns:
            np.ndarray: Net taxes by industry, converted to sub-annual frequency
        """
        return (
            self.iot.xs(country_name, axis=1, level=0)
            .loc[("TOTAL", "Taxes Less Subsidies"), self.industries]
            .fillna(0)
            .values
        ) / self.yearly_factor

    def get_taxes_less_subsidies_rates(self, country_name: str) -> np.ndarray:
        """Calculate net tax rates by industry.

        Computes the ratio of net taxes (taxes minus subsidies) to total output
        for each industry.

        Args:
            country_name (str): Country to analyze

        Returns:
            np.ndarray: Net tax rates by industry
        """
        return self.get_taxes_less_subsidies(country_name) / self.get_total_output(country_name)

    def get_hh_consumption(self, country_name: str) -> np.ndarray:
        """Get total household consumption by industry.

        Computes household consumption from all sources (domestic and imported)
        for each industry.

        Args:
            country_name (str): Country to analyze

        Returns:
            np.ndarray: Household consumption by industry, converted to sub-annual frequency
        """
        return self.column_allc(country_name, "Household Consumption").values / self.yearly_factor

    def get_hh_consumption_series(self, country_name: str) -> pd.Series:
        """Get household consumption by industry as a pandas Series.

        Computes total household consumption (domestic and imported) by industry,
        returned as a labeled Series.

        Args:
            country_name (str): Country to analyze

        Returns:
            pd.Series: Household consumption by industry, converted to sub-annual frequency
        """
        return self.column_allc(country_name, "Household Consumption") / self.yearly_factor

    def get_hh_consumption_domestic(self, country_name: str) -> np.ndarray:
        """Get domestic household consumption by industry.

        Computes household consumption of domestically produced goods and services
        for each industry.

        Args:
            country_name (str): Country to analyze

        Returns:
            np.ndarray: Domestic household consumption by industry, converted to sub-annual frequency
        """
        return self.iot.loc[country_name, (country_name, "Household Consumption")].values / self.yearly_factor

    def get_hh_consumption_weights(self, country_name: str) -> np.ndarray:
        """Calculate household consumption weights by industry.

        Computes the proportion of total household consumption represented
        by each industry.

        Args:
            country_name (str): Country to analyze

        Returns:
            np.ndarray: Household consumption weights by industry
        """
        hh_cons = self.get_hh_consumption(country_name)
        return hh_cons / hh_cons.sum()

    def get_govt_consumption(self, country_name: str) -> np.ndarray:
        """Get total government consumption by industry.

        Computes government consumption from all sources (domestic and imported)
        for each industry.

        Args:
            country_name (str): Country to analyze

        Returns:
            np.ndarray: Government consumption by industry, converted to sub-annual frequency
        """
        return self.column_allc(country_name, "Government Consumption").values / self.yearly_factor

    def get_govt_consumption_domestic(self, country_name: str) -> np.ndarray:
        """Get domestic government consumption by industry.

        Computes government consumption of domestically produced goods and services
        for each industry.

        Args:
            country_name (str): Country to analyze

        Returns:
            np.ndarray: Domestic government consumption by industry, converted to sub-annual frequency
        """
        return self.iot.loc[country_name, (country_name, "Government Consumption")].values / self.yearly_factor

    def govt_consumption_weights(self, country_name: str) -> np.ndarray:
        """Calculate government consumption weights by industry.

        Computes the proportion of total government consumption represented
        by each industry.

        Args:
            country_name (str): Country to analyze

        Returns:
            np.ndarray: Government consumption weights by industry
        """
        gov_cons = self.get_govt_consumption(country_name)
        return gov_cons / gov_cons.sum()

    def get_imports(self, country_name: str) -> pd.Series:
        """Calculate total imports by industry.

        Computes the sum of imports from all other countries (including ROW)
        for each industry.

        Args:
            country_name (str): Country to analyze

        Returns:
            pd.Series: Import values by industry, converted to sub-annual frequency
        """
        considered_countries_row = self.considered_countries + ["ROW"]
        imports = reduce(
            lambda a, b: a + b,
            (self.iot.loc[c2, country_name].sum(axis=1) for c2 in considered_countries_row if c2 != country_name),
        )
        return imports.loc[self.industries] / self.yearly_factor

    def get_exports(self, country_name: str) -> pd.Series:
        """Calculate total exports by industry.

        Computes the sum of exports to all other countries (including ROW)
        for each industry.

        Args:
            country_name (str): Country to analyze

        Returns:
            pd.Series: Export values by industry, converted to sub-annual frequency
        """
        considered_countries_row = self.considered_countries + ["ROW"]
        exports = reduce(
            lambda a, b: a + b,
            (self.iot.loc[country_name, c2].sum(axis=1) for c2 in considered_countries_row if c2 != country_name),
        )
        return exports.loc[self.industries] / self.yearly_factor

    def get_trade(self, start_country: str, end_country: str) -> pd.Series:
        """Calculate bilateral trade flows between two countries.

        Computes the trade flow from the start country to the end country
        for each industry.

        Args:
            start_country (str): Exporting country
            end_country (str): Importing country

        Returns:
            pd.Series: Trade values by industry, converted to sub-annual frequency
        """
        return self.iot.loc[start_country, end_country].sum(axis=1).loc[self.industries] / self.yearly_factor

    def get_origin_trade_proportions(self) -> pd.DataFrame:
        """Calculate trade proportions from origin country perspective.

        Computes the fraction of each industry's imports that comes from each
        source country, including domestic production.

        Returns:
            pd.DataFrame: Multi-indexed DataFrame with trade proportions
                Index levels: [start_country, end_country, industry]
        """
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
        """Calculate trade proportions from destination country perspective.

        Computes the fraction of each industry's exports that goes to each
        destination country, including domestic consumption.

        Returns:
            pd.DataFrame: Multi-indexed DataFrame with trade proportions
                Index levels: [start_country, end_country, industry]
        """
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

    def get_intermediate_inputs_matrix(self, country_name: str) -> pd.DataFrame:
        """Calculate the intermediate inputs coefficient matrix.

        Computes the technical coefficients matrix showing the amount of
        intermediate inputs required per unit of output for each industry.

        Args:
            country_name (str): Country to analyze

        Returns:
            pd.DataFrame: Matrix of input-output coefficients
        """
        total_output = self.get_total_output(country_name)
        total_monthly_intermediate_inputs = self.get_intermediate_inputs_use(country_name)
        return total_output[None, :] / total_monthly_intermediate_inputs  # noqa

    def get_capital_inputs_matrix(
        self,
        country_name: str,
        capital_stock: np.ndarray,
    ) -> pd.DataFrame:
        """Calculate the capital inputs coefficient matrix.

        Computes the matrix showing how capital from each industry is used
        in the production processes of other industries, normalized by
        capital stock.

        Args:
            country_name (str): Country to analyze
            capital_stock (np.ndarray): Current capital stock by industry

        Returns:
            pd.DataFrame: Matrix of capital input coefficients
        """
        norm_investment_matrix = self.investment_matrices[country_name].copy()
        norm_investment_matrix /= norm_investment_matrix.sum(axis=0)
        cap_inputs_matrix = (self.get_total_output(country_name) / capital_stock) / norm_investment_matrix
        return cap_inputs_matrix.xs(country_name, axis=0, level=0).xs(country_name, axis=1, level=0).fillna(np.inf)

    def get_capital_inputs_depreciation(
        self,
        country_name: str,
        capital_compensation: np.ndarray,
    ) -> pd.DataFrame:
        """Calculate the capital depreciation matrix.

        Computes the matrix of depreciation rates for capital used in each industry,
        normalized by total output and adjusted for capital compensation.

        Args:
            country_name (str): Country to analyze
            capital_compensation (np.ndarray): Capital compensation by industry

        Returns:
            pd.DataFrame: Matrix of capital depreciation rates, converted to sub-annual frequency
        """
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

    def get_updated_dictionary(self) -> dict:
        """Get updated industry aggregation dictionary.

        Returns a mapping of aggregate sectors to their constituent
        sub-sectors based on the current industry list.

        Returns:
            dict: Mapping of aggregate sectors to lists of sub-sectors
        """
        dictionary = ICIO_AGGREGATE
        return update_dictionary(self.industries, dictionary)

    def get_inverse_updated_dictionary(self) -> dict:
        """Get inverse industry aggregation dictionary.

        Returns a mapping of sub-sectors to their parent aggregate sectors,
        inverting the standard aggregation hierarchy.

        Returns:
            dict: Mapping of sub-sectors to their aggregate sectors
        """
        updated_dict = self.get_updated_dictionary()

        inverse_dict = {}
        # each value in the updated dictionary is a list of sub-sectors
        # we want to create a dictionary where each sub-sector is a key, and the value is the aggregate sector
        for agg_sector, sub_sectors in updated_dict.items():
            for sub_sector in sub_sectors:
                inverse_dict[sub_sector] = agg_sector
        return inverse_dict


def normalise_iot(
    iot: pd.DataFrame,
    industries: list[str],
    considered_countries: list[Country] | list[str],
    investment_fractions: dict[str | Country, dict[str, float]],
) -> pd.DataFrame:
    """
    Normalize input-output table by adjusting value-added and investment allocations.

    This function ensures the input-output relationships are properly balanced by:
    1. Adjusting value-added components
    2. Allocating investment across industries
    3. Ensuring consistency of trade flows

    The normalization is essential for maintaining accounting identities and
    ensuring the economic relationships represented in the table are coherent.

    Args:
        iot (pd.DataFrame): Raw input-output table
        industries (list[str]): Industries to process
        considered_countries (list[Country | str]): Countries to include
        investment_fractions (dict): Investment allocation ratios

    Returns:
        pd.DataFrame: Normalized input-output table
    """
    # Remove aggregates
    iot = iot.loc[iot.index != ("TOTAL", "Gross Output")]
    iot = iot.loc[iot.index != ("TOTAL", "Output")]
    iot = iot.loc[:, iot.columns.get_level_values(1) != "Gross Output"]
    iot = iot.loc[:, iot.columns.get_level_values(1) != "Output"]
    iot = iot.loc[:, iot.columns.get_level_values(0) != "TOTAL"]

    # Remove aggregates from non-industry columns
    industry_columns = iot.columns.get_level_values(1).isin(industries)
    iot.loc[
        iot.index.get_level_values(0) == "TOTAL",
        np.logical_not(industry_columns),
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
        industry_columns,
    ] = iot.loc[
        (iot.index != ("TOTAL", "Value Added")) & (iot.index.get_level_values(1) != "Taxes Less Subsidies"),
        industry_columns,
    ].sum(axis=0)

    # Sums-up taxes-less-subsidies into a new row
    iot.loc[
        ("TOTAL", "Taxes Less Subsidies"),
        industry_columns,
    ] = iot.loc[
        iot.index.get_level_values(1) == "Taxes Less Subsidies",
        industry_columns,
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

    iot = split_gfcf_column(considered_countries, industries, investment_fractions, iot)
    iot.sort_index(axis=0, inplace=True)
    iot.sort_index(axis=1, inplace=True)

    return iot


def split_gfcf_column(
    considered_countries: list[Country | Region],
    industries: list[str],
    investment_fractions: dict[str | Country, dict[str, float]],
    iot: pd.DataFrame,
):
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
    return iot


def default_no_agg_dict(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Create a default no-aggregation dictionary for industries.

    This utility function creates a mapping where each industry maps to itself,
    effectively specifying no aggregation should occur. It's useful as a
    fallback when no specific aggregation scheme is provided.

    Args:
        df (pd.DataFrame): Input-output table with industry information

    Returns:
        dict[str, list[str]]: One-to-one mapping of industries
    """
    ind_cols = df.columns.get_level_values(1).unique()
    ind_rows = df.index.get_level_values(1).unique()

    names = set(ind_rows).union(set(ind_cols))
    return {c: [c] for c in names}


def update_dictionary(industries: list[str], dictionary: dict) -> dict:
    """
    Update industry mapping dictionary to match specified industries.

    This utility function ensures the industry mapping dictionary is consistent
    with the list of industries being used in the analysis. It's particularly
    useful when working with different industry aggregation levels.

    Args:
        industries (list[str]): Target industry list
        dictionary (dict): Original mapping dictionary

    Returns:
        dict: Updated mapping dictionary
    """
    pattern = re.compile(r"[A-Z]\d{2}[a-z]")
    new_industries = [ind for ind in industries if pattern.match(ind)]

    for ind in new_industries:
        key = ind[0]
        if key in dictionary:
            dictionary[key].append(ind)
        else:
            dictionary[key] = [ind]
    return dictionary
