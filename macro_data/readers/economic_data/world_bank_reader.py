"""
Module for reading and processing World Bank economic data.

This module provides functionality to read and analyze various economic indicators
from the World Bank database. It handles a wide range of economic data including
GDP, tax rates, unemployment, inflation, and other key economic indicators.

Key Features:
    - GDP and population statistics
    - Tax rates (VAT, export taxes)
    - Labor market indicators (unemployment, participation rates)
    - Inflation and price indices
    - Government debt data
    - Income inequality measures (Gini coefficients)
    - Financial sector health indicators (NPL ratios)

Example:
    ```python
    from pathlib import Path
    from macro_data.configuration.countries import Country

    # Initialize reader
    reader = WorldBankReader(path=Path("path/to/world_bank_data"))

    # Get GDP data for a country
    gdp = reader.get_historic_gdp(country=Country.USA, year=2020)

    # Get VAT rate
    vat = reader.get_tau_vat(country=Country.GBR, year=2020)
    ```

Note:
    - Uses standardized World Bank data files
    - Handles missing data through proxies and interpolation
    - Supports data pruning for specific date ranges
    - Includes forced values for certain tax rates where data is unavailable
"""

import logging
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.readers.util.prune_util import DataFilterWarning, prune_index

forced_vat = {
    "TWN": 0.05,
    "JPN": 0.1,
    "ESP": 0.21,
    "BRN": 0.0,
    "HKG": 0.0,
    "LAO": 0.0,
    "IDN": 0.0,
    "VNM": 0.0,
    "MMR": 0.0,
    "COL": 0.0,
    "CHL": 0.0,
    "CRI": 0.0,
    "KOR": 0.0,
    "KHM": 0.0,
}


class WorldBankReader:
    """
    Reader class for World Bank economic data.

    This class provides methods to read and process various economic indicators
    from World Bank datasets. It handles data loading, scaling, and provides
    access to a wide range of economic statistics.

    Args:
        path (Path): Path to directory containing World Bank data files

    Attributes:
        data (dict[str, pd.DataFrame]): Dictionary of loaded World Bank datasets
        files_with_codes (dict[str, str]): Mapping of data categories to file names

    Key Methods:
        - GDP and Growth:
            - get_historic_gdp: Get historical GDP values
            - get_current_scaled_gdp: Get scaled current GDP
        - Labor Market:
            - get_unemployment_rate: Get unemployment rates
            - get_participation_rate: Get labor force participation
        - Taxes:
            - get_tau_vat: Get VAT rates
            - get_tau_exp: Get export tax rates
        - Prices and Inflation:
            - get_log_inflation: Get log inflation rates
            - get_inflation: Get raw inflation data
        - Other Indicators:
            - get_gini_coef: Get income inequality measures
            - get_central_gov_debt: Get government debt data
            - get_npl_ratios: Get non-performing loan ratios
    """

    def __init__(self, path: Path):
        """
        Initialize the WorldBankReader with data path.

        Args:
            path (Path): Path to directory containing World Bank data files

        Note:
            - Loads data files specified in files_with_codes
            - Special handling for certain files that don't require row skipping
            - Uses ISO-8859-1 encoding for file reading
        """
        self.files_with_codes = self.get_files_with_codes()

        self.data = {}
        for key in self.files_with_codes.keys():
            if key in [
                "long_term_interest_rates",
                "short_term_interest_rates",
                "gov_debt",
                "ppi",
                "cpi",
                "npl_ratios",
                "inflation_arg",
            ]:
                skiprows = []
            else:
                skiprows = [0, 1, 2, 3]
            self.data[key] = pd.read_csv(
                path / (self.files_with_codes[key] + ".csv"),
                skiprows=skiprows,
                encoding="ISO-8859-1",
            )

    @staticmethod
    def get_files_with_codes() -> dict[str, str]:
        """
        Get mapping of data categories to file names.

        Returns:
            dict[str, str]: Dictionary mapping data categories to their file names,
                           including unemployment, tax rates, GDP, and other indicators

        Note:
            File names follow World Bank API naming conventions:
            - API_* files are direct World Bank indicators
            - Other files are supplementary data sources
        """
        return {
            "unemployment": "API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_4325868",
            "participation": "API_SL.TLF.CACT.NE.ZS_DS2_en_csv_v2_4354787",
            "tau_vat": "API_GC.TAX.GSRV.VA.ZS_DS2_en_csv_v2_4028900",
            "tau_exp": "API_GC.TAX.EXPT.CN_DS2_en_csv_v2_4157140",
            "gini_coefs": "API_SI.POV.GINI_DS2_en_csv_v2_5358360",
            "fertility_rates": "API_SP.DYN.TFRT.IN_DS2_en_csv_v2_4151057",
            "interest_rates_on_govt_debt": "API_FR.INR.RINR_DS2_en_csv_v2_4150781",
            "long_term_interest_rates": "LONG_TERM_IR",
            "short_term_interest_rates": "SHORT_TERM_IR",
            "ppi": "ppi",
            "cpi": "cpi",
            "historic_gdp": "API_NY.GDP.MKTP.CN_DS2_en_csv_v2_5358562",
            "population": "API_SP.POP.TOTL_DS2_en_csv_v2_79",
            "gov_debt": "central_gov_debt",
            "npl_ratios": "npl_ratios",
            "inflation_arg": "inflation_arg",
        }

    def get_central_gov_debt(self, country: str, year: int) -> float:
        """
        Get central government debt for a country and year.

        Args:
            country (str): Country code (ISO 3-letter)
            year (int): Year to get debt data for

        Returns:
            float: Central government debt value

        Note:
            - Returns 0.0 for Argentina and Taiwan
            - Falls back to previous year's value if data not available
            - Returns 0.0 for year 1959
        """
        df = self.data["gov_debt"].set_index("Country Code", drop=True)
        if country == "ARG":
            return 0.0
        if country == "TWN":
            return 0.0
        if year == 1959:
            return 0.0
        val = df.at[country, str(year) + " [YR" + str(year) + "]"]
        if val == "..":
            return self.get_central_gov_debt(country, year - 1)
        else:
            return float(val)

    def get_population(self, country: Country, year: int) -> float:
        """
        Get total population for a country and year.

        Args:
            country (Country): Country to get population for
            year (int): Year to get population data for

        Returns:
            float: Total population count

        Note:
            Uses World Bank's total population indicator (SP.POP.TOTL)
        """
        df = self.data["population"].set_index("Country Code")
        return df.loc[country, str(year)]

    def get_participation_rate(self, country: Country) -> pd.DataFrame:
        """
        Retrieves the participation rate for a specific country and year.

        Parameters:
            country (Country): The country code for the desired country.

        Returns:
            pd.DataFrame: A DataFrame containing the participation rate for the specified country.
        """
        df = self.data["participation"]
        df = df.loc[df["Country Code"] == country]
        data = []
        index = []
        for year in range(1960, 2024):
            for month in [1, 4, 7, 10]:
                index.append(pd.Timestamp(year, month, 1))
                if country == "TWN":
                    data.append(0.592)
                else:
                    if str(year) in df.columns:
                        val = df[str(year)].values[0] / 100.0
                    else:
                        val = np.nan
                    data.append(val)
        return pd.DataFrame(data={"Participation Rate": data}, index=index).bfill()

    def get_tau_vat(self, country: Country, year: int) -> float:
        """
        Get VAT (Value Added Tax) rate for a country and year.

        Args:
            country (Country): Country to get VAT rate for
            year (int): Year to get tax rate for

        Returns:
            float: VAT rate as decimal

        Note:
            - Uses forced_vat values for countries with missing or unreliable data
            - Tax rate is expressed as a decimal (e.g., 0.20 for 20% VAT)
            - Returns 0.0 if data not available and country not in forced_vat
        """
        df = self.data["tau_vat"]
        if country in forced_vat:
            return forced_vat[country]
        df = df.loc[df["Country Code"] == country][str(year)]
        return df.values[0] / 100.0

    def get_lcu_exports(self, country: Country, year: int) -> float:
        """
        Retrieves the export tax rate for a specific country and year.

        Parameters:
            country (Country): The country code for the desired country.
            year (int): The year for the data.

        Returns:
            float: The export tax rate for the specified country and year.
        """
        df = self.data["tau_exp"].fillna(0)
        df = df.loc[df["Country Code"] == country][str(year)]
        return df.values[0] / 100.0

    def get_gini_coef(self, country: Country, year: int) -> float:
        """
        Retrieves the Gini coefficient for a specific country and year.

        Parameters:
            country (Country): The country code for the desired country.
            year (int): The year for the data.

        Returns:
            float: The Gini coefficient for the specified country and year.
        """
        df = self.data["gini_coefs"]
        return df.loc[df["Country Code"] == country][str(year)].values[0] / 100

    def get_historic_gdp(self, country: Country, year: int) -> float:
        """
        Get historical GDP value for a country and year.

        Args:
            country (Country): Country to get GDP for
            year (int): Year to get GDP data for

        Returns:
            float: GDP value in local currency units (LCU)

        Note:
            - Uses World Bank's GDP indicator (NY.GDP.MKTP.CN)
            - Values are in current local currency units
            - Returns raw value without scaling
        """
        df = self.data["historic_gdp"]
        df = df.loc[df["Country Code"] == country].iloc[:, 4:]
        return df.loc[:, str(year)].values[0]

    def get_current_scaled_gdp(self, country: Country, year: int, rescale_factor: float = 4.0) -> float:
        """
        Get scaled current GDP value for a country and year.

        Args:
            country (Country): Country to get GDP for
            year (int): Year to get GDP data for
            rescale_factor (float, optional): Factor to scale GDP by.
                                            Defaults to 4.0 for quarterly data.

        Returns:
            float: Scaled GDP value in local currency units (LCU)

        Note:
            - Uses historic GDP values divided by rescale_factor
            - Typically used to convert annual to quarterly values
            - Values are in current local currency units
        """
        return self.get_historic_gdp(country, year) / rescale_factor

    def get_log_inflation(self, country: Country, start_year: int = 1970, end_year: int = 2024) -> pd.DataFrame:
        """
        Retrieves the log inflation data for a specific country within a given time range.

        Parameters:
            country (Country): The country code for the desired country.
            start_year (int): The starting year for the data (default: 1970).
            end_year (int): The ending year for the data (default: 2024).

        Returns:
            pd.DataFrame: A DataFrame containing the log growth of inflation for the specified country and time range.
        """
        # Get CPI and PPI data
        data_cpi = self.data["cpi"].loc[self.data["cpi"]["Country Code"] == country]
        data_ppi = self.data["ppi"].loc[self.data["ppi"]["Country Code"] == country]

        # Get the columns that are dates and convert them to datetime objects
        # then create series with the datetime objects as the index
        cpi_cols = data_cpi.columns
        cpi_datetime = pd.to_datetime(cpi_cols, errors="coerce", format="%Y%m")
        data_cpi = data_cpi[cpi_cols[cpi_datetime.notnull()]]
        data_cpi.columns = cpi_datetime[cpi_datetime.notnull()]
        data_cpi.index = ["CPI"]
        data_cpi = data_cpi.T

        ppi_cols = data_ppi.columns
        ppi_datetime = pd.to_datetime(ppi_cols, errors="coerce", format="%Y%m")
        data_ppi = data_ppi[ppi_cols[ppi_datetime.notnull()]]
        data_ppi.columns = ppi_datetime[ppi_datetime.notnull()]
        data_ppi.index = ["PPI"]
        data_ppi = data_ppi.T
        data_cpi.sort_index(inplace=True)
        data_ppi.sort_index(inplace=True)

        inflation_data = pd.merge_asof(data_cpi, data_ppi, left_index=True, right_index=True)

        # Calculate log inflation
        inflation_data = np.log(inflation_data).diff()

        # rename
        inflation_data.columns = ["Real CPI Inflation", "Real PPI Inflation"]
        return inflation_data

    def get_unemployment_rate(self, country: str) -> pd.DataFrame:
        """
        Get time series of unemployment rates.

        Args:
            country (str): Country to get unemployment rates for

        Returns:
            pd.DataFrame: DataFrame with dates as index and unemployment rates
                         as values (in decimal form)

        Note:
            - Returns quarterly data
            - Uses World Bank's total unemployment indicator (SL.UEM.TOTL.ZS)
            - Forward fills missing values
        """
        df = self.data["unemployment"]
        df = df.loc[df["Country Code"] == country]
        df = df.drop(columns=["Country Code", "Country Name", "Indicator Name", "Indicator Code", "Unnamed: 66"])
        df = df.T
        df.index = pd.to_datetime(df.index, format="%Y")
        df.columns = ["Unemployment Rate"]
        df = df.resample("QS").first().ffill().bfill() / 100.0
        # this is a pandas bug!
        df.index.freq = None
        return df

    def get_inflation(self, country: str) -> pd.DataFrame:
        if country == "ARG":
            inflation_arg = 1.0 + self.data["inflation_arg"].set_index("Date") / 100.0
            inflation_arg.index = pd.to_datetime(inflation_arg.index)
            inflation_arg = inflation_arg.groupby(pd.Grouper(freq="QE")).cumprod() - 1.0
            inflation_arg.index = pd.to_datetime([d + pd.Timedelta(days=1) for d in inflation_arg.index.values])
            inflation_arg = inflation_arg.iloc[2:].iloc[::3]
            inflation_arg = inflation_arg[["Amount", "Amount"]]
            inflation_arg.columns = ["CPI Inflation", "PPI Inflation"]
            return inflation_arg.astype(float)

        # Get CPI and PPI data
        data_cpi = self.data["cpi"].loc[self.data["cpi"]["Country Code"] == country]
        data_ppi = self.data["ppi"].loc[self.data["ppi"]["Country Code"] == country]
        dates, vals_cpi, vals_ppi = [], [], []
        for year in range(1970, 2024):
            for quarter in range(1, 5):
                month = 3 * quarter - 2
                s_month = str(month) if month > 9 else "0" + str(month)
                dates.append(str(year) + "-Q" + str(quarter))

                # CPI
                if str(year) + s_month in data_cpi.columns:
                    val_cpi = data_cpi.loc[:, str(year) + s_month].values
                    if len(val_cpi) == 0:
                        vals_cpi.append(np.nan)
                    else:
                        vals_cpi.append(val_cpi[0])
                else:
                    vals_cpi.append(np.nan)

                # PPI
                if str(year) + s_month in data_ppi.columns:
                    val_ppi = data_ppi.loc[:, str(year) + s_month].values
                    if len(val_ppi) == 0:
                        vals_ppi.append(np.nan)
                    else:
                        vals_ppi.append(val_ppi[0])
                else:
                    vals_ppi.append(np.nan)

        # Compute inflation
        data_df = pd.DataFrame(
            index=dates,
            data={
                "CPI Inflation": vals_cpi,
                "PPI Inflation": vals_ppi,
            },
        )
        data_df["CPI Inflation"] = np.log(data_df["CPI Inflation"] / data_df["CPI Inflation"].shift(1))
        data_df["PPI Inflation"] = np.log(data_df["PPI Inflation"] / data_df["PPI Inflation"].shift(1))
        data_df.index = [pd.Timestamp(int(ind[0:4]), 3 * int(ind[6]) - 2, 1) for ind in data_df.index]  # noqa

        return data_df.astype(float)

    def get_tau_exp(self, country: str, year: int, default_value: float = 0.0) -> float:
        df = self.data["tau_exp"]
        val = df.loc[df["Country Code"] == country, str(year)].values
        if len(val) == 0 or np.isnan(val[0]):
            return default_value
        else:
            return val[0]

    def prune(self, prune_date: date) -> None:
        """
        Prunes the data based on a given prune date.

        Parameters:
            prune_date (date): The date to prune the data. Can be an integer, string, or pandas Timestamp.

        Returns:
            None
        """

        for key, value in self.data.items():
            years_as_columns = True
            for col in value.columns:
                if col.lower() in ["year", "time"]:
                    years_as_columns = False
                    # Check if column can be transformed in a date
                    dates = pd.to_datetime(value[col], errors="coerce", format="mixed")
                    if dates.isnull().sum() == 0:
                        logging.log(level=logging.DEBUG, msg=f"WBank: NAT dates {dates.isna().sum()}.")
                        mask = dates >= pd.to_datetime(prune_date)
                        if mask.sum() == 0:
                            warnings.warn(
                                f"No rows were kept for date {prune_date} in World Bank dataset {key}.",
                                DataFilterWarning,
                            )
                        logging.log(
                            level=logging.DEBUG, msg=f"WBank: Keeping {mask.sum()} rows for prune date {prune_date}."
                        )
                        self.data[key] = value.loc[mask, :]
                        break

            if years_as_columns is True:
                mask = prune_index(value.columns, prune_date)
                self.data[key] = value.loc[:, mask]

    def get_npl_ratios(self, country: Country | str) -> pd.DataFrame:
        npl_ratio = self.data["npl_ratios"].set_index("Country Code", drop=True).loc[[country]]
        new_cols = [str(y) + " [YR" + str(y) + "]" for y in range(1960, 2022)]
        npl_ratio = npl_ratio.loc[:, new_cols]
        npl_ratio.columns = [str(y) for y in range(1960, 2022)]
        npl_ratio.columns = pd.to_datetime(npl_ratio.columns, format="%Y")

        npl_ratio = npl_ratio.T

        npl_ratio[npl_ratio == ".."] = np.nan

        return npl_ratio.astype(float) / 100.0
