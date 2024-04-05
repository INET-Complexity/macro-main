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
    A class for reading and retrieving economic data from the World Bank dataset.

    Methods:
        __init__(self, path: Path): Initializes a WorldBankReader instance.
        get_unemployment_rate(self, country (Country), year: int) -> float: Retrieves the unemployment rate for a specific country and year.
        get_participation_rate(self, country (Country), year: int) -> float: Retrieves the participation rate for a specific country and year.
        get_tau_vat(self, country (Country), year: int) -> float: Retrieves the VAT tax rate for a specific country and year.
        get_tau_exp(self, country (Country), year: int) -> float: Retrieves the export tax rate for a specific country and year.
        get_gini_coef(self, country (Country), year: int) -> float: Retrieves the Gini coefficient for a specific country and year.
        get_historic_gdp(self, country (Country), year: int) -> float: Retrieves the historic GDP for a specific country and year.
        get_current_monthly_gdp(self, country (Country), year: int) -> float: Retrieves the current monthly GDP for a specific country and year.
        get_log_inflation(self, country (Country), start_year: int = 1970, end_year: int = 2024) -> pd.DataFrame: Retrieves the log inflation data for a specific country within a given time range.
        prune(self, prune_date: int | str | pd.Timestamp, date_format: str = "%Y-%m-%d") -> None: Prunes the data based on a given prune date.
    """

    def __init__(self, path: Path):
        self.files_with_codes = self.get_files_with_codes()

        self.data = {}
        for key in self.files_with_codes.keys():
            if key in [
                "long_term_interest_rates",
                "short_term_interest_rates",
                "government_debt_perc_gdp",
                "ppi",
                "cpi",
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

    def get_unemployment_rate(self, country: Country, year: int) -> float:
        """
        Retrieves the unemployment rate for a specific country and year.

        Parameters:
            country (Country): The country code for the desired country.
            year (int): The year for the data.

        Returns:
            float: The unemployment rate for the specified country and year.
        """
        df = self.data["unemployment"]
        df = df.loc[df["Country Code"] == country, str(year)]
        return df.values[0] / 100.0

    def get_population(self, country: Country, year: int) -> float:
        df = self.data["population"].set_index("Country Code")
        return df.loc[country, str(year)]

    def get_participation_rate(self, country: Country) -> pd.DataFrame:
        """
        Retrieves the participation rate for a specific country and year.

        Parameters:
            country (Country): The country code for the desired country.
            year (int): The year for the data.

        Returns:
            float: The participation rate for the specified country and year.
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
        Retrieves the VAT tax rate for a specific country and year.

        Parameters:
            country (Country): The country code for the desired country.
            year (int): The year for the data.

        Returns:
            float: The VAT tax rate for the specified country and year.
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
        Retrieves the historic GDP for a specific country and year.

        Parameters:
            country (Country): The country code for the desired country.
            year (int): The year for the data.

        Returns:
            float: The historic GDP for the specified country and year.
        """
        df = self.data["historic_gdp"]
        df = df.loc[df["Country Code"] == country].iloc[:, 4:]
        return df.loc[:, str(year)].values[0]

    def get_current_scaled_gdp(self, country: Country, year: int, rescale_factor: float = 4.0) -> float:
        """
        Retrieves the current monthly GDP for a specific country and year.

        Parameters:
            country (Country): The country code for the desired country.
            year (int): The year for the data.
            rescale_factor (float): The factor to rescale the GDP by (default: 4.0 for 4 quarters).

        Returns:
            float: The current monthly GDP for the specified country and year.
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
