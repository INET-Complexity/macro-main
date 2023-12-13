import warnings
from pathlib import Path
import logging

import numpy as np
import pandas as pd

from inet_data.readers.util.prune_util import DataFilterWarning, prune_index


class WorldBankReader:
    """
    A class for reading and retrieving economic data from the World Bank dataset.

    Methods:
        __init__(self, path: Path): Initializes a WorldBankReader instance.
        get_unemployment_rate(self, country: str, year: int) -> float: Retrieves the unemployment rate for a specific country and year.
        get_participation_rate(self, country: str, year: int) -> float: Retrieves the participation rate for a specific country and year.
        get_tau_vat(self, country: str, year: int) -> float: Retrieves the VAT tax rate for a specific country and year.
        get_tau_exp(self, country: str, year: int) -> float: Retrieves the export tax rate for a specific country and year.
        get_gini_coef(self, country: str, year: int) -> float: Retrieves the Gini coefficient for a specific country and year.
        get_historic_gdp(self, country: str, year: int) -> float: Retrieves the historic GDP for a specific country and year.
        get_current_monthly_gdp(self, country: str, year: int) -> float: Retrieves the current monthly GDP for a specific country and year.
        get_log_inflation(self, country: str, start_year: int = 1970, end_year: int = 2024) -> pd.DataFrame: Retrieves the log inflation data for a specific country within a given time range.
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
        }

    def get_unemployment_rate(self, country: str, year: int) -> float:
        """
        Retrieves the unemployment rate for a specific country and year.

        Parameters:
            country (str): The country code for the desired country.
            year (int): The year for the data.

        Returns:
            float: The unemployment rate for the specified country and year.
        """
        df = self.data["unemployment"]
        df = df.loc[df["Country Code"] == country, str(year)]
        return df.values[0] / 100.0

    def get_participation_rate(self, country: str, year: int) -> float:
        """
        Retrieves the participation rate for a specific country and year.

        Parameters:
            country (str): The country code for the desired country.
            year (int): The year for the data.

        Returns:
            float: The participation rate for the specified country and year.
        """
        df = self.data["participation"]
        df = df.loc[df["Country Code"] == country, str(year)]
        return df.values[0] / 100.0

    def get_tau_vat(self, country: str, year: int) -> float:
        """
        Retrieves the VAT tax rate for a specific country and year.

        Parameters:
            country (str): The country code for the desired country.
            year (int): The year for the data.

        Returns:
            float: The VAT tax rate for the specified country and year.
        """
        df = self.data["tau_vat"]
        df = df.loc[df["Country Code"] == country][str(year)]
        return df.values[0] / 100.0

    def get_tau_exp(self, country: str, year: int) -> float:
        """
        Retrieves the export tax rate for a specific country and year.

        Parameters:
            country (str): The country code for the desired country.
            year (int): The year for the data.

        Returns:
            float: The export tax rate for the specified country and year.
        """
        df = self.data["tau_exp"]
        df = df.loc[df["Country Code"] == country][str(year)]
        return df.values[0] / 100.0

    def get_gini_coef(self, country: str, year: int) -> float:
        """
        Retrieves the Gini coefficient for a specific country and year.

        Parameters:
            country (str): The country code for the desired country.
            year (int): The year for the data.

        Returns:
            float: The Gini coefficient for the specified country and year.
        """
        df = self.data["gini_coefs"]
        return df.loc[df["Country Code"] == country][str(year)].values[0] / 100

    def get_historic_gdp(self, country: str, year: int) -> float:
        """
        Retrieves the historic GDP for a specific country and year.

        Parameters:
            country (str): The country code for the desired country.
            year (int): The year for the data.

        Returns:
            float: The historic GDP for the specified country and year.
        """
        df = self.data["historic_gdp"]
        df = df.loc[df["Country Code"] == country].iloc[:, 4:]
        return df.loc[:, str(year)].values[0]

    def get_current_monthly_gdp(self, country: str, year: int) -> float:
        """
        Retrieves the current monthly GDP for a specific country and year.

        Parameters:
            country (str): The country code for the desired country.
            year (int): The year for the data.

        Returns:
            float: The current monthly GDP for the specified country and year.
        """
        return self.get_historic_gdp(country, year) / 12.0

    def get_log_inflation(self, country: str, start_year: int = 1970, end_year: int = 2024) -> pd.DataFrame:
        """
        Retrieves the log inflation data for a specific country within a given time range.

        Parameters:
            country (str): The country code for the desired country.
            start_year (int): The starting year for the data (default: 1970).
            end_year (int): The ending year for the data (default: 2024).

        Returns:
            pd.DataFrame: A DataFrame containing the log growth of inflation for the specified country and time range.
        """
        # Get CPI and PPI data
        data_cpi = self.data["cpi"].loc[self.data["cpi"]["Country Code"] == country]
        data_ppi = self.data["ppi"].loc[self.data["cpi"]["Country Code"] == country]
        dates, vals_cpi, vals_ppi = [], [], []
        for year in range(start_year, end_year):
            for month in range(1, 13):
                s_month = str(month) if month > 9 else "0" + str(month)
                dates.append(str(year) + "-" + str(month))

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
                "Real CPI Inflation": vals_cpi,
                "Real PPI Inflation": vals_ppi,
            },
        )
        data_df["Real CPI Inflation"] = np.log(data_df["Real CPI Inflation"] / data_df["Real CPI Inflation"].shift(1))
        data_df["Real PPI Inflation"] = np.log(data_df["Real PPI Inflation"] / data_df["Real PPI Inflation"].shift(1))

        return data_df

    def prune(self, prune_date: int | str | pd.Timestamp, prune_date_format: str = "%Y-%m-%d") -> None:
        """
        Prunes the data based on a given prune date.

        Parameters:
            prune_date (int | str | pd.Timestamp): The date to prune the data. Can be an integer, string, or pandas Timestamp.
            prune_date_format (str): The format of the prune_date if it is a string. Default is "%Y-%m-%d".

        Returns:
            None
        """

        if isinstance(prune_date, str):
            prune_date = pd.to_datetime(prune_date, format=prune_date_format)
        elif isinstance(prune_date, int):
            prune_date = pd.to_datetime(str(prune_date), format="%Y")

        for key, value in self.data.items():
            years_as_columns = True
            for col in value.columns:
                if col.lower() in ["year", "time"]:
                    years_as_columns = False
                    # Check if column can be transformed in a date
                    dates = pd.to_datetime(value[col], errors="coerce", format="mixed")
                    if dates.isnull().sum() == 0:
                        logging.log(level=logging.DEBUG, msg=f"WBank: NAT dates {dates.isna().sum()}.")
                        mask = dates >= prune_date
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
                mask = prune_index(value.columns, prune_date, "World Bank")
                self.data[key] = value.loc[:, mask]
