"""
This module provides functionality for reading and processing Eurostat economic data.
It handles various economic indicators including financial balance sheets, GDP,
debt ratios, interest rates, and sectoral growth rates across European countries.

Key Features:
- Read and process multiple Eurostat datasets
- Handle country code conversions (Alpha-2 to Alpha-3)
- Support for financial indicators and ratios
- GDP and sectoral growth calculations
- Proxy mechanisms for missing data
- Data pruning capabilities

Example:
    ```python
    from pathlib import Path
    from macro_data.readers.economic_data.eurostat_reader import EuroStatReader
    from macro_data.configuration.countries import Country

    # Initialize reader with data directory
    reader = EuroStatReader(
        path=Path("path/to/eurostat/data"),
        country_code_path=Path("path/to/country_codes.csv"),
        proxy_country="GBR"
    )

    # Get various economic indicators
    gdp = reader.get_quarterly_gdp("FRA", 2020, 1)
    debt_ratio = reader.nonfin_firm_debt_ratios("DEU", 2020)
    growth = reader.get_perc_sectoral_growth("ITA")
    ```

Note:
    Most monetary values are in millions of national currency units.
    Ratios and rates are typically returned as decimals (e.g., 0.05 for 5%).
"""

import warnings
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.readers.util.prune_util import DataFilterWarning, prune_index


def get_perc_growth_series(country: str, growth_df: pd.DataFrame, series_name: Optional[str] = None) -> pd.Series:
    """
    Extract and format percentage growth series for a specific country.

    Args:
        country (str): Country code to extract data for
        growth_df (pd.DataFrame): DataFrame containing growth data
        series_name (Optional[str]): Name to assign to resulting series

    Returns:
        pd.Series: Time series of growth rates with datetime index

    Note:
        Expects DataFrame with 'TIME' column for dates and country columns
    """
    df = growth_df.copy()
    df.rename(columns={"TIME": "Country"}, inplace=True)
    df.set_index("Country", inplace=True)
    df = df.T
    df.index = pd.to_datetime(df.index, format="%Y-%m")
    series = df.loc[:, country]
    if series_name is not None:
        series.name = series_name
    return series


class EuroStatReader:
    """
    A class for reading and processing Eurostat economic data.

    This class handles various economic indicators including:
    1. Financial balance sheets and ratios
    2. GDP and sectoral growth
    3. Debt and deposit statistics
    4. Interest rates and bond yields
    5. Household and firm statistics

    Args:
        path (Path | str): Path to directory containing Eurostat data files
        country_code_path (Path | str): Path to CSV file containing country code mappings
        total_output (Optional[dict[str, float]]): Dictionary of total output by country for scaling
        proxy_country (str): Country to use as proxy when data is missing (default: "GBR")

    Attributes:
        c_map (pd.DataFrame): Country code mapping DataFrame
        remap_2_to_3 (dict): Mapping from Alpha-2 to Alpha-3 country codes
        remap_3_to_2 (dict): Mapping from Alpha-3 to Alpha-2 country codes
        proxy_country (str): Country used for proxying missing data
        total_output (Optional[dict[str, float]]): Total output by country
        data (dict[str, pd.DataFrame]): Dictionary of loaded Eurostat datasets

    Note:
        - Handles special cases for Greece (GR -> EL) and UK (GB -> UK)
        - Most monetary values are in millions of national currency
        - Ratios and rates are typically returned as decimals
    """

    def __init__(
        self,
        path: Path | str,
        country_code_path: Path | str,
        total_output: Optional[dict[str, float]] = None,
        proxy_country: str = "GBR",
    ):
        # Handle country codes
        self.c_map = pd.read_csv(country_code_path)
        # switch 2-digit code for Greece
        self.c_map.loc[self.c_map["Alpha-2 code"] == "GR", "Alpha-2 code"] = "EL"
        # switch 2-digit code United Kingdom
        self.c_map.loc[self.c_map["Alpha-2 code"] == "GB", "Alpha-2 code"] = "UK"

        self.remap_2_to_3 = self.c_map.set_index(["Alpha-2 code"])["Alpha-3 code"].to_dict()
        self.remap_3_to_2 = self.c_map.set_index(["Alpha-3 code"])["Alpha-2 code"].to_dict()

        # For proxying
        self.proxy_country = proxy_country
        self.total_output = total_output

        # Load data files
        self.files_with_codes = self.get_files_with_codes()
        self.data = {}
        for key in self.files_with_codes.keys():
            self.data[key] = pd.read_csv(path / (self.files_with_codes[key] + ".csv"))
            if "geo" in self.data[key].columns:
                self.data[key]["geo"] = self.country_code_switch(self.data[key]["geo"])

    @staticmethod
    def get_files_with_codes() -> dict[str, str]:
        """
        Get mapping of data categories to file names.

        Returns:
            dict[str, str]: Dictionary mapping data categories to their file names

        Note:
            File categories include:
            - Financial indicators (debt ratios, equity ratios)
            - Economic indicators (GDP, CPI)
            - Input-output tables
            - Sectoral growth rates
            - Balance sheets and transactions
        """
        return {
            "central_bank_debt_ratio": "eurostat_cbdebt_ratios",
            "central_bank_equity_ratio": "eurostat_cbequity_ratios",
            "central_gov_debt_ratio": "eurostat_central_govdebt_ratios",
            "cpi": "eurostat_cpi",
            "iot_tables": "naio_10_cp1700",
            "firm_debt_ratio": "eurostat_firmdebt_ratios",
            "firm_deposits_ratio": "eurostat_firmdeposit_ratios",
            "gdp": "namq_10_gdp",
            "general_gov_debt_ratio": "eurostat_general_govdebt_ratios",
            "nonfinancial_transactions": "nasa_10_nf_tr",
            "longterm_central_gov_bond_rates": "eurostat_longterm_govbond_rates",
            "shortterm_interest_rates": "irt_st_a",
            "financial_balance_sheets": "nasa_10_f_bs",
            "number_of_households": "lfst_hhnhtych",
            "capital_formation": "nama_10_nfa_fl",
            "perc_growth_sector_B": "perc_growth_sector_B",
            "perc_growth_sector_C": "perc_growth_sector_C",
            "perc_growth_sector_D": "perc_growth_sector_D",
            "perc_growth_sector_F": "perc_growth_sector_F",
            "perc_growth_services": "perc_growth_services",
            "real_estate_services": "sector_l_iot",
            "investment_percentage_of_gdp": "tec00132_linear",
        }

    def country_code_switch(self, codes: pd.Series) -> pd.Series:
        """
        Convert Alpha-2 country codes to Alpha-3 format.

        Args:
            codes (pd.Series): Series of Alpha-2 country codes

        Returns:
            pd.Series: Series of corresponding Alpha-3 country codes

        Note:
            Uses the remap_2_to_3 dictionary for conversion
        """
        return codes.map(lambda x: self.remap_2_to_3.get(x, x))

    @staticmethod
    def find_value(df: pd.DataFrame, country: Country, year: str, return_last_value: bool = True) -> Optional[float]:
        """
        Find value in DataFrame for specific country and year.

        Args:
            df (pd.DataFrame): DataFrame containing the data
            country (Country): Country to find data for
            year (str): Year to find data for
            return_last_value (bool): Whether to return last available value if year not found (default: True)

        Returns:
            Optional[float]: Found value, or None if not found and return_last_value is False

        Note:
            - If country not found, returns mean value for the year
            - If year not found and return_last_value True, returns last available value
        """
        country_data = df.loc[df["geo"] == country]

        if country_data.empty:
            return df.loc[df["TIME_PERIOD"] == int(year), "OBS_VALUE"].mean()
        if int(year) in country_data["TIME_PERIOD"].values:
            return country_data.loc[country_data["TIME_PERIOD"] == int(year), "OBS_VALUE"].values[0]
        elif return_last_value:
            values = country_data["OBS_VALUE"].values
            # return last value
            return values[-1]
        else:
            return None

    def nonfin_firm_debt_ratios(self, country: Country, year: int) -> float:
        """
        Get non-financial firm debt ratios for a specific country and year.

        Args:
            country (Country): Country to get debt ratios for
            year (int): Year to get debt ratios for

        Returns:
            float: Non-financial firm debt ratio as a decimal

        Note:
            Returns ratio of total non-financial firm debt to GDP
        """
        df = self.data["firm_debt_ratio"]
        return self.find_value(df, country, str(year)) / 100.0

    # historic domestic
    def get_total_nonfin_firm_debt(self, country: Country | str, year: int) -> float:
        """
        Get total non-financial firm debt for a specific country and year.

        Args:
            country (Country | str): Country to get debt for
            year (int): Year to get debt for

        Returns:
            float: Total non-financial firm debt in millions of national currency
        """
        if isinstance(country, str):
            country = Country(country)
        df = self.data["financial_balance_sheets"]
        country_name_short = self.c_map.loc[self.c_map["Alpha-3 code"] == country, "Alpha-2 code"].values[0]
        df = df.loc[
            df[r"unit,co_nco,sector,finpos,na_item,geo\time"] == "MIO_NAC,NCO,S11,LIAB,F4," + country_name_short
        ]
        if str(year) in df.columns:
            res = df[str(year)].values[0]
            if len(res) <= 2:
                return np.nan
            if " " in res:
                return float(res[:-2]) * 1e6
            else:
                return float(res) * 1e6
        else:
            return np.nan

    def get_total_fin_firm_debt(self, country: Country | str, year: int) -> float:
        """
        Get total financial firm debt for a specific country and year.

        Args:
            country (Country | str): Country to get debt for
            year (int): Year to get debt for

        Returns:
            float: Total financial firm debt in millions of national currency
        """
        if isinstance(country, str):
            country = Country(country)
        df = self.data["financial_balance_sheets"]
        country_name_short = self.c_map.loc[self.c_map["Alpha-3 code"] == country, "Alpha-2 code"].values[0]
        df = df.loc[
            df[r"unit,co_nco,sector,finpos,na_item,geo\time"] == "MIO_NAC,NCO,S12,LIAB,F4," + country_name_short
        ]
        return float(df[str(year)].values[0]) * 1e6

    def get_total_household_deposits(self, country: str, year: int, proxy_country: str = "FRA") -> float:
        """
        Get total household deposits for a specific country and year.

        Args:
            country (str): Country to get deposits for
            year (int): Year to get deposits for
            proxy_country (str): Country to use as proxy if data not available (default: "FRA")

        Returns:
            float: Total household deposits in millions of national currency

        Note:
            If data not available for specified country, uses proxy country scaled by total output ratio
        """
        df = self.data["financial_balance_sheets"]
        country_name_short = self.c_map.loc[self.c_map["Alpha-3 code"] == country, "Alpha-2 code"].values[0]
        df = df.loc[
            df[r"unit,co_nco,sector,finpos,na_item,geo\time"] == "MIO_NAC,CO,S1314,ASS,F2," + country_name_short
        ]
        val = df[str(year)].values
        if len(val) == 0 or val[0] == ": ":
            if proxy_country in self.total_output:
                return (
                    self.total_output[country]
                    / self.total_output[proxy_country]
                    * self.get_total_household_deposits(proxy_country, year)
                )
            else:
                return self.get_total_household_deposits(proxy_country, year)
        return float(val[0]) * 1e6

    def get_total_household_fixed_assets(self, country: str, year: int, proxy_country: str = "GBR") -> float:
        """
        Get total household fixed assets for a specific country and year.

        Args:
            country (str): Country to get fixed assets for
            year (int): Year to get fixed assets for
            proxy_country (str): Country to use as proxy if data not available (default: "GBR")

        Returns:
            float: Total household fixed assets in millions of national currency

        Note:
            If data not available for specified country, uses proxy country scaled by total output ratio
        """
        df = self.data["non_financial_balance_sheets"]
        country_name_short = self.c_map.loc[self.c_map["Alpha-3 code"] == country, "Alpha-2 code"].values[0]
        df = df.loc[df[r"freq,unit,nace_r2,asset10,geo\TIME_PERIOD"] == "A,CRC_MNAC,TOTAL,N111N," + country_name_short]
        val = df[str(year)].values
        if len(val) == 0:
            return (
                self.total_output[country]
                / self.total_output[proxy_country]
                * self.get_total_household_fixed_assets(proxy_country, year)
            )
        if " " in val[0]:
            return float(val[0][:-2]) * 1e6
        return float(val[0]) * 1e6

    def nonfin_firm_deposit_ratios(self, country: Country, year: int) -> float:
        """
        Get non-financial firm deposit ratios for a specific country and year.

        Args:
            country (Country): Country to get deposit ratios for
            year (int): Year to get deposit ratios for

        Returns:
            float: Non-financial firm deposit ratio as a decimal

        Note:
            Returns ratio of total non-financial firm deposits to GDP
        """
        df = self.data["firm_deposits_ratio"]
        return self.find_value(df, country, str(year)) / 100.0

    def get_quarterly_gdp(self, country: Country, year: int, quarter: int) -> float:
        """
        Get quarterly GDP for a specific country, year, and quarter.

        Args:
            country (Country): Country to get GDP for
            year (int): Year to get GDP for
            quarter (int): Quarter to get GDP for (1-4)

        Returns:
            float: Quarterly GDP in millions of national currency
        """
        df = self.data["gdp"]
        return (
            df.loc[
                (df["geo"] == country) & (df["TIME_PERIOD"] == f"{year}-Q{quarter}"),
                "OBS_VALUE",
            ].values[0]
            * 1e6
        )

    def get_monthly_gdp(self, country: Country, year: int, month: int) -> float:
        """
        Get monthly GDP for a specific country, year, and month.

        Args:
            country (Country): Country to get GDP for
            year (int): Year to get GDP for
            month (int): Month to get GDP for (1-12)

        Returns:
            float: Monthly GDP in millions of national currency

        Note:
            Interpolates quarterly GDP to monthly values using linear interpolation
        """
        start_quarter = (month - 1) // 3 + 1
        start = self.get_quarterly_gdp(country, year, start_quarter)

        if start_quarter == 4:
            end = self.get_quarterly_gdp(country, year + 1, 1)
        else:
            end = self.get_quarterly_gdp(country, year, start_quarter + 1)

        return start + (end - start) * ((month - 1) % 3) / 3

    # historic domestic
    def get_total_nonfin_firm_deposits(self, country: Country | str, year: int) -> float:
        """
        Get total non-financial firm deposits for a specific country and year.

        Args:
            country (Country | str): Country to get deposits for
            year (int): Year to get deposits for

        Returns:
            float: Total non-financial firm deposits in millions of national currency
        """
        if isinstance(country, str):
            country = Country(country)
        df = self.data["financial_balance_sheets"]
        country_name_short = self.c_map.loc[self.c_map["Alpha-3 code"] == country, "Alpha-2 code"].values[0]
        df = df.loc[df[r"unit,co_nco,sector,finpos,na_item,geo\time"] == "MIO_NAC,NCO,S11,ASS,F2," + country_name_short]
        # Check if DataFrame is empty or year doesn't exist
        if df.empty or str(year) not in df.columns:
            return np.nan
            
        res = df[str(year)].values[0]
        if len(res) <= 2:
            return np.nan
        if " " in res:
            return float(res[:-2]) * 1e6
        else:
            return float(res) * 1e6

    # historic domestic
    def get_total_bank_equity(self, country: str, year: int, proxy_country: str = "FRA") -> float:
        """
        Get total bank equity for a specific country and year.

        Args:
            country (str): Country to get bank equity for
            year (int): Year to get bank equity for
            proxy_country (str): Country to use as proxy if data not available (default: "FRA")

        Returns:
            float: Total bank equity in millions of national currency

        Note:
            If data not available for specified country, uses proxy country scaled by total output ratio
        """
        df = self.data["financial_balance_sheets"]
        country_name_short = self.c_map.loc[self.c_map["Alpha-3 code"] == country, "Alpha-2 code"].values[0]
        df = df.loc[
            df[r"unit,co_nco,sector,finpos,na_item,geo\time"] == "MIO_NAC,NCO,S122_S123,ASS,F5," + country_name_short
        ]
        if str(year) in df.columns:
            val = df[str(year)].values
            if len(val) > 0 and val[0] != ": d" and val[0] != ": de" and val[0] != ": ":
                return float(val[0]) * 1e6
        if proxy_country in self.total_output:
            return (
                self.total_output[country]
                / self.total_output[proxy_country]
                * self.get_total_bank_equity(proxy_country, year)
            )
        else:
            return self.get_total_bank_equity(proxy_country, year)

    def cb_debt_ratios(self, country: Country, year: int) -> float:
        """
        Get central bank debt ratios for a specific country and year.

        Args:
            country (Country): Country to get debt ratios for
            year (int): Year to get debt ratios for

        Returns:
            float: Central bank debt ratio as a decimal

        Note:
            Returns ratio of central bank debt to GDP
        """
        df = self.data["central_bank_debt_ratio"]
        return self.find_value(df, country, str(year)) / 100.0

    def cb_equity_ratios(self, country: Country, year: int) -> float:
        """
        Get central bank equity ratios for a specific country and year.

        Args:
            country (Country): Country to get equity ratios for
            year (int): Year to get equity ratios for

        Returns:
            float: Central bank equity ratio as a decimal

        Note:
            Returns ratio of central bank equity to GDP
        """
        df = self.data["central_bank_equity_ratio"]
        return self.find_value(df, country, str(year)) / 100.0

    def general_gov_debt_ratios(self, country: Country, year: int) -> float:
        """
        Get general government debt ratios for a specific country and year.

        Args:
            country (Country): Country to get debt ratios for
            year (int): Year to get debt ratios for

        Returns:
            float: General government debt ratio as a decimal

        Note:
            Returns ratio of general government debt to GDP
        """
        df = self.data["general_gov_debt_ratio"]
        return self.find_value(df, country, str(year)) / 100

    def central_gov_debt_ratios(self, country: Country, year: int) -> float:
        """
        Get central government debt ratios for a specific country and year.

        Args:
            country (Country): Country to get debt ratios for
            year (int): Year to get debt ratios for

        Returns:
            float: Central government debt ratio as a decimal

        Note:
            Returns ratio of central government debt to GDP
        """
        df = self.data["central_gov_debt_ratio"]
        return self.find_value(df, country, str(year)) / 100

    def shortterm_interest_rates(self, country: Country | str, year: int, months: int) -> float:
        """
        Get short-term interest rates for a specific country and year.

        Args:
            country (Country | str): Country to get interest rates for
            year (int): Year to get interest rates for
            months (int): Number of months for interest rate term (0, 1, 3, 6, or 12)

        Returns:
            float: Short-term interest rate as a decimal

        Note:
            Returns money market interest rates for specified term length
        """
        assert months in [0, 1, 3, 6, 12]
        df = self.data["shortterm_interest_rates"]

        if months == 0:
            str_months = "IRT_DTD"
        else:
            str_months = f"IRT_M{str(months)}"

        return (
            df.loc[
                (df["geo"] == country) & (df["TIME_PERIOD"] == year) & (df["int_rt"] == str_months),
                "OBS_VALUE",
            ].values[0]
            / 100
        )

    def longterm_central_gov_bond_rates(self, country: Country, year: int) -> float:
        """
        Get long-term central government bond rates for a specific country and year.

        Args:
            country (Country): Country to get bond rates for
            year (int): Year to get bond rates for

        Returns:
            float: Long-term government bond rate as a decimal

        Note:
            Returns yield on long-term (typically 10-year) government bonds
        """
        df = self.data["longterm_central_gov_bond_rates"]
        return self.find_value(df, country, str(year)) / 100

    # in domestic
    # numerator only available in current prices
    # denominator only available in historic prices
    def dividend_payout_ratio(
        self, country: str | Country, year: int, proxy_country: Country = Country("FRA")
    ) -> float:
        """
        Calculate the dividend payout ratio for a given country and year.

        Args:
            country (str | Country): Country to get payout ratio for
            year (int): Year to get payout ratio for (defaults to 2011 if not 2010/2011)
            proxy_country (Country): Country to use as proxy if data not available (default: "FRA")

        Returns:
            float: Dividend payout ratio as a decimal

        Note:
            - Returns ratio of (household property income + household surplus) to firm surplus
            - Only 2010 and 2011 data available, defaults to 2011 for other years
        """
        if year not in [2010, 2011]:
            # print("Warning: Using the 2011 data for the dividend payout ratio.")
            year = 2011
        df = self.data["nonfinancial_transactions"]

        hh_prop_df = df[(df["na_item"] == "D4") & (df["direct"] == "RECV") & (df["sector"] == "S14")]

        hh_surplus_df = df[(df["na_item"] == "B2A3N") & (df["direct"] == "RECV") & (df["sector"] == "S14")]

        df = self.data["iot_tables"]
        firm_surplus_df = df[(df["induse"] == "TOTAL") & (df["prod_na"] == "B2A3N")]

        hh_prop = self.find_value(hh_prop_df, country, str(year), return_last_value=False)
        hh_surplus = self.find_value(hh_surplus_df, country, str(year), return_last_value=False)
        firm_surplus = self.find_value(firm_surplus_df, country, str(year), return_last_value=False)

        if hh_prop is None or hh_surplus is None or firm_surplus is None:
            return self.dividend_payout_ratio(proxy_country, year)

        return (hh_prop + hh_surplus) / firm_surplus

    def firm_risk_premium(self, country: Country, year: int) -> float:
        """
        Calculate the firm risk premium for a given country and year.

        Args:
            country (Country): Country to get risk premium for
            year (int): Year to get risk premium for

        Returns:
            float: Monthly firm risk premium as a decimal

        Note:
            Returns spread between firm borrowing rate and Euribor rate,
            converted to monthly rate
        """
        euribor_rate = self.shortterm_interest_rates("EA", year, 3)

        df = self.data["nonfinancial_transactions"]

        nonfin_firm_interest_payments_df = df[
            (df["na_item"] == "D41") & (df["direct"] == "PAID") & (df["sector"] == "S11")
        ]

        fin_firm_interest_payments_df = df[
            (df["na_item"] == "D41") & (df["direct"] == "PAID") & (df["sector"] == "S12")
        ]

        nonfin_firm_payments = self.find_value(nonfin_firm_interest_payments_df, country, str(year)) * 1e6
        fin_firm_payments = self.find_value(fin_firm_interest_payments_df, country, str(year)) * 1e6
        nonfin_firm_debt = self.get_total_nonfin_firm_debt(country, year)
        fin_firm_debt = self.get_total_fin_firm_debt(country, year)

        annual_premium = (nonfin_firm_payments + fin_firm_payments) / (nonfin_firm_debt + fin_firm_debt) - euribor_rate

        return (1 + annual_premium) ** (1.0 / 12) - 1.0

    def number_of_households(self, country: Country, year: int) -> float:
        """
        Get number of households for a specific country and year.

        Args:
            country (Country): Country to get household count for
            year (int): Year to get household count for

        Returns:
            float: Number of households (in thousands)
        """
        df = self.data["number_of_households"].set_index("geo")
        return int(df.loc[country, str(year)] * 1000)

    def taxrate_on_capital_formation(self, country: Country, year: int) -> float:
        """
        Get tax rate on capital formation for a specific country and year.

        Args:
            country (Country): Country to get tax rate for
            year (int): Year to get tax rate for

        Returns:
            float: Tax rate on capital formation as a decimal

        Note:
            Returns ratio of taxes on capital formation to total capital formation
        """
        capform_df = self.data["capital_formation"]

        df = self.data["iot_tables"]
        taxes_df = df[(df["induse"] == "P5") & (df["prod_na"] == "D21X31")]

        capform = self.find_value(capform_df, country, str(year))
        taxes = self.find_value(taxes_df, country, str(year))

        return taxes / capform

    def get_perc_sectoral_growth(self, country: Country) -> pd.DataFrame:
        """
        Retrieves the percentage sectoral growth data for a specific country.

        Args:
            country (Country): Country to get sectoral growth for

        Returns:
            pd.DataFrame: DataFrame containing growth rates by sector over time,
                         with sectors as columns and time as index

        Note:
            - Growth rates are in decimal form (divided by 100)
            - Includes sectors B, C, D, F and services
            - Service sector growth rate is applied to sectors A, E, G-S
        """
        # Get growth rates
        data_b = get_perc_growth_series(country=country, growth_df=self.data["perc_growth_sector_B"], series_name="B")
        data_c = get_perc_growth_series(country=country, growth_df=self.data["perc_growth_sector_C"], series_name="C")
        data_d = get_perc_growth_series(country=country, growth_df=self.data["perc_growth_sector_D"], series_name="D")
        data_f = get_perc_growth_series(country=country, growth_df=self.data["perc_growth_sector_F"], series_name="F")

        services = get_perc_growth_series(
            country=country, growth_df=self.data["perc_growth_services"], series_name="Services"
        )

        growth_df = pd.concat([data_b, data_c, data_d, data_f], axis=1)

        for serv_ind in [
            "A",
            "E",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R_S",
        ]:
            growth_df[serv_ind] = services

        growth_df = growth_df.astype(float)
        growth_df /= 100.0

        growth_df.columns.name = "Industry"
        growth_df.index.name = "Time"
        growth_df.sort_index(axis=1, inplace=True)

        return growth_df

    def get_total_industry_debt_and_deposits(
        self, country: Country, proxy_country: Optional[Country] = None
    ) -> pd.DataFrame:
        """
        Get total industry debt and deposits time series for a specific country.

        Args:
            country (Country): Country to get data for
            proxy_country (Optional[Country]): Country to use as proxy if data not available

        Returns:
            pd.DataFrame: DataFrame with monthly time series of total debt and deposits

        Raises:
            ValueError: If no data available and no proxy country provided

        Note:
            Returns monthly data from 1970 to 2024, using annual values repeated monthly
        """
        try:
            dates, total_deposits, total_debt = [], [], []
            for year in range(1970, 2024):
                dep = self.get_total_nonfin_firm_deposits(country, year)
                debt = self.get_total_nonfin_firm_debt(country, year)
                for month in range(1, 13):
                    dates.append(str(year) + "-" + str(month))
                    total_deposits.append(dep)
                    total_debt.append(debt)

            dates = pd.to_datetime(dates, format="%Y-%m")
            return pd.DataFrame(
                index=dates,
                data={
                    "Total Deposits": total_deposits,
                    "Total Debt": total_debt,
                },
            )
        except IndexError:
            if proxy_country is not None:
                return self.get_total_industry_debt_and_deposits(proxy_country)
            else:
                raise ValueError("No data available for the given country. Please provide a proxy country.")

    def get_imputed_rent_fraction_of_country(self, country: Country, year: int) -> float:
        """
        Get imputed rent fraction for a specific country and year.

        Args:
            country (Country): Country to get rent fraction for
            year (int): Year to get rent fraction for

        Returns:
            float: Imputed rent fraction as a decimal

        Note:
            Returns ratio of imputed rent (CPA_L68A) to total real estate services (CPA_L68A + CPA_L68B)
        """
        df = self.data["real_estate_services"].set_index("freq,unit,stk_flow,induse,prod_na,geo\TIME_PERIOD")
        country_name_short = self.c_map.loc[self.c_map["Alpha-3 code"] == country, "Alpha-2 code"].values[0]
        return float(df.at["A,MIO_NAC,TOTAL,P3_S14,CPA_L68A," + country_name_short, str(year)]) / (
            float(df.at["A,MIO_NAC,TOTAL,P3_S14,CPA_L68A," + country_name_short, str(year)])
            + float(df.at["A,MIO_NAC,TOTAL,P3_S14,CPA_L68B," + country_name_short, str(year)])
        )

    def get_investment_fractions_of_country(
        self,
        country_name: str,
        year: int,
    ) -> dict[str, float]:
        """
        Get investment fractions by sector for a specific country and year.

        Args:
            country_name (str): Country to get investment fractions for
            year (int): Year to get investment fractions for

        Returns:
            dict[str, float]: Dictionary with investment fractions by sector
                             (keys: "Firm", "Household", "Government")

        Note:
            - Returns normalized fractions (sum to 1)
            - Falls back to proxy country if data not available
            - Uses previous year's data if year > 2011
        """
        df = self.data["investment_percentage_of_gdp"].copy()
        df_country = df.loc[df["geo"] == country_name]
        if len(df_country) == 0:
            return self.get_investment_fractions_of_country(self.proxy_country, year)
        df_year = df_country.loc[df["TIME_PERIOD"] == year]
        if len(df_year) == 0:
            if year > 2011:
                return self.get_investment_fractions_of_country(country_name, year - 1)
            else:
                return self.get_investment_fractions_of_country(country_name, 2011)
        # total_perc = df_year.loc[df["indic"] == "INV_TOT"]["OBS_VALUE"].values[0] / 100.
        firm_perc = df_year.loc[df["indic"] == "INV_BSN"]["OBS_VALUE"].values[0] / 100.0
        hh_perc = df_year.loc[df["indic"] == "INV_HH"]["OBS_VALUE"].values[0] / 100.0
        gov_perc = df_year.loc[df["indic"] == "INV_GOV"]["OBS_VALUE"].values[0] / 100.0
        total_perc = firm_perc + hh_perc + gov_perc
        return {"Firm": firm_perc / total_perc, "Household": hh_perc / total_perc, "Government": gov_perc / total_perc}

    def get_imputed_rent_fraction(
        self,
        country_names: list[Country],
        year: int,
    ) -> dict[str, float]:
        """
        Get imputed rent fractions for multiple countries and a specific year.

        Args:
            country_names (list[Country]): List of countries to get rent fractions for
            year (int): Year to get rent fractions for

        Returns:
            dict[str, float]: Dictionary mapping country codes to their imputed rent fractions

        Note:
            Includes "ROW" (rest of world) entry with mean of all countries' fractions
        """
        fractions = {c: self.get_imputed_rent_fraction_of_country(c, year) for c in country_names}
        fractions[Country("ROW")] = np.mean(list(fractions.values()))
        return fractions

    def prune(self, prune_date: date):
        """
        Prune data to only include entries after specified date.

        Args:
            prune_date (date): Date to prune data from

        Returns:
            EuroStatReader: Self for method chaining

        Note:
            - Modifies data in place
            - Handles both time period columns and date-based columns
            - Warns if no data remains after pruning
        """
        # Eurostat
        for key, value in self.data.items():
            if "TIME_PERIOD" in value.columns:
                dates = value["TIME_PERIOD"].apply(convert_date)
                mask = dates.dt.date >= prune_date
                if mask.sum() == 0:
                    warnings.warn(
                        f"No rows were kept for date {prune_date} in Eurostat dataset {key}.",
                        DataFilterWarning,
                    )
                self.data[key] = value.loc[mask, :]
            else:
                mask = prune_index(value.columns, prune_date)
                self.data[key] = value.loc[:, mask]
        return self


def convert_date(date_value: str | int):
    """
    Convert various date formats to pandas datetime.

    Args:
        date_value (str | int): Date value to convert, can be year (int) or string format

    Returns:
        pd.Timestamp: Converted date

    Note:
        Handles both yearly dates (int) and quarterly dates (e.g., '2012-Q1')
    """
    if isinstance(date_value, int):
        return pd.to_datetime(date_value, format="%Y")
    # Check if the date is in quarterly format like '2012-Q1'
    if "Q" in date_value:
        year, quarter = date_value.split("-Q")
        month = (int(quarter) - 1) * 3 + 1  # Convert quarter to month
        return pd.to_datetime(f"{year}-{month:02d}-01")  # Format: YYYY-MM-DD
    else:
        # For other formats, directly use to_datetime
        return pd.to_datetime(date_value, errors="coerce", yearfirst=True)
