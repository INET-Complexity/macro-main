import warnings
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from inet_data.readers.util.prune_util import DataFilterWarning, prune_index


def get_perc_growth_series(country: str, growth_df: pd.DataFrame, series_name: Optional[str] = None):
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
    A class for reading and retrieving economic data from EuroStat.

    Args:
        path (Path | str): The path to the directory containing the data files.
        country_code_path (Path | str): The path to the country code CSV file.

    Attributes:
        c_map (pd.DataFrame): The country code mapping DataFrame.
        files_with_codes (dict[str, str]): A dictionary mapping data file names to their codes.
        data (dict[str, pd.DataFrame]): A dictionary containing the loaded data files.

    Methods:
        get_files_with_codes(): Returns the dictionary mapping data file names to their codes.
        country_code_switch(codes): Converts 2-digit country codes to 3-digit country codes.
        find_value(df, country, year): Finds the value for a specific country and year in a DataFrame.
        nonfin_firm_debt_ratios(country, year): Returns the non-financial firm debt ratios for a specific country and year.
        get_total_nonfin_firm_debt(country, year): Returns the total non-financial firm debt for a specific country and year.
        get_total_fin_firm_debt(country, year): Returns the total financial firm debt for a specific country and year.
        nonfin_firm_deposit_ratios(country, year): Returns the non-financial firm deposit ratios for a specific country and year.
        get_quarterly_gdp(country, year, quarter): Returns the quarterly GDP for a specific country, year, and quarter.
        get_monthly_gdp(country, year, month): Returns the monthly GDP for a specific country, year, and month.
        get_total_nonfin_firm_deposits(country, year): Returns the total non-financial firm deposits for a specific country and year.
        get_total_bank_equity(country, year): Returns the total bank equity for a specific country and year.
        cb_debt_ratios(country, year): Returns the central bank debt ratios for a specific country and year.
        cb_equity_ratios(country, year): Returns the central bank equity ratios for a specific country and year.
        general_gov_debt_ratios(country, year): Returns the general government debt ratios for a specific country and year.
        central_gov_debt_ratios(country, year): Returns the central government debt ratios for a specific country and year.
        shortterm_interest_rates(country, year, months): Returns the short-term interest rates for a specific country, year, and number of months.
        longterm_central_gov_bond_rates(country, year): Returns the long-term central government bond rates for a specific country and year.
        dividend_payout_ratio(country, year): Returns the dividend payout ratio for a specific country and year.
        firm_risk_premium(country, year): Returns the firm risk premium for a specific country and year.
        number_of_households(country, year): Returns the number of households for a specific country and year.
        taxrate_on_capital_formation(country, year): Returns the tax rate on capital formation for a specific country and year.
        get_perc_sectoral_growth(country): Returns the percentage sectoral growth for a specific country.
        get_total_industry_debt_and_deposits(country_name): Returns the total industry debt and deposits for a specific country.
        get_imputed_rent_fraction_of_country(country, year): Returns the imputed rent fraction for a specific country and year.
        get_imputed_rent_fraction(country_names, year): Returns the imputed rent fraction for a specific list of countries and year.
        prune(prune_date): Prunes the data to only include data from a specific date.
    """

    def __init__(self, path: Path | str, country_code_path: Path | str):
        # Handle country codes
        self.c_map = pd.read_csv(country_code_path)
        # switch 2-digit code for Greece
        self.c_map.loc[self.c_map["Alpha-2 code"] == "GR", "Alpha-2 code"] = "EL"
        # switch 2-digit code United Kingdom
        self.c_map.loc[self.c_map["Alpha-2 code"] == "GB", "Alpha-2 code"] = "UK"
        # add Euro Area code
        self.c_map.loc[len(self.c_map)] = [
            "Euro Area",
            "EA",
            "EA",
            "",
            "",
        ]

        # Load data files
        self.files_with_codes = self.get_files_with_codes()
        self.data = {}
        for key in self.files_with_codes.keys():
            self.data[key] = pd.read_csv(path / (self.files_with_codes[key] + ".csv"))
            if "geo" in self.data[key].columns:
                self.data[key]["geo"] = self.country_code_switch(self.data[key]["geo"])

    @staticmethod
    def get_files_with_codes() -> dict[str, str]:
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
        }

    def country_code_switch(self, codes):
        """
        Converts a list of Alpha-2 country codes to Alpha-3 country codes.

        Args:
            codes (list): A list of Alpha-2 country codes.

        Returns:
            list: A list of corresponding Alpha-3 country codes.
        """
        return [self.c_map.loc[self.c_map["Alpha-2 code"] == c, "Alpha-3 code"].values[0] for c in codes]

    def find_value(self, df, country: str, year: str) -> float:
        """
        Find the value in the given DataFrame for the specified country and year.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            country (str): The name of the country.
            year (str): The year.

        Returns:
            float: The value found in the DataFrame.

        Raises:
            ValueError: If multiple data points are found for the given country and year.
        """
        res = df.loc[(df["geo"] == country) & (df["TIME_PERIOD"] == int(year)), "OBS_VALUE"].values
        if len(res) == 0:
            return self.find_value(df, country, str(int(year) + 1))
        elif len(res) == 1:
            return res[0]
        else:
            raise ValueError("Multiple inet_data points found in", df, country, year)

    def nonfin_firm_debt_ratios(self, country: str, year: int) -> float:
        """
        Calculate the non-financial firm debt ratio for a specific country and year.

        Args:
            country (str): The name of the country.
            year (int): The year for which the debt ratio is calculated.

        Returns:
            float: The non-financial firm debt ratio.

        """
        df = self.data["firm_debt_ratio"]
        return self.find_value(df, country, str(year)) / 100.0

    # historic domestic
    def get_total_nonfin_firm_debt(self, country: str, year: int) -> float:
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

    def get_total_fin_firm_debt(self, country: str, year: int) -> float:
        df = self.data["financial_balance_sheets"]
        country_name_short = self.c_map.loc[self.c_map["Alpha-3 code"] == country, "Alpha-2 code"].values[0]
        df = df.loc[
            df[r"unit,co_nco,sector,finpos,na_item,geo\time"] == "MIO_NAC,NCO,S12,LIAB,F4," + country_name_short
        ]
        return float(df[str(year)].values[0]) * 1e6

    def nonfin_firm_deposit_ratios(self, country: str, year: int) -> float:
        df = self.data["firm_deposits_ratio"]
        return self.find_value(df, country, str(year)) / 100.0

    def get_quarterly_gdp(self, country: str, year: int, quarter: int) -> float:
        df = self.data["gdp"]
        return (
            df.loc[
                (df["geo"] == country) & (df["TIME_PERIOD"] == f"{year}-Q{quarter}"),
                "OBS_VALUE",
            ].values[0]
            * 1e6
        )

    def get_monthly_gdp(self, country: str, year: int, month: int) -> float:
        start_quarter = (month - 1) // 3 + 1
        start = self.get_quarterly_gdp(country, year, start_quarter)

        if start_quarter == 4:
            end = self.get_quarterly_gdp(country, year + 1, 1)
        else:
            end = self.get_quarterly_gdp(country, year, start_quarter + 1)

        return start + (end - start) * ((month - 1) % 3) / 3

    # historic domestic
    def get_total_nonfin_firm_deposits(self, country: str, year: int) -> float:
        df = self.data["financial_balance_sheets"]
        country_name_short = self.c_map.loc[self.c_map["Alpha-3 code"] == country, "Alpha-2 code"].values[0]
        df = df.loc[df[r"unit,co_nco,sector,finpos,na_item,geo\time"] == "MIO_NAC,NCO,S11,ASS,F2," + country_name_short]
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

    # historic domestic
    def get_total_bank_equity(self, country: str, year: int) -> float:
        df = self.data["financial_balance_sheets"]
        country_name_short = self.c_map.loc[self.c_map["Alpha-3 code"] == country, "Alpha-2 code"].values[0]
        df = df.loc[
            df[r"unit,co_nco,sector,finpos,na_item,geo\time"] == "MIO_NAC,NCO,S122_S123,ASS,F5," + country_name_short
        ]
        return float(df[str(year)].values[0]) * 1e6

    def cb_debt_ratios(self, country: str, year: int) -> float:
        df = self.data["central_bank_debt_ratio"]
        return self.find_value(df, country, str(year)) / 100.0

    def cb_equity_ratios(self, country: str, year: int) -> float:
        df = self.data["central_bank_equity_ratio"]
        return self.find_value(df, country, str(year)) / 100.0

    def general_gov_debt_ratios(self, country: str, year: int) -> float:
        df = self.data["general_gov_debt_ratio"]
        return self.find_value(df, country, str(year)) / 100

    def central_gov_debt_ratios(self, country: str, year: int) -> float:
        df = self.data["central_gov_debt_ratio"]
        return self.find_value(df, country, str(year)) / 100

    def shortterm_interest_rates(self, country: str, year: int, months: int) -> float:
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

    def longterm_central_gov_bond_rates(self, country: str, year: int) -> float:
        df = self.data["longterm_central_gov_bond_rates"]
        return self.find_value(df, country, str(year)) / 100

    # in domestic
    # numerator only available in current prices
    # denominator only available in historic prices
    def dividend_payout_ratio(self, country: str, year: int) -> float:
        """
        Calculate the dividend payout ratio for a given country and year.

        Parameters:
        - country (str): The name of the country.
        - year (int): The year for which to calculate the dividend payout ratio.

        Returns:
        - float: The dividend payout ratio.
        """
        df = self.data["nonfinancial_transactions"]

        hh_prop_df = df[(df["na_item"] == "D4") & (df["direct"] == "RECV") & (df["sector"] == "S14")]

        hh_surplus_df = df[(df["na_item"] == "B2A3N") & (df["direct"] == "RECV") & (df["sector"] == "S14")]

        df = self.data["iot_tables"]
        firm_surplus_df = df[(df["induse"] == "TOTAL") & (df["prod_na"] == "B2A3N")]

        hh_prop = self.find_value(hh_prop_df, country, str(year))
        hh_surplus = self.find_value(hh_surplus_df, country, str(year))
        firm_surplus = self.find_value(firm_surplus_df, country, str(year))

        return (hh_prop + hh_surplus) / firm_surplus

    def firm_risk_premium(self, country: str, year: int) -> float:
        """
        Calculate the firm risk premium for a given country and year.

        Parameters:
        country (str): The country for which the risk premium is calculated.
        year (int): The year for which the risk premium is calculated.

        Returns:
        float: The calculated firm risk premium.
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

    def number_of_households(self, country: str, year: int) -> float:
        df = self.data["number_of_households"].set_index("geo")
        return int(df.loc[country, str(year)] * 1000)

    def taxrate_on_capital_formation(self, country: str, year: int) -> float:
        capform_df = self.data["capital_formation"]

        df = self.data["iot_tables"]
        taxes_df = df[(df["induse"] == "P5") & (df["prod_na"] == "D21X31")]

        capform = self.find_value(capform_df, country, str(year))
        taxes = self.find_value(taxes_df, country, str(year))

        return taxes / capform

    def get_perc_sectoral_growth(self, country: str) -> pd.DataFrame:
        """
        Retrieves the percentage sectoral growth data for a specific country.

        Args:
            country (str): The name of the country.

        Returns:
            pd.DataFrame: A DataFrame containing the percentage sectoral growth data.
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

    def get_total_industry_debt_and_deposits(self, country_name: str) -> pd.DataFrame:
        dates, total_deposits, total_debt = [], [], []
        for year in range(1970, 2024):
            dep = self.get_total_nonfin_firm_deposits(country_name, year)
            debt = self.get_total_nonfin_firm_debt(country_name, year)
            for month in range(1, 13):
                dates.append(str(year) + "-" + str(month))
                total_deposits.append(dep)
                total_debt.append(debt)
        return pd.DataFrame(
            index=dates,
            data={
                "Total Deposits": total_deposits,
                "Total Debt": total_debt,
            },
        )

    def get_imputed_rent_fraction_of_country(self, country: str, year: int) -> float:
        df = self.data["real_estate_services"].set_index("freq,unit,stk_flow,induse,prod_na,geo\TIME_PERIOD")
        country_name_short = self.c_map.loc[self.c_map["Alpha-3 code"] == country, "Alpha-2 code"].values[0]
        return float(df.at["A,MIO_NAC,TOTAL,P3_S14,CPA_L68A," + country_name_short, str(year)]) / (
            float(df.at["A,MIO_NAC,TOTAL,P3_S14,CPA_L68A," + country_name_short, str(year)])
            + float(df.at["A,MIO_NAC,TOTAL,P3_S14,CPA_L68B," + country_name_short, str(year)])
        )

    def get_imputed_rent_fraction(
        self,
        country_names: list[str],
        year: int,
    ) -> dict[str, float]:
        fractions = {c: self.get_imputed_rent_fraction_of_country(c, year) for c in country_names}
        fractions["ROW"] = np.mean(list(fractions.values()))  # c'est la vie
        return fractions

    def prune(self, prune_date: date):
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
                mask = prune_index(value.columns, prune_date, f"Eurostat {key}")
                self.data[key] = value.loc[:, mask]
        return self


def convert_date(date: str | int):
    if isinstance(date, int):
        return pd.to_datetime(date, format="%Y")
    # Check if the date is in quarterly format like '2012-Q1'
    if "Q" in date:
        year, quarter = date.split("-Q")
        month = (int(quarter) - 1) * 3 + 1  # Convert quarter to month
        return pd.to_datetime(f"{year}-{month:02d}-01")  # Format: YYYY-MM-DD
    else:
        # For other formats, directly use to_datetime
        return pd.to_datetime(date, errors="coerce", yearfirst=True)
