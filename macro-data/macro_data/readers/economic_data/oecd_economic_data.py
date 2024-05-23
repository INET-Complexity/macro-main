import json
import logging
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import zetac

from macro_data.configuration.countries import Country
from macro_data.readers.util.prune_util import DataFilterWarning

force_tau_sif = {
    "AUS": 0.0,
    "CHL": 0.0,
    "CRI": 0.0,
    "DNK": 0.0,
    "GRC": 0.22,
    "LVA": 0.0,
    "LTU": 0.015,
    "TUR": 0.0,
    "KHM": 0.0,
    "KOR": 0.0,
    "COL": 0.0,
    "MEX": 0.0,
    "IND": 0.0,
    "IDN": 0.0,
    "LAO": 0.0,
    "MMR": 0.0,
    "VNM": 0.0,
}

# tau siw
force_tau_siw = {
    "AUS": 0.0,
    "CHL": 0.0,
    "CRI": 0.0,
    "COL": 0.0,
    "DNK": 0.0,
    "GRC": 0.14,
    "LVA": 0.0,
    "LTU": 0.07,
    "TUR": 0.0,
    "KHM": 0.0,
    "KOR": 0.0,
    "MEX": 0.0,
    "IND": 0.0,
    "IDN": 0.0,
    "LAO": 0.0,
    "MMR": 0.0,
    "VNM": 0.0,
}

# tau firm
force_tau_firm = {
    "KHM": 0.0,
    "CYP": 0.125,
    "KAZ": 0.2,
    "LAO": 0.0,
    "MAR": 0.2,
    "MMR": 0.0,
    "PHL": 0.25,
    "IDN": 0.0,
    "TUN": 0.15,
    "TWN": 0.2,
    "CRI": 0.0,
    "KOR": 0.0,
    "MEX": 0.0,
    "VNM": 0.0,
    "CHL": 0.0,
    "COL": 0.0,
}

force_tau_income = {
    "ARG": 0.04,
    "BRA": 0.075,
    "BRN": 0.0,
    "BGR": 0.1,
    "KHM": 0.0,
    "CHN": 0.1,
    "CRI": 0.0,
    "HRV": 0.2,
    "CHL": 0.0,
    "CYP": 0.2,
    "IND": 0.05,
    "IDN": 0.0,
    "HKG": 0.1,
    "KAZ": 0.1,
    "KOR": 0.0,
    "MEX": 0.0,
    "LAO": 0.0,
    "MYS": 0.08,
    "MLT": 0.1,
    "MAR": 0.12,
    "MMR": 0.0,
    "PER": 0.17,
    "PHL": 0.05,
    "ROU": 0.1,
    "RUS": 0.13,
    "SAU": 0.2,
    "SGP": 0.1,
    "ZAF": 0.18,
    "TWN": 0.05,
}


class OECDEconData:
    def __init__(
        self,
        path: Path | str,
        industry_mappings_path: Path,
        sector_mapping_path: Path,
        scale_dict: dict[Country, int],
    ):
        # Parameters
        self.scale_dict = scale_dict
        self.industry_mapping = json.load(open(industry_mappings_path))
        self.sector_mapping = json.load(open(sector_mapping_path))

        # Load data files
        self.files_with_codes = self.get_files_with_codes()
        self.data = {
            key: pd.read_csv(path / (self.files_with_codes[key] + ".csv")) for key in self.files_with_codes.keys()
        }

        self.default_industries = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
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
            "T",
        ]

    @staticmethod
    def get_files_with_codes() -> dict[str, str]:
        return {
            "employment_by_industry": "ALFS_EMP",
            "bank_income_statement_balance_sheet": "BPF1",
            "corporate_income_tax_rate": "CTS_CIT",
            "total_social_benefits_perc_gdp": "SOCX_AGG",
            "total_unemployment_benefits_perc_gdp": "DP_LIVE_UNEMP",
            "business_demography": "SDBS_BDI_ISIC4",
            "business_demography1": "SSIS_BSC_ISIC4",
            "business_birth_rates": "SDBS_BDI_ISIC4_BIRTH",
            "business_death_rates": "SDBS_BDI_ISIC4_DEATH",
            "business_sizes": "SSIS_BSC_ISIC4",
            "employers_contribution_social_insurance": "TABLE_III2",
            "employees_contribution_social_insurance": "TABLE_III1",
            "average_personal_income_tax_by_family_type": "TABLE_I6",
            "top_statutory_personal_income_tax": "TABLE_I7",
            "gini_coefficients": "GINI_COEF",
            "ppi": "DP_LIVE_PPI_INFL",
            "bank_demography": "oecd_bank_data",
            "bank_capital_requirements": "oecd_bank_capital_reqs",
            "long_term_interest_rates": "DP_LIVE_21042023152525378",
            "short_term_interest_rates": "DP_LIVE_21042023152700149",
            "general_gov_debt": "SNA_TABLE750_24042023172303744",
            "unemployment_rates": "unemployment_rates",
            "consumption_by_income_quintiles": "consumption_by_income_quintiles",
            "housing_index": "HOUSE_PRICES",
            "active_population_size": "STLABOUR",
            "total_job_vacancies": "LAB_REG_VAC",
            "experimental_consumption_statistics": "EGDNA_PUBLIC",
            "gross_gov_debt_usd_ppp": "oecd_govt_debt_usd_ppp",
            "KEI": "KEI",
            "QNA": "QNA",
        }

    def employees_by_industry(self, year: int, country: Country) -> pd.Series:
        df = self.data["employment_by_industry"]
        df = df.loc[(df["Time"] == year) & (df["SEX"] == "TT") & (df["LOCATION"] == country)].copy()
        # this scary looking line is just a regular expression
        # it captures any letter in a parenthesis for the industry names
        # ie if something like Agriculture (A) , it will get A
        df.loc[:, "industry"] = df["Subject"].str.extract(r"\([,]?\s?([\w+]*)\s?\)")[0].replace(["R", "S"], "R_S")
        df = df.dropna(subset="industry")
        industry_indices = {s: idx for idx, s in enumerate(self.default_industries)}
        df["industry_indices"] = df["industry"].map(industry_indices)
        col_data = df.groupby(["industry_indices"])["Value"].sum()
        col_data = col_data.reindex(range(len(self.default_industries)))
        col_data.fillna(col_data.median(), inplace=True)
        return col_data.values.astype("int") / self.scale_dict[country]

    def read_business_demography(
        self,
        country: Country,
        output: pd.Series,
        year: int,
    ) -> np.ndarray:
        """
        Reads business individuals data.

        Parameters
        ----------
        country : str
            The list of countries.
        output : pd.Series
            The total output of each country by industry.
        year : int
            The current year.

        Returns
        -------
        n_firms_in_industry : dict
            The number of active employer enterprises by country and industry.
        """

        # Load data

        # TODO: OECD data doesn't have US data for this, so we use Canada as a proxy
        # if country in {"USA", "MEX"}:
        #     data_country = Country("CAN")
        # else:
        #     data_country = country

        if country == "GBR" and year < 2014:
            year = 2014
        if country == "GBR" and year > 2018:
            year = 2018
        if country == "DEU" and year < 2012:
            year = 2012
        if country == "AUS" and year > 2010:
            year = 2014

        df = self.data["business_demography1"].copy()
        df = df.loc[
            (df["VAR"] == "ENTR")
            & (df["SRC"] == "SSIS")
            & (df["TIME"] == year)
            & (df["Size Class"] == "Total")
            & (df["LOCATION"] == country)
        ].copy()

        output.index = range(len(output))
        output.index = range(len(output))
        df.loc[:, "ISIC"] = df["ISIC4"].copy().map(self.industry_mapping)
        df.dropna(subset="ISIC", inplace=True)
        isic_table = df.set_index(["ISIC", "LOCATION"])["Value"].sort_index().unstack()
        isic_table = isic_table.reindex(range(len(output))).fillna(0)

        isic_table = isic_table[country]

        # basic linear regression to fill missing values in
        # number of businesses
        missing_data = isic_table == 0
        fit_params = np.polyfit(
            x=output.loc[~missing_data],
            y=isic_table.loc[~missing_data],
            deg=1,
        )
        isic_table.loc[missing_data] = output.loc[missing_data].apply(lambda x: fit_params[0] * x + fit_params[1])

        isic_table /= self.scale_dict[country]

        isic_table = isic_table.astype("int")

        # if any values are 0 set them to 1
        isic_table[isic_table == 0] = 1

        return isic_table.values.astype(int)

    @staticmethod
    def zeta_dist(x, a):
        z = 1 / (x**a * zetac(a))
        return z / sum(z)

    def find_sector_code(self, code):
        for sector, codes in self.sector_mapping.items():
            sector_string = "".join(codes)
            subsecs = code.split("_")
            if np.sum([s in sector_string for s in subsecs]) == len(subsecs):
                return sector
        return None

    def read_firm_size_zetas(
        self,
        country: str,
        year: int,
    ) -> dict[int, np.ndarray] | None:
        sizes = ["1-9", "10-19", "20-49", "50-249", "250"]
        size_means = [np.mean([int(v) for v in s.split("-")]) for s in sizes]

        df = self.data["business_demography1"]
        df = df.loc[(df["LOCATION"] == country) & (df["TIME"] == year) & (df["VAR"] == "ENTR")]
        if len(df) == 0:
            return None

        ind_zetas = {}
        for ind in df["ISIC4"].unique():
            try:
                counts = []
                for size in sizes:
                    count = df.loc[
                        (df["ISIC4"] == ind) & (df["Size Class"].map(lambda x: x[: len(size)]) == size),
                        "Value",
                    ].values[0]
                    assert not np.isnan(count)
                    counts.append(count)
            except AssertionError:
                continue
            except IndexError:
                continue

            counts = np.array(counts)
            if np.sum(counts) == 0:
                ind_zetas[ind] = np.mean(list(ind_zetas.values()))
            else:
                freq = counts / np.sum(counts)
                ind_zetas[ind] = curve_fit(self.zeta_dist, size_means, freq, p0=[0.1])[0][0]

        isic_zetas = {self.find_sector_code(k): v for k, v in ind_zetas.items()}
        final_zetas = {}
        for ind in self.default_industries:
            if ind in isic_zetas.keys():
                final_zetas[self.default_industries.index(ind)] = isic_zetas[ind]
            # missing data takes the mean shape parameter
            else:
                final_zetas[self.default_industries.index(ind)] = np.mean(list(isic_zetas.values()))

        return final_zetas

    @staticmethod
    def find_closest_year(df: pd.DataFrame, year: int):
        years = df["YEA"].unique()
        min_year = min(years, key=lambda x: abs(x - year))
        return df.loc[df["YEA"] == min_year]

    def read_tau_sif(self, country: Country | str, year: int) -> float:
        if isinstance(country, str):
            country = Country(country)
        if country.value in force_tau_sif:
            return force_tau_sif[country.value]
        df = self.data["employers_contribution_social_insurance"]
        df = df.loc[(df["COU"] == country) & (df["YEA"] == year)]
        if len(df) == 0:
            df = self.find_closest_year(df, year)
        return df.loc[df["RATE_THRESH"] == "01_MR", "Value"].iloc[0] / 100.0

    def read_tau_siw(self, country: Country | str, year: int) -> float:
        if isinstance(country, str):
            country = Country(country)
        if country.value in force_tau_siw:
            return force_tau_siw[country.value]
        df = self.data["employees_contribution_social_insurance"]
        df = df.loc[(df["COU"] == country) & (df["YEA"] == year)]
        if len(df) == 0:
            df = self.find_closest_year(df, year)
        return df.loc[df["RATE_THRESH"] == "01_MR", "Value"].iloc[0] / 100.0

    def read_tau_firm(self, country: Country | str, year: int) -> float:
        if isinstance(country, str):
            country = Country(country)
        if country.value in force_tau_firm:
            return force_tau_firm[country.value]
        df = self.data["corporate_income_tax_rate"]
        df = df.loc[(df["COU"] == country) & (df["YEA"] == year)]
        if len(df) == 0:
            df = self.find_closest_year(df, year)
        df.set_index("CORP_TAX", inplace=True)
        return df.loc["COMB_CIT_RATE", "Value"] / 100.0

    def read_tau_income(self, country: Country, year: int) -> float:
        # df = self.data["average_personal_income_tax_by_family_type"]
        # df = df.loc[df["COU"] == country]
        # df = df.loc[df["Year"] == year]
        # df = df.loc[df["ALL_IN"] == "ALL_IN_RATE_SING_NO_CH"]
        # return df["Value"].values[0] / 100.0

        if country in force_tau_income:
            return force_tau_income[country]
        else:
            return 0.09  # OECD average

    def read_short_term_interest_rates(self, country: Country, year: int) -> float:
        df = self.data["short_term_interest_rates"]
        df = df.loc[df["LOCATION"] == country]
        df = df.loc[df["FREQUENCY"] == "Q"]
        df = df.loc[df["TIME"] == str(year) + "-Q1"]
        val = df["Value"].values
        if len(val) == 0:
            return np.nan
        elif len(val) == 1:
            return val[0] / 100.0
        else:
            raise ValueError("Multiple values found for short-term interest rates.")

    def read_long_term_interest_rates(self, country: Country, year: int) -> float:
        df = self.data["long_term_interest_rates"]
        df = df.loc[df["LOCATION"] == country]
        df = df.loc[df["FREQUENCY"] == "Q"]
        df = df.loc[df["TIME"] == str(year) + "-Q1"]
        val = df["Value"].values
        if len(val) == 0:
            return np.nan
        elif len(val) == 1:
            return val[0] / 100.0
        else:
            raise ValueError("Multiple values found for long-term interest rates.")

    def get_bank_demographics(self, country: Country, year: int, code: str) -> float:
        df = self.data["bank_demography"]

        sel = df.loc[(df["COU"] == country) & (df["ITEM"] == code)]

        if len(sel) == 0:
            sel = df.loc[df["ITEM"] == code, "Value"]
            return int(sel.mean())
        else:
            if year in sel["YEA"].values:
                return sel.loc[sel["YEA"] == year, "Value"].iloc[0]
            else:
                return sel["Value"].iloc[-1]

    def read_tierone_reserves(self, country: Country, year: int):
        # noinspection PyTypeChecker
        reserves = self.get_bank_demographics(country, year, "BC32TE")
        rwa = self.get_bank_demographics(country, year, "BC36TE")
        return reserves / rwa

    def read_number_of_banks(self, country: Country, year: int) -> int:
        return int(self.get_bank_demographics(country, year, "SI37TE"))

    def read_number_of_bank_branches(self, country: Country, year: int) -> int:
        return int(self.get_bank_demographics(country, year, "SI38TE"))

    def read_number_of_bank_employees(self, country: Country, year: int) -> int:
        return int(self.get_bank_demographics(country, year, "SI39TE"))

    # current domestic
    def read_bank_distributed_profit(self, country: Country, year: int):
        return self.get_bank_demographics(country, year, "IN12TE") * 1000000

    # current domestic
    def read_bank_retained_profit(self, country: Country, year: int):
        return self.get_bank_demographics(country, year, "IN13TD") * 1000000

    # current domestic
    def read_bank_total_assets(self, country: Country, year: int):
        return self.get_bank_demographics(country, year, "BT25TE") * 1000000

    def unemployment_benefits_gdp_pct(self, country: Country, year: int) -> float:
        df = self.data["total_unemployment_benefits_perc_gdp"]
        if country.value in df["LOCATION"].values:
            value = df.loc[(df["LOCATION"] == country) & (df["TIME"] == year), "Value"].iloc[0]
            return value / 100.0
        else:
            return df.loc[df["TIME"] == year, "Value"].mean() / 100.0

    def all_benefits_gdp_pct(self, country: str, year: int, average_oecd: float = 0.212) -> float:
        if country == "ARG":
            country = "CHL"
        if country == "BRA":
            return 0.367
        if country == "JPN":  # https://www.ids.ac.uk/download.php?file=files/dmfile/SocialProtectioninSouthAsia.pdf
            return 0.16
        if country == "LKA":  # https://www.ids.ac.uk/download.php?file=files/dmfile/SocialProtectioninSouthAsia.pdf
            return 0.057
        if country == "IND":  # https://www.ids.ac.uk/download.php?file=files/dmfile/SocialProtectioninSouthAsia.pdf
            return 0.1
        if country == "BGD":  # https://www.ids.ac.uk/download.php?file=files/dmfile/SocialProtectioninSouthAsia.pdf
            return 0.053
        if country == "NPL":  # https://www.ids.ac.uk/download.php?file=files/dmfile/SocialProtectioninSouthAsia.pdf
            return 0.023
        if country == "PAK":  # https://www.ids.ac.uk/download.php?file=files/dmfile/SocialProtectioninSouthAsia.pdf
            return 0.016
        if country == "BRN":  # https://databankfiles.worldbank.org/public/ddpext_download/hci/HCI_2pager_BRN.pdf
            return 0.034
        if country == "CEN":
            return 0.0
        if country == "GRC":  # https://data.oecd.org/socialexp/social-benefits-to-households.htm
            return 0.29
        if country == "LVA":
            return 0.19
        df = self.data["total_social_benefits_perc_gdp"]
        val = df.loc[(df["COUNTRY"] == country) & (df["YEAR"] == year), "Value"].values
        if len(val) == 0 or np.isnan(val[0]):
            return average_oecd
        else:
            return val[0] / 100.0

    # current domestic
    def general_gov_debt(self, country: Country, year: int) -> float:
        df = self.data["general_gov_debt"]
        value = df.loc[(df["LOCATION"] == country) & (df["TIME"] == year), "Value"].iloc[0]
        return value * 1e6

    def get_unemployment_rate(self, country: str) -> pd.DataFrame:
        data = self.data["unemployment_rates"].loc[
            (self.data["unemployment_rates"]["LOCATION"] == country)
            & (self.data["unemployment_rates"]["SUBJECT"] == "TOT")
            & (self.data["unemployment_rates"]["FREQUENCY"] == "Q")
        ]
        dates, vals = [], []
        for year in range(1970, 2024):
            for quarter in range(1, 5):
                dates.append(str(year) + "-Q" + str(quarter))
                val = data.loc[data["TIME"] == str(year) + "-Q" + str(quarter), "Value"].values
                if len(val) == 0:
                    vals.append(np.nan)
                else:
                    vals.append(val[0] / 100.0)
        data = pd.DataFrame(
            index=dates,
            data={"Unemployment Rate": vals},
        )
        data.index = [pd.Timestamp(int(ind[0:4]), 3 * int(ind[6]) - 2, 1) for ind in data.index]
        return data

    def get_consumption_rates_by_income(self, country: Country) -> pd.DataFrame:
        df = self.data["consumption_by_income_quintiles"]
        df["country_year"] = [df["country_year"][i][0:3] for i in range(len(df))]
        df = df.loc[df["country_year"] == country]
        df = df.set_index("industry")
        df = df.loc[:, df.columns != "country_year"]
        df.index.name = "Industry"
        df.columns.name = "Income Quantile"

        return df

    def get_house_price_index(self, country: str) -> pd.DataFrame:
        df = self.data["housing_index"]
        dates, val_real, val_nominal = [], [], []
        for year in range(1970, 2024):
            for quarter in range(1, 5):
                if quarter == 1:
                    prev_year = year - 1
                    prev_quarter = 4
                else:
                    prev_year = year
                    prev_quarter = quarter - 1
                curr_value_real = df.loc[
                    (df["COU"] == country) & (df["IND"] == "RHP") & (df["TIME"] == str(year) + "-Q" + str(quarter)),
                    "Value",
                ].values
                prev_value_real = df.loc[
                    (df["COU"] == country)
                    & (df["IND"] == "RHP")
                    & (df["TIME"] == str(prev_year) + "-Q" + str(prev_quarter)),
                    "Value",
                ].values
                curr_value_nominal = df.loc[
                    (df["COU"] == country) & (df["IND"] == "HPI") & (df["TIME"] == str(year) + "-Q" + str(quarter)),
                    "Value",
                ].values
                prev_value_nominal = df.loc[
                    (df["COU"] == country)
                    & (df["IND"] == "HPI")
                    & (df["TIME"] == str(prev_year) + "-Q" + str(prev_quarter)),
                    "Value",
                ].values
                dates.append(str(year) + "-Q" + str(quarter))
                if len(curr_value_real) == 1 and len(prev_value_real) == 1:
                    val_real.append(curr_value_real[0] / prev_value_real[0] - 1.0)
                else:
                    val_real.append(np.nan)
                if len(curr_value_nominal) == 1 and len(prev_value_nominal) == 1:
                    val_nominal.append(curr_value_nominal[0] / prev_value_nominal[0] - 1.0)
                else:
                    val_nominal.append(np.nan)
        data = pd.DataFrame(
            index=dates,
            data={
                "Real House Price Index Growth": val_real,
                "Nominal House Price Index Growth": val_nominal,
            },
        )
        data.index = [pd.Timestamp(int(ind[0:4]), 3 * int(ind[6]) - 2, 1) for ind in data.index]
        return data

    def get_vacancy_rate(self, country: Country) -> pd.DataFrame:
        active_population_size = self.data["active_population_size"]
        total_job_vacancies = self.data["total_job_vacancies"]
        dates, vacancy_rate = [], []
        for year in range(1970, 2024):
            for quarter in range(1, 5):
                pop_size = (
                    1000
                    * active_population_size.loc[
                        (active_population_size["LOCATION"] == country)
                        & (active_population_size["TIME"] == str(year) + "-Q" + str(quarter)),
                        "Value",
                    ].values
                )
                total_vacs = total_job_vacancies.loc[
                    (total_job_vacancies["LOCATION"] == country)
                    & (total_job_vacancies["TIME"] == str(year) + "-Q" + str(quarter)),
                    "Value",
                ].values
                dates.append(str(year) + "-Q" + str(quarter))
                if len(pop_size) == 1 and len(total_vacs) == 1:
                    vacancy_rate.append(total_vacs[0] / pop_size[0])
                else:
                    vacancy_rate.append(np.nan)
        data = pd.DataFrame(index=dates, data={"Vacancy Rate": vacancy_rate})
        data.index = [pd.Timestamp(int(ind[0:4]), 3 * int(ind[6]) - 2, 1) for ind in data.index]
        return data

    def get_household_consumption_by_income_quantile(self, country: Country, year: int) -> pd.DataFrame:
        assert year
        data = self.data["consumption_by_income_quintiles"]
        if country != "FRA":
            logging.warning("Overwriting Consumption Weights by Income with French Data")
        country = "FRA"
        data = data.loc[data["country_year"].str.contains(country)]
        data = data.set_index("industry")[["Q1", "Q2", "Q3", "Q4", "Q5"]]
        data /= data.sum(axis=0)
        data.index = pd.Index(range(len(data)), name="Industry")
        return data

    def get_govt_debt_usd_ppp(self, country: Country, year: int) -> float:
        df = self.data["gross_gov_debt_usd_ppp"]
        df = df[df["Financial instrument"] == "Total"]
        df = df[df["REF_AREA"] == country]
        return df.set_index("TIME_PERIOD").loc[year, "OBS_VALUE"] * 1e6

    def get_inflation(self, country: str) -> pd.DataFrame:
        data = self.data["KEI"]
        data = data.loc[data["LOCATION"] == country]
        dates, cpi, ppi = [], [], []
        for year in range(1970, 2024):
            for quarter in range(1, 5):
                dates.append(str(year) + "-Q" + str(quarter))
                curr_data = data.loc[data["TIME"] == str(year) + "-Q" + str(quarter)]
                cpi_data = curr_data.loc[curr_data["SUBJECT"] == "CPALTT01"]["Value"].values
                if len(cpi_data) == 1:
                    cpi.append(cpi_data[0])
                else:
                    cpi.append(np.nan)
                ppi_data = curr_data.loc[curr_data["SUBJECT"] == "PIEAMP01"]["Value"].values
                if len(ppi_data) == 1:
                    ppi.append(ppi_data[0])
                else:
                    ppi.append(np.nan)
        if np.isnan(cpi).sum() == len(cpi):
            data = pd.DataFrame(data={"CPI Inflation": ppi, "PPI Inflation": ppi}, index=dates)
        elif np.isnan(ppi).sum() == len(ppi):
            data = pd.DataFrame(data={"CPI Inflation": cpi, "PPI Inflation": cpi}, index=dates)
        else:
            data = pd.DataFrame(data={"CPI Inflation": cpi, "PPI Inflation": ppi}, index=dates)
        data.index = [pd.Timestamp(int(ind[0:4]), 3 * int(ind[6]) - 2, 1) for ind in data.index]
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            return data.pct_change()

    def get_na_growth_rates(self, country: str) -> pd.DataFrame:
        data = self.data["QNA"]  # CQRSA
        data = data.loc[(data["LOCATION"] == country) & (data["FREQUENCY"] == "Q")]
        # if len(data) == 0:
        #     return self.get_na_growth_rates(self.proxy_country)  # won't need this anymore
        na_data = data.pivot(index="TIME", columns="SUBJECT", values="Value")
        fields = {
            "GDP": "B1_GE",
            "HH Cons": "P31S14_S15",
            "Gov Cons": "P3S13",
            "Gross Fixed Capital Formation": "P51",
            "Changes in Inventories": "P52",
            # "Acquisitions less Disposals of Valuables": "P53",
            "Exports": "P6",
            "Imports": "P7",
            "Compensation of Employees": "D1S1",
            "Gross Operating Surplus and Mixed Income": "B2G_B3G",
            # "Taxes less Subsidies on Production and Imports": "D2_D3",
            "Gross Value Added": "B1G",
            "Gross Value Added - A": "B1GVA",
            "Gross Value Added - B, C, D, E": "B1GVB_E",
            "Gross Value Added - C": "B1GVC",
            "Gross Value Added - F": "B1GVF",
            "Gross Value Added - G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U": "B1GVG_U",
            "Gross Value Added - G, H, I": "B1GVG_I",
            "Gross Value Added - J": "B1GVJ",
            "Gross Value Added - K": "B1GVK",
            "Gross Value Added - L": "B1GVL",
            "Gross Value Added - M, N": "B1GVM_N",
            "Gross Value Added - O, P, Q": "B1GVO_Q",
            "Gross Value Added - R, S, T, U": "B1GVR_U",
            "Taxes less Subsidies on Production": "D21_D31",
        }

        # Get values
        cols, col_desc = [], []
        for col in fields.keys():
            if fields[col] in na_data.columns:
                cols.append(fields[col])
                col_desc.append(col)
        na_data = na_data.loc[:, cols]
        na_data.columns = col_desc
        na_data = 1000.0 * na_data

        # Calculate growth rates
        na_data = (na_data / na_data.shift(1) - 1.0).iloc[1:]
        if "GDP" in na_data.columns:
            na_data["Gross Output"] = na_data["GDP"].values
            na_data["Intermediate Consumption"] = na_data["GDP"].values
            for col in fields.keys():
                if col not in na_data.columns:
                    na_data[col] = na_data["GDP"].values

        # Change index
        na_data.index = [pd.Timestamp(int(ind[0:4]), 3 * int(ind[6]) - 2, 1) for ind in na_data.index]  # noqa

        return na_data

    def prune(self, prune_date: date):
        for key, value in self.data.items():
            for col in value.columns:
                if col.lower() in ["year", "time"]:
                    dates = pd.to_datetime(value[col].astype(str), errors="coerce", format="mixed")
                    if dates.isnull().sum() == 0:
                        mask = dates >= pd.to_datetime(prune_date)
                        if mask.sum() == 0:
                            warnings.warn(
                                f"No rows after {prune_date} in OECD dataset {key}; No filter applied.",
                                DataFilterWarning,
                            )
                            mask = np.ones(len(value), dtype=bool)
                        self.data[key] = value.loc[mask, :]
                        break
                if col == "country_year":
                    years = value[col].apply(lambda x: x.split("_")[1])
                    years = pd.to_datetime(years, errors="coerce", format="%Y")
                    mask = years >= pd.to_datetime(prune_date)
                    if mask.sum() == 0:
                        warnings.warn(
                            f"No rows were kept for date {prune_date} in OECD dataset {key}.",
                            DataFilterWarning,
                        )
                    self.data[key] = value.loc[mask, :]
                    break


def prune_oecd(oecd_econ, start_date):
    # OECD
    for key, value in oecd_econ.data.items():
        for col in value.columns:
            if col.lower() in ["year", "time"]:
                # Check if column can be transformed in a date
                dates = pd.to_datetime(value[col].astype(str), errors="coerce")
                if dates.isnull().sum() == 0:
                    mask = dates >= pd.to_datetime(f"{start_date}-01-01")
                    if mask.sum() == 0:
                        warnings.warn(
                            f"No rows after {start_date} in OECD dataset {key}; No filter applied.",
                            DataFilterWarning,
                        )
                        mask = np.ones(len(value), dtype=bool)
                    oecd_econ.data[key] = value.loc[mask, :]
                    break
            if col == "country_year":
                mask = value[col].apply(lambda x: x.split("_")[1]) >= f"{start_date}"
                if mask.sum() == 0:
                    warnings.warn(
                        f"No rows were kept for date {start_date} in OECD dataset {key}.",
                        DataFilterWarning,
                    )
                oecd_econ.data[key] = value.loc[mask, :]
                break
    return oecd_econ
