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
        df = self.data["business_demography"]
        df = df.loc[
            (df["IND"] == "ENTR_BD_EMPL")
            & (df["TIME"] == year)
            & (df["Size Class"] == "Total")
            & (df["LOCATION"] == country)
        ].copy()

        output.index = range(len(output))
        df.loc[:, "ISIC"] = df["SEC"].copy().map(self.industry_mapping)
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

        df = self.data["business_sizes"]
        df = df.loc[(df["LOCATION"] == country) & (df["TIME"] == year)]
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

    def read_tau_sif(self, country: Country, year: int) -> float:
        df = self.data["employers_contribution_social_insurance"]
        df = df.loc[(df["COU"] == country) & (df["YEA"] == year)]
        return df.loc[df["RATE_THRESH"] == "01_MR", "Value"].iloc[0] / 100.0

    def read_tau_siw(self, country: Country, year: int) -> float:
        df = self.data["employees_contribution_social_insurance"]
        df = df.loc[(df["COU"] == country) & (df["YEA"] == year)]
        return df.loc[df["RATE_THRESH"] == "01_MR", "Value"].iloc[0] / 100.0

    def read_tau_firm(self, country: Country, year: int) -> float:
        df = self.data["corporate_income_tax_rate"]
        df = df.loc[(df["COU"] == country) & (df["YEA"] == year)]
        df.set_index("CORP_TAX", inplace=True)
        return df.loc["COMB_CIT_RATE", "Value"] / 100.0

    def read_tau_income(self, country: Country, year: int) -> float:
        df = self.data["average_personal_income_tax_by_family_type"]
        df = df.loc[df["COU"] == country]
        df = df.loc[df["Year"] == year]
        df = df.loc[df["ALL_IN"] == "ALL_IN_RATE_SING_NO_CH"]
        return df["Value"].values[0] / 100.0

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
        res = df.loc[
            (df["COU"] == country) & (df["YEA"] == year) & (df["ITEM"] == code),
            "Value",
        ].values
        if len(res) == 0:
            return self.get_bank_demographics(country, year - 1, code)
        elif len(res) == 1:
            return res[0]
        else:
            raise ValueError(
                "Multiple values when fetching bank demography inet_data",
                country,
                year,
                code,
            )

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
        value = df.loc[(df["LOCATION"] == country) & (df["TIME"] == year), "Value"].iloc[0]
        return value / 100.0

    def all_benefits_gdp_pct(self, country: Country, year: int) -> float:
        all_benefits = self.data["total_social_benefits_perc_gdp"]
        value = all_benefits.loc[
            (all_benefits["COUNTRY"] == country) & (all_benefits["YEAR"] == year),
            "Value",
        ].iloc[0]
        return value / 100.0

    # current domestic
    def general_gov_debt(self, country: Country, year: int) -> float:
        df = self.data["general_gov_debt"]
        value = df.loc[(df["LOCATION"] == country) & (df["TIME"] == year), "Value"].iloc[0]
        return value * 1e6

    def get_unemployment_rate(self, country: Country) -> pd.DataFrame:
        data = self.data["unemployment_rates"].loc[
            (self.data["unemployment_rates"]["LOCATION"] == country)
            & (self.data["unemployment_rates"]["SUBJECT"] == "TOT")
            & (self.data["unemployment_rates"]["FREQUENCY"] == "M")
        ]
        dates, vals = [], []
        for year in range(1970, 2024):
            for month in range(1, 13):
                s_month = str(month) if month > 9 else "0" + str(month)
                dates.append(str(year) + "-" + str(month))
                val = data.loc[data["TIME"] == str(year) + "-" + s_month, "Value"].values
                if len(val) == 0:
                    vals.append(np.nan)
                else:
                    vals.append(val[0] / 100.0)

        dates = pd.to_datetime(dates, format="%Y-%m")
        return pd.DataFrame(
            index=dates,
            data={"Unemployment Rate": vals},
        )

    def get_consumption_rates_by_income(self, country: Country) -> pd.DataFrame:
        df = self.data["consumption_by_income_quintiles"]
        df["country_year"] = [df["country_year"][i][0:3] for i in range(len(df))]
        df = df.loc[df["country_year"] == country]
        df = df.set_index("industry")
        df = df.loc[:, df.columns != "country_year"]
        df.index.name = "Industry"
        df.columns.name = "Income Quantile"

        return df

    def get_house_price_index(self, country: Country) -> pd.DataFrame:
        df = self.data["housing_index"]
        dates, val_real, val_nominal = [], [], []
        corr_months = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}
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
                for month in corr_months[quarter]:
                    dates.append(str(year) + "-" + str(month))
                    if len(curr_value_real) == 1 and len(prev_value_real) == 1:
                        val_real.append(curr_value_real[0] / prev_value_real[0] - 1.0)
                    else:
                        val_real.append(np.nan)
                    if len(curr_value_nominal) == 1 and len(prev_value_nominal) == 1:
                        val_nominal.append(curr_value_nominal[0] / prev_value_nominal[0] - 1.0)
                    else:
                        val_nominal.append(np.nan)

        dates = pd.to_datetime(dates, format="%Y-%m")

        return pd.DataFrame(
            index=dates,
            data={
                "Real House Price Index Growth": val_real,
                "Nominal House Price Index Growth": val_nominal,
            },
        )

    def get_vacancy_rate(self, country: Country) -> pd.DataFrame:
        active_population_size = self.data["active_population_size"]
        total_job_vacancies = self.data["total_job_vacancies"]
        dates, vacancy_rate = [], []
        corr_months = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}
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
                for month in corr_months[quarter]:
                    s_month = str(month) if month > 9 else "0" + str(month)
                    total_vacs = total_job_vacancies.loc[
                        (total_job_vacancies["LOCATION"] == country)
                        & (total_job_vacancies["TIME"] == str(year) + "-" + str(s_month)),
                        "Value",
                    ].values
                    dates.append(str(year) + "-" + str(month))
                    if len(pop_size) == 1 and len(total_vacs) == 1:
                        vacancy_rate.append(total_vacs[0] / pop_size[0])
                    else:
                        vacancy_rate.append(np.nan)
        dates = pd.to_datetime(dates, format="%Y-%m")
        return pd.DataFrame(index=dates, data={"Vacancy Rate": vacancy_rate})

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
