import warnings
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.readers.util.prune_util import prune_index


class IMFReader:
    # def __init__(self, path: Path | str, scale: int):
    #     # Parameters
    #     self.scale = scale
    #
    #     # Load data files
    #     self.files_with_codes = self.get_files_with_codes()
    #     self.data = {
    #         key: pd.read_csv(path / (self.files_with_codes[key] + ".csv")) for key in self.files_with_codes.keys()
    #     }

    def __init__(self, data: dict[str, pd.DataFrame], scale_dict: dict[Country, int]):
        # Parameters
        self.scale_dict = scale_dict

        # Load data files
        self.data = {key: data[key] for key in data.keys()}

    @classmethod
    def from_data(cls, data_path: Path | str, scale_dict: dict[Country, int]) -> "IMFReader":
        data = {
            "bank_demography": pd.read_csv(
                data_path / "imf_fas_bank_demographics.csv", encoding="latin-1", engine="pyarrow"
            ),
            "international_financial_statistics": pd.read_csv(
                data_path / "IFS.csv", encoding="latin-1", engine="pyarrow"
            ),
        }
        return cls(data, scale_dict)

    @staticmethod
    def get_files_with_codes() -> dict[str, str]:
        return {
            "bank_demography": "imf_fas_bank_demographics",
            "international_financial_statistics": "IFS",
        }

    def get_value(self, year: int, country: str, stat: str) -> float:
        df = self.data["bank_demography"]
        mask = (df["STAT"] == stat) & (df["COU"] == country)
        value = df.loc[mask][str(year)].iloc[0]
        return float(value.replace(",", ""))

    def number_of_commercial_banks(self, year: int, country: str | Country) -> float:
        return self.get_value(year, country, "Institutions of commercial banks") / self.scale_dict[country]

    def number_of_commercial_depositors(self, year: int, country: str | Country) -> float:
        return self.get_value(year, country, "Depositors with commercial banks") / self.scale_dict[country]

    def number_of_commercial_borrowers(self, year: int, country: str | Country) -> float:
        return self.get_value(year, country, "Borrowers from commercial banks") / self.scale_dict[country]

    # domestic currency
    def total_commercial_deposits(self, year: int, country: str | Country) -> float:
        return self.get_value(year, country, "Outstanding deposits with commercial banks") * 1e6

    # domestic currency
    def total_commercial_loans(self, year: int, country: str | Country) -> float:
        return self.get_value(year, country, "Outstanding loans from commercial banks") * 1e6

    def get_inflation(self, country: Country | str) -> Optional[pd.DataFrame]:
        if isinstance(country, str):
            country = Country(country)
        if country.value == "ARG":  # using CB data instead
            return None

        country_english = str(country).lower()
        data = self.data["international_financial_statistics"]
        data.rename(columns=data.iloc[0]).drop(data.index[0]).reset_index(drop=True)
        data = data.loc[(data["Attribute"] == "Value") & (data["Country Name"].str.lower() == country_english)]
        data.set_index("Indicator Name", inplace=True)
        if (
            "Prices, Consumer Price Index, All items, Index" not in data.index
            and "Prices, Producer Price Index, All Commodities, Index" not in data.index
        ):
            return None
        elif "Prices, Consumer Price Index, All items, Index" not in data.index:
            data = data.loc[
                [
                    "Prices, Producer Price Index, All Commodities, Index",
                    "Prices, Producer Price Index, All Commodities, Index",
                ]
            ].T.iloc[4:-1]
        elif "Prices, Producer Price Index, All Commodities, Index" not in data.index:
            data = data.loc[
                [
                    "Prices, Consumer Price Index, All items, Index",
                    "Prices, Consumer Price Index, All items, Index",
                ]
            ].T.iloc[4:-1]
        else:
            data = data.loc[
                [
                    "Prices, Consumer Price Index, All items, Index",
                    "Prices, Producer Price Index, All Commodities, Index",
                ]
            ].T.iloc[4:-1]
        data.columns = ["CPI Inflation", "PPI Inflation"]
        data.index = [pd.Timestamp(int(ind[0:4]), 3 * int(ind[5]) - 2, 1) for ind in data.index]  # noqa
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            return data.astype(float).pct_change()

    def get_na_growth_rates(self, country: str | Country) -> pd.DataFrame:
        # country_english = self.c_map.loc[
        #     self.c_map["Alpha-3 code"] == country,
        #     "English short name lower case",
        # ].values[0]
        if isinstance(country, str):
            country = Country(country)
        country_english = str(country).lower()
        data = self.data["international_financial_statistics"]
        data.rename(columns=data.iloc[0]).drop(data.index[0]).reset_index(drop=True)
        data = data.loc[(data["Attribute"] == "Value") & (data["Country Name"].str.lower() == country_english)]
        data.set_index("Indicator Name", inplace=True)
        data = data.T.iloc[4:-1]

        # Get the data
        fields = {
            "Gross Domestic Product": "GDP",
            "Households Final Consumption Expenditure": "HH Cons",
            "Non-profit Institutions Serving Households (NPISHs) Final Consumption Expenditure": "NPISH Cons",
            "General Government Final Consumption Expenditure": "Gov Cons",
            "Gross Fixed Capital Formation": "Gross Fixed Capital Formation",
            "Changes in Inventories": "Changes in Inventories",
            # "Acquisitions less Disposals of Valuables": "Acquisitions less Disposals of Valuables",
            "Exports of Goods": "Exports of Goods",
            "Exports of Services": "Exports of Services",
            "Exports of Goods and Services": "Exports of Goods and Services",
            "Imports of Goods": "Imports of Goods",
            "Imports of Services": "Imports of Services",
            "Imports of Goods and Services": "Imports of Goods and Services",
        }
        gdp_field = "Gross Domestic Product"
        data_ls = {}

        for field in fields.keys():
            if field + ", Nominal, Seasonally Adjusted, Domestic Currency" in data.columns:
                data_ls[fields[field]] = data[field + ", Nominal, Seasonally Adjusted, Domestic Currency"].values
            elif field + ", Nominal, Domestic Currency" in data.columns:
                data_ls[fields[field]] = data[field + ", Nominal, Domestic Currency"].values
            elif field + ", Nominal, Unadjusted, Domestic Currency" in data.columns:
                data_ls[fields[field]] = data[field + ", Nominal, Unadjusted, Domestic Currency"].values

            # Otherwise take GDP
            elif gdp_field + ", Nominal, Seasonally Adjusted, Domestic Currency" in data.columns:
                data_ls[fields[field]] = data[gdp_field + ", Nominal, Seasonally Adjusted, Domestic Currency"].values
            elif gdp_field + ", Nominal, Domestic Currency" in data.columns:
                data_ls[fields[field]] = data[gdp_field + ", Nominal, Domestic Currency"].values
            elif gdp_field + ", Nominal, Unadjusted, Domestic Currency" in data.columns:
                data_ls[fields[field]] = data[gdp_field + ", Nominal, Unadjusted, Domestic Currency"].values
            else:
                raise ValueError("uh oh", country, field)
                # Calculate growth rates
        data = pd.DataFrame(
            data=data_ls,
            index=[pd.Timestamp(int(ind[0:4]), 3 * int(ind[5]) - 2, 1) for ind in data.index],
        ).iloc[0:-1]
        data = data.astype(float)
        data["HH Cons"] = data["HH Cons"] + data["NPISH Cons"]
        data = (data / data.shift(1) - 1.0).iloc[1:]
        data["Gross Output"] = data["GDP"].values
        data["Intermediate Consumption"] = data["GDP"].values
        data = data.loc[:, data.columns != "NPISH Cons"]

        return data

    def get_labour_stats(self, country: str | Country) -> Optional[pd.DataFrame]:
        if isinstance(country, str):
            country = Country(country)
        country_english = str(country).lower()
        data = self.data["international_financial_statistics"]
        data.rename(columns=data.iloc[0]).drop(data.index[0]).reset_index(drop=True)
        data = data.loc[(data["Attribute"] == "Value") & (data["Country Name"].str.lower() == country_english)]
        data.set_index("Indicator Name", inplace=True)
        if (
            "Labor Force, Persons, Number of" not in data.index
            or "Employment, Persons, Number of" not in data.index
            or "Unemployment, Persons, Number of" not in data.columns
            or "Labor Markets, Unemployment Rate, Percent" not in data.columns
        ):
            return None
        data = data.loc[
            [
                "Labor Force, Persons, Number of",
                "Employment, Persons, Number of",
                "Unemployment, Persons, Number of",
                "Labor Markets, Unemployment Rate, Percent",
            ]
        ].T.iloc[4:-1]
        data.columns = [
            "Labour Force",
            "Employment Number",
            "Unemployment Number",
            "Unemployment Rate",
        ]
        data.index = [pd.Timestamp(int(ind[0:4]), 3 * int(ind[5]) - 2, 1) for ind in data.index]  # noqa
        data = data.astype(float)
        data["Unemployment Rate"] /= 100.0
        return data

    def prune(self, prune_date: date):
        mask = prune_index(self.data["bank_demography"].columns, prune_date)
        self.data["bank_demography"] = self.data["bank_demography"].loc[:, mask]
