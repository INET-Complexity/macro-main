from datetime import date
from pathlib import Path

import pandas as pd

from inet_data.configuration.countries import Country
from inet_data.readers.util.prune_util import prune_index


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
            "bank_demography": pd.read_csv(data_path / "imf_fas_bank_demographics.csv"),
        }
        return cls(data, scale_dict)

    @staticmethod
    def get_files_with_codes() -> dict[str, str]:
        return {
            "bank_demography": "imf_fas_bank_demographics",
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

    def prune(self, prune_date: date):
        mask = prune_index(self.data["bank_demography"].columns, prune_date)
        self.data["bank_demography"] = self.data["bank_demography"].loc[:, mask]
