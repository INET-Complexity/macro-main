from pathlib import Path

import pandas as pd


class WorldBankEmployment:
    def __init__(self, unemployment_rates: pd.DataFrame, part_rates: pd.DataFrame):
        self.unemployment_rates = unemployment_rates
        self.part_rates = part_rates

    @classmethod
    def from_hdf_path(cls, hdf_path: Path | str):
        # noinspection PyTypeChecker
        unemployment_rates: pd.DataFrame = pd.read_hdf(hdf_path, key="unemployment")
        # noinspection PyTypeChecker
        part_rates: pd.DataFrame = pd.read_hdf(hdf_path, key="participation")
        return cls(unemployment_rates, part_rates)

    def get_unemployment_rates(self, year: int):
        year = str(year)
        if year not in self.unemployment_rates.columns:
            raise KeyError(f"Year {year} not in data")
        col = self.unemployment_rates[str(year)] / 100
        return col.to_dict()

    def get_participation_rates(self, year: int):
        year = str(year)
        if year not in self.part_rates.columns:
            raise KeyError(f"Year {year} not in data")
        col = self.part_rates[str(year)] / 100
        return col.to_dict()
