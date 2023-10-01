from pathlib import Path

import pandas as pd


class WorldBankRatesReader:
    def __init__(self, df):
        self.df = df

    @classmethod
    def from_csv(cls, path: Path | str) -> "WorldBankRatesReader":
        df = pd.read_csv(path, index_col=0)
        df.columns.name = "Year"
        return cls(df)

    def exchange_rates_dict(self, year: int) -> dict[str, float]:
        return self.df[str(year)].to_dict()

    def to_usd(self, country: str, year: int) -> float:
        return 1 / self.df.loc[country, str(year)]

    def from_usd(self, country: str, year: int) -> float:
        return self.df.loc[country, str(year)]

    def from_eur_to_lcu(self, country: str, year: int) -> float:
        return self.to_usd("DEU", year) * self.from_usd(country, year)

    def from_usd_to_lcu(self, country: str, year: int) -> float:
        return self.from_usd(country, year)
