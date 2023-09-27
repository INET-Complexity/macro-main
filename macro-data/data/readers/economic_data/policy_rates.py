from pathlib import Path

import numpy as np
import pandas as pd


class PolicyRatesReader:
    def __init__(self, path: Path | str, country_code_path: Path | str):
        temp_df = pd.read_csv(path)

        self.c_map = pd.read_csv(country_code_path)
        # adds Euro Area to avoid breaking the country code mapping
        self.c_map.loc[len(self.c_map)] = [
            "Euro Area",
            "XM",
            "XM",
            "",
            "",
        ]

        years = np.unique([y[0:4] for y in temp_df.columns[13:]])
        countries = [{"country": c} for c in temp_df["Reference area"].values]

        for y in years:
            yearly_rates = temp_df[[date for date in temp_df.columns if y in date]].mean(axis=1)
            for i, country in enumerate(countries):
                country[y] = yearly_rates[i]

        self.df = pd.DataFrame(countries)
        self.df["code"] = self.country_code_switch(temp_df["REF_AREA"].values)

    def cb_policy_rate(self, country: str, year: int) -> float:
        if country in self.get_eu_country_code():
            country = "XM"
        annual_policy_rate = self.df.loc[self.df["code"] == country, str(year)].values[0] / 100.0
        return (1 + annual_policy_rate) ** (1.0 / 12) - 1.0

    def country_code_switch(self, codes):
        return [self.c_map.loc[self.c_map["Alpha-2 code"] == c, "Alpha-3 code"].values[0] for c in codes]

    @staticmethod
    def get_eu_country_code():
        return [
            "AUT",
            "BEL",
            "BGR",
            "HRV",
            "CYP",
            "CZE",
            "DNK",
            "EST",
            "FIN",
            "FRA",
            "DEU",
            "GRC",
            "HUN",
            "IRL",
            "ITA",
            "LVA",
            "LTU",
            "LUX",
            "MLT",
            "NLD",
            "POL",
            "PRT",
            "ROU",
            "SVK",
            "SVN",
            "ESP",
            "SWE",
            "GBR",
        ]
