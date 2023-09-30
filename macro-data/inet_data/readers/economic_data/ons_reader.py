import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import zetac


class ONSReader:
    def __init__(self, path: Path | str):
        # Load inet_data files
        self.files_with_codes = self.get_files_with_codes()
        self.data = {
            key: pd.read_csv(
                path / (self.files_with_codes[key] + ".csv"),
                sep="\t",
                header=0,
                index_col=0,
            )
            for key in self.files_with_codes.keys()
        }

    @staticmethod
    def get_files_with_codes() -> dict[str, str]:
        return {
            "uk_firm_sizes": "UKCompSizes",
            "uk_sector_map": "UKSec_map",
        }

    @staticmethod
    def zeta_dist(x, a):
        z = 1 / (x**a * zetac(a))
        return z / sum(z)

    def get_firm_size_zetas(self) -> dict[int, float]:
        # get shape parameters for zeta distribution
        firms_df = self.data["uk_firm_sizes"]
        group_means = [np.mean([int(val) for val in c.split("-")]) for c in firms_df.columns[:-1]]
        shapes = {}
        for i, row in firms_df.iterrows():
            freq = row[:-1].map(lambda x: int(x.replace(",", "")))
            freq = freq / np.sum(freq)
            shapes[i] = curve_fit(self.zeta_dist, group_means, freq, p0=[1.16])[0][0]

        # map shape parameters to ISIC sector
        map_df = self.data["uk_sector_map"]
        map_isic = {l: [] for l in list(string.ascii_uppercase)[:21]}
        for i, row in map_df.iterrows():
            for sec in re.findall(r"[A-Z]", row["SIC07 section letter"]):
                if row.name in shapes.keys():
                    map_isic[sec].append(shapes[row.name])

        return {i: np.mean(map_isic[sec]) for i, sec in enumerate(map_isic.keys())}
