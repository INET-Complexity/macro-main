"""
Module for reading and processing UK Office for National Statistics (ONS) data.

This module provides functionality to read and analyze firm size distribution data
from the UK Office for National Statistics. It specializes in fitting Zeta
distributions to firm size data and mapping between different industry classification
systems.

Key Features:
    - Reads firm size distribution data from ONS
    - Fits Zeta distributions to firm size data
    - Maps between SIC07 and ISIC industry classifications
    - Calculates shape parameters for firm size distributions by sector

Example:
    ```python
    from pathlib import Path

    # Initialize reader
    reader = ONSReader(path=Path("path/to/ons_data"))

    # Get Zeta distribution parameters by sector
    sector_zetas = reader.get_firm_size_zetas()
    ```

Note:
    - Uses tab-separated CSV files for data input
    - Assumes specific file naming conventions for data files
    - Handles UK-specific industry classifications
"""

import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import zetac


class ONSReader:
    """
    Reader class for UK Office for National Statistics (ONS) data.

    This class handles reading and processing of ONS data files, particularly
    focusing on firm size distributions and sector mappings.

    Args:
        path (Path | str): Path to directory containing ONS data files

    Attributes:
        files_with_codes (dict[str, str]): Mapping of data categories to file names
        data (dict[str, pd.DataFrame]): Dictionary of loaded ONS datasets
    """

    def __init__(self, path: Path | str):
        """Initialize the ONSReader with data path."""
        # Load data files
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
        """
        Get mapping of data categories to file names.

        Returns:
            dict[str, str]: Dictionary mapping data categories to their file names,
                           including firm sizes and sector mappings
        """
        return {
            "uk_firm_sizes": "UKCompSizes",
            "uk_sector_map": "UKSec_map",
        }

    @staticmethod
    def zeta_dist(x: np.ndarray, a: float) -> np.ndarray:
        """
        Calculate normalized Zeta distribution values.

        Args:
            x (np.ndarray): Input values (firm sizes)
            a (float): Shape parameter of the Zeta distribution

        Returns:
            np.ndarray: Normalized probability values from Zeta distribution

        Note:
            Uses Riemann zeta function minus 1 (zetac) for calculations
        """
        z = 1 / (x**a * zetac(a))
        return z / sum(z)

    def get_firm_size_zetas(self) -> dict[int, float]:
        """
        Calculate Zeta distribution shape parameters for each industry sector.

        Returns:
            dict[int, float]: Dictionary mapping sector indices (0-20) to their
                             corresponding Zeta distribution shape parameters

        Note:
            - Fits Zeta distributions to empirical firm size distributions
            - Maps SIC07 sectors to ISIC sectors
            - Returns average shape parameter when multiple SIC07 sectors map to
              same ISIC sector
        """
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
