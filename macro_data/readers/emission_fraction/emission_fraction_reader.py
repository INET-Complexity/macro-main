"""
Module for reading and processing emission fraction data.

Classes:
    - EmissionFractions: Container dataclass for emission fraction arrays
    - EmissionsFractionReader: Reads emission fraction CSVs from disk

Note:
    - CO2 fractions are dimensionless multipliers per industry
    - CH4 fractions are dimensionless multipliers per industry
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class EmissionFractions:
    """Container for emission fraction arrays extracted from reader DataFrames.

    Attributes:
        co2: CO2 emission fractions by industry (shape: n_emitting x n_industries)
        ch4: CH4 emission fractions by industry (shape: 1 x n_industries)
        consumption: Household consumption emission fractions (shape: 1 x n_industries)
        investment: Household investment emission fractions (shape: 1 x n_industries)
    """

    co2: Optional[np.ndarray] = None
    ch4: Optional[np.ndarray] = None
    consumption: Optional[np.ndarray] = None
    investment: Optional[np.ndarray] = None

    @classmethod
    def from_reader(cls, reader: EmissionsFractionReader) -> EmissionFractions:
        """Build an EmissionFractions container from a reader instance."""
        return cls(
            co2=reader.emitting_fraction_co2.values,
            ch4=reader.emitting_fraction_ch4.values,
            consumption=reader.emitting_fraction_consumption.values,
            investment=reader.emitting_fraction_investment.values,
        )


@dataclass
class EmissionsFractionReader:
    """Reads emission fraction CSVs for CO2, CH4, consumption, and investment.

    Attributes:
        emitting_fraction_co2: Per-industry CO2 multipliers (rows = emitting industries)
        emitting_fraction_ch4: Per-industry CH4 multipliers (rows = gas types)
        emitting_fraction_consumption: Household consumption fractions
        emitting_fraction_investment: Household investment fractions
    """

    emitting_fraction_co2: pd.DataFrame
    emitting_fraction_ch4: pd.DataFrame
    emitting_fraction_consumption: pd.DataFrame
    emitting_fraction_investment: pd.DataFrame

    @classmethod
    def read_fraction_data(cls, data_path: Path | str) -> EmissionsFractionReader:
        """Read emission fraction CSVs from data_path directory.

        Expects files:
            emitting_fraction_co2.csv
            emitting_fraction_ch4.csv
            emitting_fraction_consumption.csv
            emitting_fraction_investment.csv
        """
        if isinstance(data_path, str):
            data_path = Path(data_path)

        return cls(
            emitting_fraction_co2=pd.read_csv(data_path / "emitting_fraction_co2.csv", index_col=0),
            emitting_fraction_ch4=pd.read_csv(data_path / "emitting_fraction_ch4.csv", index_col=0),
            emitting_fraction_consumption=pd.read_csv(
                data_path / "emitting_fraction_consumption.csv", index_col=0, header=0
            ),
            emitting_fraction_investment=pd.read_csv(
                data_path / "emitting_fraction_investment.csv", index_col=0, header=0
            ),
        )
