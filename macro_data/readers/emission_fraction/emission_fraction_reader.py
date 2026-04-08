"""
Module for reading and processing emissions-related data.

This module provides functionality to read and analyze emissions data from various
energy sources (coal, oil, gas) and calculate emissions factors. It handles both
direct emissions from fuel consumption and indirect emissions from refining processes.

Key Features:
    - Emissions factors for different fuel types

Classes:
    - EmissionsFractionReader: Reads and processes emission fraction data

Example:
    ```python
    from pathlib import Path
    from macro_data.readers.emissions.emissions_reader import EmissionsReader

    # Initialize reader with emission fraction data
    reader = EmissionsFractionReader.read_emission_fraction_data(Path("path/to/emission_fraction_data"))
    ```

Note:
    - All emissions factors are in tCO2 (metric tons of CO2)
    - All emissions factors are in tCH4 (metric tons of CH4)
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class EmissionsFractionReader:
    """
    Reader class for emissions-related fraction data.

    This class handles reading and processing fraction data for different emissions (CO2, CH4).

    Args:
        emitting_fraction_CO2 (pd.DataFrame): DataFrame containing CO2 emissions fractions
        emitting_fraction_CH4 (pd.DataFrame): DataFrame containing CH4 emissions fractions

    Attributes:
        emitting_fraction_CO2 (pd.DataFrame): DataFrame containing historical CO2 emissions fractions,
            indexed by date with columns for each fuel type
        emitting_fraction_CH4 (pd.DataFrame): DataFrame containing historical CH4 emissions fractions,
            indexed by date with columns for each fuel type
    """

    emitting_fraction_co2: pd.DataFrame
    emitting_fraction_ch4: pd.DataFrame
    emitting_fraction_consumption: pd.DataFrame
    emitting_fraction_investment: pd.DataFrame
    emission_industries_ch4: list[str]

    @classmethod
    def read_fraction_data(cls, data_path: Path | str):
        """
        Read emissions fraction data from CSV files.
        Args:
            data_path (Path | str): Path to directory containing emission fraction data files:
                ./raw_data/emission_factors/emitting_fraction_CO2.csv
                ./raw_data/emission_factors/emitting_fraction_CH4.csv

        Returns:
            EmissionsFractionReader: New instance with loaded fraction data
        """
        emission_industries_ch4 = [
                "A01",
                "B05a",
                "B05b",
                "B05c",
                "B07",
                "B09",
                "C17",
                "C19",
                "C20",
                "C21",
                "C22",
                "C23",
                "C24a",
                "C24b",
                "D01b",
                "D01c",
                "E",
                "F",
                "H49",
                "H50",
                "H51",
            ]
        
        if isinstance(data_path, str):
            data_path = Path(data_path)

        emitting_fraction_co2_df = pd.read_csv(data_path / "emitting_fraction_CO2.csv", index_col=0)
        emitting_fraction_ch4_df = pd.read_csv(data_path / "emitting_fraction_CH4.csv", index_col=0)
        emitting_fraction_consumption_df = pd.read_csv(
            data_path / "emitting_fraction_consumption.csv", index_col=0, header=0
        )
        emitting_fraction_investment_df = pd.read_csv(
            data_path / "emitting_fraction_investment.csv", index_col=0, header=0
        )

        return cls(
            emitting_fraction_co2=emitting_fraction_co2_df,
            emitting_fraction_ch4=emitting_fraction_ch4_df,
            emitting_fraction_consumption=emitting_fraction_consumption_df,
            emitting_fraction_investment=emitting_fraction_investment_df,
            emission_industries_ch4=emission_industries_ch4,
        )
