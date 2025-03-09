"""
This module provides functionality for reading and processing World Bank employment statistics.
It handles both unemployment rates and labor force participation rates across countries
and years.

Key Features:
- Read employment data from HDF5 files
- Access unemployment rates by country and year
- Access labor force participation rates by country and year
- Automatic percentage to decimal conversion

Example:
    ```python
    from pathlib import Path
    from macro_data.readers.population_data.employment_data import WorldBankEmployment

    # Initialize reader from HDF5 file
    employment = WorldBankEmployment.from_hdf_path(
        Path("path/to/employment_data.h5")
    )

    # Get unemployment rate for a specific year
    unemployment_2020 = employment.get_unemployment_rates(2020)
    usa_unemployment = unemployment_2020["USA"]

    # Get participation rate for a specific year
    participation_2020 = employment.get_participation_rates(2020)
    usa_participation = participation_2020["USA"]
    ```

Note:
    All rates are returned as decimals (e.g., 0.05 for 5%) for direct use in calculations.
"""

from pathlib import Path

import pandas as pd


class WorldBankEmployment:
    """
    A class for reading and processing World Bank employment statistics.

    This class provides access to two key employment metrics:
    1. Unemployment rates: Percentage of labor force that is unemployed
    2. Labor force participation rates: Percentage of working-age population in labor force

    Parameters
    ----------
    unemployment_rates : pd.DataFrame
        DataFrame containing unemployment rates by country and year
    part_rates : pd.DataFrame
        DataFrame containing participation rates by country and year

    Attributes
    ----------
    unemployment_rates : pd.DataFrame
        Unemployment rates with countries as index and years as columns
    part_rates : pd.DataFrame
        Participation rates with countries as index and years as columns

    Notes
    -----
    - All rates are stored as percentages but returned as decimals
    - Years are stored as string columns in the DataFrames
    """

    def __init__(self, unemployment_rates: pd.DataFrame, part_rates: pd.DataFrame):
        self.unemployment_rates = unemployment_rates
        self.part_rates = part_rates

    @classmethod
    def from_hdf_path(cls, hdf_path: Path | str):
        """
        Create a WorldBankEmployment instance from an HDF5 file.

        Parameters
        ----------
        hdf_path : Path | str
            Path to the HDF5 file containing employment data

        Returns
        -------
        WorldBankEmployment
            Initialized instance with data loaded from file

        Notes
        -----
        - The HDF5 file should contain two keys:
          * "unemployment": DataFrame of unemployment rates
          * "participation": DataFrame of participation rates
        """
        # noinspection PyTypeChecker
        unemployment_rates: pd.DataFrame = pd.read_hdf(hdf_path, key="unemployment")
        # noinspection PyTypeChecker
        part_rates: pd.DataFrame = pd.read_hdf(hdf_path, key="participation")
        return cls(unemployment_rates, part_rates)

    def get_unemployment_rates(self, year: int):
        """
        Get unemployment rates for all countries in a specific year.

        Parameters
        ----------
        year : int
            Year to retrieve unemployment rates for

        Returns
        -------
        dict
            Dictionary mapping country codes to unemployment rates (as decimals)

        Raises
        ------
        KeyError
            If the specified year is not in the data

        Notes
        -----
        - Returns rates as decimals (e.g., 0.05 for 5% unemployment)
        """
        year = str(year)
        if year not in self.unemployment_rates.columns:
            raise KeyError(f"Year {year} not in data")
        col = self.unemployment_rates[str(year)] / 100
        return col.to_dict()

    def get_participation_rates(self, year: int):
        """
        Get labor force participation rates for all countries in a specific year.

        Parameters
        ----------
        year : int
            Year to retrieve participation rates for

        Returns
        -------
        dict
            Dictionary mapping country codes to participation rates (as decimals)

        Raises
        ------
        KeyError
            If the specified year is not in the data

        Notes
        -----
        - Returns rates as decimals (e.g., 0.65 for 65% participation)
        """
        year = str(year)
        if year not in self.part_rates.columns:
            raise KeyError(f"Year {year} not in data")
        col = self.part_rates[str(year)] / 100
        return col.to_dict()
