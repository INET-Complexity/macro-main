"""
Module for reading and processing emissions-related data.

This module provides functionality to read and analyze emissions data from various
energy sources (coal, oil, gas) and calculate emissions factors. It handles both
direct emissions from fuel consumption and indirect emissions from refining processes.

Key Features:
    - Emissions factors for different fuel types
    - Energy conversion rates
    - Price data handling for fuels
    - Local currency unit (LCU) conversions
    - Refining process emissions calculations

Classes:
    - EmissionsReader: Reads and processes fuel price data
    - EmissionsData: Stores emissions factors in local currency units
    - EmissionsEnergyFactors: Manages energy-to-emissions conversion factors

Example:
    ```python
    from pathlib import Path
    from macro_data.readers.emissions.emissions_reader import EmissionsReader

    # Initialize reader with price data
    reader = EmissionsReader.read_price_data(Path("path/to/price_data"))

    # Get emissions factors for a specific year
    factors = reader.get_emissions_factors(year=2020)
    ```

Note:
    - All emissions factors are in tCO2 (metric tons of CO2)
    - Energy units vary by fuel type (tons, barrels, MBTU)
    - Prices are expected in USD
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from macro_data.configuration.countries import Country
from macro_data.readers.io_tables.icio_reader import ICIOReader

# Emissions factors (tCO2 per unit)
COAL_TCO2_PER_TON = 1.57  # Metric tons of CO2 per ton of coal
OIL_TCO2_PER_BARREL = 0.43  # Metric tons of CO2 per barrel of oil
GAS_TCO2_PER_MBTU = 0.053  # Metric tons of CO2 per million BTU of natural gas

# Energy content factors (kWh per unit)
COAL_KWH_PER_TON = 8100  # Kilowatt-hours per ton of coal
OIL_KWH_PER_BARREL = 1700  # Kilowatt-hours per barrel of oil
GAS_KWH_PER_MBTU = 293  # Kilowatt-hours per million BTU of natural gas

# Energy content per tCO2 (kWh/tCO2)
COAL_KWH_PER_TCO2 = COAL_KWH_PER_TON / COAL_TCO2_PER_TON
OIL_KWH_PER_TCO2 = OIL_KWH_PER_BARREL / OIL_TCO2_PER_BARREL
GAS_KWH_PER_TCO2 = GAS_KWH_PER_MBTU / GAS_TCO2_PER_MBTU


@dataclass
class EmissionsReader:
    """
    Reader class for emissions-related price data.

    This class handles reading and processing price data for different fuel types
    (coal, oil, gas) and calculates emissions factors based on these prices.

    Args:
        prices_df (pd.DataFrame): DataFrame containing fuel prices with columns
            'coal_price', 'oil_price', and 'gas_price'

    Attributes:
        prices_df (pd.DataFrame): DataFrame containing historical fuel prices,
            indexed by date with columns for each fuel type
    """

    prices_df: pd.DataFrame

    @classmethod
    def read_price_data(cls, data_path: Path | str):
        """
        Read fuel price data from CSV files.

        Args:
            data_path (Path | str): Path to directory containing price data files (downloaded from the
            Federal Reserve Bank of St. Louis FRED database):
                - PCOALAUUSDM.csv: Coal prices
                - POILBREUSDM.csv: Oil prices
                - PNGASEUUSDM.csv: Gas prices

        Returns:
            EmissionsReader: New instance with loaded price data

        Note:
            - Expects specific file names for each fuel type
            - Resamples data to yearly frequency (first value of each year)
            - All prices should be in USD
        """
        if isinstance(data_path, str):
            data_path = Path(data_path)

        coal = pd.read_csv(data_path / "PCOALAUUSDM.csv")
        coal["observation_date"] = pd.to_datetime(coal["observation_date"])
        coal.rename(columns={"PCOALAUUSDM": "coal_price", "observation_date": "DATE"}, inplace=True)
        coal.set_index("DATE", inplace=True)
        coal = coal.resample("YS").first()

        oil = pd.read_csv(data_path / "POILBREUSDM.csv")
        oil["observation_date"] = pd.to_datetime(oil["observation_date"])
        oil.rename(columns={"POILBREUSDM": "oil_price", "observation_date": "DATE"}, inplace=True)
        oil.set_index("DATE", inplace=True)
        oil = oil.resample("YS").first()

        gas = pd.read_csv(data_path / "PNGASEUUSDM.csv")
        gas["observation_date"] = pd.to_datetime(gas["observation_date"])
        gas.rename(columns={"PNGASEUUSDM": "gas_price", "observation_date": "DATE"}, inplace=True)
        gas.set_index("DATE", inplace=True)
        gas = gas.resample("YS").first()

        prices_df = pd.merge(coal, oil, left_index=True, right_index=True)
        prices_df = pd.merge(prices_df, gas, left_index=True, right_index=True)

        return cls(prices_df=prices_df)

    def get_emissions_factors(self, year: int) -> dict[str, float]:
        """
        Calculate emissions factors for each fuel type based on prices.

        Args:
            year (int): Year to get emissions factors for

        Returns:
            dict[str, float]: Dictionary mapping fuel types ('coal', 'oil', 'gas')
                             to their emissions factors in tCO2 per USD

        Note:
            Emissions factors are calculated by dividing the standard emissions
            factor for each fuel by its price, giving tCO2 per USD spent
        """
        coal_tco2_per_usd = COAL_TCO2_PER_TON / self.prices_df.loc[f"{year}", "coal_price"].iloc[0]
        oil_tco2_per_usd = OIL_TCO2_PER_BARREL / self.prices_df.loc[f"{year}", "oil_price"].iloc[0]
        gas_tco2_per_usd = GAS_TCO2_PER_MBTU / self.prices_df.loc[f"{year}", "gas_price"].iloc[0]

        return {
            "coal": coal_tco2_per_usd,
            "oil": oil_tco2_per_usd,
            "gas": gas_tco2_per_usd,
        }


@dataclass
class EmissionsData:
    """
    Container for emissions factors in local currency units (LCU).

    This class stores emissions factors for different fuel types and refining
    processes, all converted to local currency units for a specific country.

    Args:
        coal_factor_lcu (float): Coal emissions factor in tCO2 per LCU
        gas_factor_lcu (float): Natural gas emissions factor in tCO2 per LCU
        oil_factor_lcu (float): Oil emissions factor in tCO2 per LCU
        refining_factor_lcu (float): Refining process emissions factor in tCO2 per LCU

    Attributes:
        coal_factor_lcu (float): Coal emissions factor
        gas_factor_lcu (float): Natural gas emissions factor
        oil_factor_lcu (float): Oil emissions factor
        refining_factor_lcu (float): Refining process emissions factor
    """

    coal_factor_lcu: float
    gas_factor_lcu: float
    oil_factor_lcu: float
    refining_factor_lcu: float

    @classmethod
    def from_readers(
        cls,
        usd_emission_factors: dict[str, float],
        exchange_rate: float,
    ):
        """
        Create EmissionsData instance from USD factors and exchange rate.

        Args:
            usd_emission_factors (dict[str, float]): Dictionary mapping fuel types
                to their emissions factors in tCO2 per USD
            exchange_rate (float): Exchange rate from USD to local currency unit

        Returns:
            EmissionsData: New instance with emissions factors in local currency

        Note:
            Exchange rate should be in LCU per USD format
            (e.g., 1.3 means 1 USD = 1.3 LCU)
        """
        oil_factor_lcu = usd_emission_factors["oil"] / exchange_rate
        gas_factor_lcu = usd_emission_factors["gas"] / exchange_rate
        coal_factor_lcu = usd_emission_factors["coal"] / exchange_rate
        refining_factor_lcu = usd_emission_factors["coke_refining"] / exchange_rate

        return cls(
            oil_factor_lcu=oil_factor_lcu,
            gas_factor_lcu=gas_factor_lcu,
            coal_factor_lcu=coal_factor_lcu,
            refining_factor_lcu=refining_factor_lcu,
        )

    @property
    def emissions_array(self) -> np.ndarray:
        """
        Get emissions factors as a numpy array.

        Returns:
            np.ndarray: Array of emissions factors in order:
                       [coal, gas, oil, refining]

        Note:
            Order of factors in array is fixed and matches the order
            expected by other parts of the system
        """
        return np.array([self.coal_factor_lcu, self.gas_factor_lcu, self.oil_factor_lcu, self.refining_factor_lcu])


@dataclass
class EmissionsEnergyFactors:
    """
    Container for energy-to-emissions conversion factors.

    This class stores conversion factors between energy (kWh) and emissions (tCO2)
    for different fuel types and refining processes.

    Args:
        refining_kwh_per_tco2 (float): Energy output per tCO2 for refining process
        coal_kwh_per_tco2 (float): Energy output per tCO2 for coal.
                                  Defaults to COAL_KWH_PER_TCO2.
        oil_kwh_per_tco2 (float): Energy output per tCO2 for oil.
                                 Defaults to OIL_KWH_PER_TCO2.
        gas_kwh_per_tco2 (float): Energy output per tCO2 for natural gas.
                                 Defaults to GAS_KWH_PER_TCO2.

    Attributes:
        refining_kwh_per_tco2 (float): Refining process energy factor
        coal_kwh_per_tco2 (float): Coal energy factor
        oil_kwh_per_tco2 (float): Oil energy factor
        gas_kwh_per_tco2 (float): Natural gas energy factor
    """

    refining_kwh_per_tco2: float
    coal_kwh_per_tco2: float = COAL_KWH_PER_TCO2
    oil_kwh_per_tco2: float = OIL_KWH_PER_TCO2
    gas_kwh_per_tco2: float = GAS_KWH_PER_TCO2

    @classmethod
    def from_readers(cls, icio_reader: ICIOReader, countries: list[Country | str]):
        """
        Create EmissionsEnergyFactors instance from ICIO reader.

        Args:
            icio_reader (ICIOReader): Reader for input-output tables
            countries (list[Country | str]): List of countries to average over

        Returns:
            EmissionsEnergyFactors: New instance with calculated energy factors

        Note:
            Uses average refining coefficient across specified countries plus ROW
        """
        refining_coeff = get_avg_coke_refining_kwh_per_tco2(icio_reader, countries)
        return cls(refining_kwh_per_tco2=refining_coeff)


def get_country_coke_refining_kwh_per_tco2(icio_reader: ICIOReader, country: str | Country) -> float:
    """
    Calculate energy output per tCO2 for coke refining in a country.

    Args:
        icio_reader (ICIOReader): Reader for input-output tables
        country (str | Country): Country to calculate coefficient for

    Returns:
        float: Energy output (kWh) per tCO2 for coke refining process

    Note:
        Uses input-output coefficients to weight the energy content
        of different fuel inputs to the refining process
    """
    coefficients = (1 / icio_reader.get_intermediate_inputs_matrix(country)).loc[["B05a", "B05b", "B05c"], "C19"]
    return coefficients @ np.array([COAL_KWH_PER_TCO2, OIL_KWH_PER_TCO2, GAS_KWH_PER_TCO2])


def get_avg_coke_refining_kwh_per_tco2(icio_reader: ICIOReader, countries: list[str | Country]) -> float:
    """
    Calculate average energy output per tCO2 for coke refining across countries.

    Args:
        icio_reader (ICIOReader): Reader for input-output tables
        countries (list[str | Country]): List of countries to average over

    Returns:
        float: Average energy output (kWh) per tCO2 for coke refining

    Note:
        Includes ROW (Rest of World) in the average calculation
    """
    return np.mean([get_country_coke_refining_kwh_per_tco2(icio_reader, country) for country in countries + ["ROW"]])
