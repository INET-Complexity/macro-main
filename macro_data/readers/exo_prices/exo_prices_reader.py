"""Reader for CIMS exogenous energy price data."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class ExoPricesReader:
    """Reader for exogenous fossil fuel and electricity prices.

    Reads CSV files containing model energy price projections for:
    - Fossil fuels (coal, natural gas, petroleum, refined products)
    - Electricity

    The CSVs contain price projections by year with specific row indices
    for different energy products.

    Attributes:
        fossil_prices: DataFrame with fossil fuel price projections
        electricity_prices: DataFrame with electricity price projections
    """

    fossil_prices: Optional[pd.DataFrame] = None
    electricity_prices: Optional[pd.DataFrame] = None
    electricity_prices_2: Optional[pd.DataFrame] = None
    B05c_raw = [61, 53, 81, 72, 47, 101, 86, 80, 75, 75, 75, 75, 75]
    t_raw = [2014, 2016, 2018, 2019, 2020, 2022, 2023, 2025, 2030, 2035, 2040, 2045, 2050]

    @classmethod
    def from_raw_data(
        cls,
        fossil_prices_path: str,
        electricity_prices_path: str,
        electricity_prices_path_2: Optional[str] = None,
    ) -> "ExoPricesReader":
        """Load exogenous price data from CSV files.

        Args:
            fossil_prices_path: Path to CSV with fossil fuel prices
            electricity_prices_path: Path to CSV with electricity prices

        Returns:
            ExoPricesReader with loaded data
        """
        fossil_prices = None
        electricity_prices = None
        electricity_prices_2 = None

        if fossil_prices_path:
            fossil_prices = pd.read_csv(fossil_prices_path)

        if electricity_prices_path:
            electricity_prices = pd.read_csv(electricity_prices_path)

        if electricity_prices_path_2:
            electricity_prices_2 = pd.read_csv(electricity_prices_path_2)

        return cls(
            fossil_prices=fossil_prices,
            electricity_prices=electricity_prices,
            electricity_prices_2=electricity_prices_2,
        )
