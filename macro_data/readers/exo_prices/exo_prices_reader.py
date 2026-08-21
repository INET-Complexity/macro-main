"""Reader and container for exogenous sector price data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SectorExoPrices:
    """Container for exogenous sector price trajectories.

    Holds a DataFrame of price trajectories keyed by industry (sector) name.
    Each column is a sector; the index is years. Prices are sector-level:
    all firms within a listed sector follow the same normalised trajectory,
    computed at runtime by SectorExogenousPriceSetter.

    Attributes:
        prices: DataFrame with years as index and sector names as columns.
        initial_year: Base year for price normalisation.
        initial_model_prices: Per-firm anchor prices injected from the model
            at construction time; each firm is scaled by its own anchor so
            relative price levels within a sector are preserved.
    """

    prices: Optional[pd.DataFrame] = None
    initial_year: int = 2014
    initial_model_prices: Optional[np.ndarray] = None

    @property
    def values_dictionary(self) -> dict:
        """Dict-based access to the price DataFrame."""
        return {"prices": self.prices}

    @classmethod
    def from_reader(
        cls,
        reader: SectorExoPricesReader,
        initial_year: int = 2014,
    ) -> SectorExoPrices:
        """Build a SectorExoPrices container from a reader.

        Args:
            reader: Loaded SectorExoPricesReader.
            initial_year: Base year for price normalisation.

        Returns:
            SectorExoPrices container ready to be passed to the model.
        """
        return cls(prices=reader.prices, initial_year=initial_year)


@dataclass
class SectorExoPricesReader:
    """Reader for a single exogenous sector prices CSV.

    The CSV must have years as the row index and sector (industry) codes as
    column headers. Column names must match the ISIC Rev. 4 codes used by the
    model — either aggregated (e.g. "D", "B", "C19") or detailed (e.g. "B05",
    "C19", "D") depending on the industries list the model is configured with.
    Each column represents a sector-level price trajectory; the setter applies
    the same normalised path to all firms within that sector. Values are prices
    in any consistent unit; the setter normalises them to the initial year.

    Example CSV layout (aggregated industries, initial_year=2014):

        year,D,B
        2013,100.0,80.0
        2014,102.5,85.0
        2015,106.0,91.0
        2030,130.0,110.0

    Start one year before initial_year so that Q1 of the first simulation
    year (which maps to initial_year − 0.25) stays within the interpolation
    range.

    Attributes:
        prices: DataFrame loaded from the CSV (None if file absent).
    """

    prices: Optional[pd.DataFrame] = None

    @classmethod
    def read_from_raw_data(
        cls,
        prices_path: Path | str,
    ) -> SectorExoPricesReader:
        """Load the sector prices CSV from disk.

        Args:
            prices_path: Path to the CSV file (index_col=0 assumed).

        Returns:
            SectorExoPricesReader with loaded DataFrame (None if file absent).
        """
        if isinstance(prices_path, str):
            prices_path = Path(prices_path)
        prices = pd.read_csv(prices_path, index_col=0) if prices_path.exists() else None
        return cls(prices=prices)
