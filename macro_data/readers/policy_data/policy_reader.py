"""
This module provides utilities for reading and assembling policy datasets used by the
macro model—specifically Canada’s Output-Based Pricing System (OBPS) tables and
consumer carbon price schedules.

Key Features:
- Load OBPS policy datasets from CSV files (policy table, electricity-specific table, rates)
- Load carbon price schedule from an in-memory CSV (bytes) buffer
- Store parsed DataFrames on the reader instance for downstream use
- Return lightweight dict containers with well-named keys

Example:
    ```python
    from pathlib import Path
    from macro_data.readers.policies.policy_reader import PolicyReader

    reader = PolicyReader()

    # OBPS policy data from CSV files on disk
    obps = reader.obps_policy_data_from_raw_data(
        obps_carbon_price_path=Path("data/obps_carbon_price.csv"),
        obps_policy_values_elec_path=Path("data/obps_policy_values_electricity.csv"),
        obps_policy_values_path=Path("data/obps_policy_values.csv"),
    )

    # Carbon price schedule from bytes (e.g., read once from storage or HTTP)
    with open("data/consumer_carbon_price_rates.csv", "rb") as f:
        consumer_bytes = f.read()
    carbon = reader.carbon_price_data_from_raw_data(
        carbon_price_bytes_path=consumer_bytes
    )

    # Access parsed DataFrames
    df_policy = reader.obps_policy_data["df_policy"]
    df_policy_elec = reader.obps_policy_data["df_policy_elec"]
    df_rates = reader.obps_policy_data["df_rates"]
    df_consumer_rates = reader.carbon_prices_data["carbon_price_bytes"]
    ```

Notes:
- `carbon_price_bytes_path` expects **bytes** containing a CSV; if you have a file path,
  read it in binary mode and pass the bytes (see example).
- Methods mutate the instance by setting `obps_policy_data` and `carbon_prices_data`
  and also return the same dicts for convenience.
- No schema validation is performed here; downstream code should validate required columns.
"""

from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class PolicyReader:
    """
    A class for reading and assembling policy datasets used by the macro model,
    specifically Canada's OBPS (Output-Based Pricing System) tables and consumer
    carbon price schedules.

    This class provides simple helpers to load CSV sources into pandas
    DataFrames and keep them attached to the instance for downstream use.

    Args:
        obps_policy_data (Optional[dict[str, pd.DataFrame]]): Preloaded OBPS tables
            (e.g., when restoring from a cache). Keys typically include:
            ``"df_policy"``, ``"df_policy_elec"``, and ``"df_rates"``.
        carbon_prices_data (Optional[dict[str, pd.DataFrame]]): Preloaded consumer
            carbon price schedule (e.g., when restoring from a cache). Expected key:
            ``"carbon_price_bytes"``.
        energy_bundle_prices (Optional[pd.DataFrame]): Preloaded energy bundle price
            timeseries data.

    Attributes:
        obps_policy_data (Optional[dict[str, pd.DataFrame]]): Container for OBPS
            policy datasets after loading. Keys:
            - ``"df_policy"``: OBPS carbon price policy table.
            - ``"df_policy_elec"``: Electricity-specific policy table.
            - ``"df_rates"``: Policy rates table.
        carbon_prices_data (Optional[dict[str, pd.DataFrame]]): Container for the
            consumer carbon price schedule after loading. Keys:
            - ``"carbon_price_bytes"``: DataFrame parsed from a CSV provided as bytes.
        energy_bundle_prices (Optional[pd.DataFrame]): Energy bundle price timeseries data.

    Notes:
        - This class does not perform schema validation; downstream code should verify
          required columns exist.
        - Monetary units and rate conventions depend on the source CSVs.
    """

    def __init__(
        self,
        obps_policy_data: Optional[dict[str, pd.DataFrame]] = None,
        carbon_prices_data: Optional[dict[str, pd.DataFrame]] = None,
        energy_bundle_prices: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the reader with optional preloaded data.

        Args:
            obps_policy_data: Optional OBPS datasets to attach immediately.
            carbon_prices_data: Optional consumer carbon price dataset to attach.
            energy_bundle_prices: Optional energy bundle price timeseries to attach.

        Note:
            Only store attributes in __init__. No processing logic here.
        """
        self.obps_policy_data = obps_policy_data
        self.carbon_prices_data = carbon_prices_data
        self.energy_bundle_prices = energy_bundle_prices

    @classmethod
    def from_data(cls, data_path: Path, **kwargs) -> "PolicyReader":
        """Create reader from policy data directory.

        Args:
            data_path: Path to directory containing policy CSV files
            **kwargs: Additional parameters (unused, for compatibility)

        Returns:
            PolicyReader: New instance with loaded policy data

        Raises:
            FileNotFoundError: If required files are missing
        """
        if isinstance(data_path, str):
            data_path = Path(data_path)

        # Return empty reader if policy directory doesn't exist
        if not data_path.exists():
            return cls(obps_policy_data={}, carbon_prices_data={}, energy_bundle_prices=None)

        # Load OBPS data
        obps_policy_data = {}
        obps_policy_data["df_policy"] = pd.read_csv(data_path / "output_based_price_system_rates.csv")
        obps_policy_data["df_policy_elec"] = pd.read_csv(data_path / "output_based_price_system_policy_values_elec.csv")
        obps_policy_data["df_rates"] = pd.read_csv(data_path / "output_based_price_system_policy_values_disagg.csv")

        # Load carbon prices (read as bytes for compatibility)
        with open(data_path / "consumer_carbon_price_rates.csv", "rb") as f:
            carbon_price_bytes = f.read()
        carbon_prices_data = {"carbon_price_bytes": pd.read_csv(BytesIO(carbon_price_bytes))}

        # Load energy bundle prices
        energy_bundle_prices = pd.read_csv(data_path / "energy_bundle_price.csv")

        return cls(
            obps_policy_data=obps_policy_data,
            carbon_prices_data=carbon_prices_data,
            energy_bundle_prices=energy_bundle_prices,
        )

    def get_energy_bundle_price(self, timestep: int, industry: str) -> float:
        """Get energy bundle price for specific timestep and industry.

        Args:
            timestep: Simulation timestep (0-based)
            industry: Industry code (e.g., 'B05a', 'B05b', 'B05c')

        Returns:
            Price multiplier for the given timestep and industry

        Raises:
            KeyError: If industry not found in data
            IndexError: If timestep out of range
        """
        if self.energy_bundle_prices is None:
            raise ValueError("Energy bundle prices not loaded")

        row = self.energy_bundle_prices[self.energy_bundle_prices["Timestep"] == timestep]
        if len(row) == 0:
            raise IndexError(f"Timestep {timestep} not found in energy bundle prices")

        return row[industry].values[0]

    def obps_policy_data_from_raw_data(
        self,
        obps_carbon_price_path: Path | str,
        obps_policy_values_elec_path: Path | str,
        obps_policy_values_path: Path | str,
    ) -> dict[str, pd.DataFrame]:
        """
        Load OBPS policy data from CSV files on disk.

        Args:
            obps_carbon_price_path: Path to the OBPS carbon price policy CSV.
            obps_policy_values_elec_path: Path to the electricity-specific
                OBPS policy CSV.
            obps_policy_values_path: Path to the general OBPS policy values CSV.

        Returns:
            dict[str, pd.DataFrame]: A dictionary containing:
                - ``"df_policy"``: OBPS carbon price policy table.
                - ``"df_policy_elec"``: Electricity-specific policy table.
                - ``"df_rates"``: Policy rates table.

        Side Effects:
            Sets :attr:`obps_policy_data` on the instance.

        Raises:
            FileNotFoundError: If any provided path does not exist.
            pd.errors.ParserError: If a CSV cannot be parsed.
        """
        obps_policy_data: dict[str, pd.DataFrame] = {}
        obps_policy_data["df_policy"] = pd.read_csv(obps_carbon_price_path)
        obps_policy_data["df_policy_elec"] = pd.read_csv(obps_policy_values_elec_path)
        obps_policy_data["df_rates"] = pd.read_csv(obps_policy_values_path)

        self.obps_policy_data = obps_policy_data
        return obps_policy_data

    def carbon_price_data_from_raw_data(
        self,
        carbon_price_bytes_path: bytes | bytearray,
    ) -> dict[str, pd.DataFrame]:
        """
        Load the consumer carbon price schedule from CSV **bytes**.

        Use this when the CSV is provided as an in-memory byte stream
        (e.g., fetched from object storage or an HTTP response). For a
        file on disk, read it in binary mode and pass the bytes:

            >>> with open("consumer_carbon_price_rates.csv", "rb") as f:
            ...     df = reader.carbon_price_data_from_raw_data(f.read())

        Args:
            carbon_price_bytes_path: Raw CSV content as bytes (or bytearray).

        Returns:
            dict[str, pd.DataFrame]: A dictionary containing:
                - ``"carbon_price_bytes"``: DataFrame parsed from the byte stream.

        Side Effects:
            Sets :attr:`carbon_prices_data` on the instance.

        Raises:
            pd.errors.ParserError: If the CSV bytes cannot be parsed.
        """
        carbon_prices_data: dict[str, pd.DataFrame] = {}
        carbon_prices_data["carbon_price_bytes"] = pd.read_csv(BytesIO(carbon_price_bytes_path))

        self.carbon_prices_data = carbon_prices_data
        return carbon_prices_data
