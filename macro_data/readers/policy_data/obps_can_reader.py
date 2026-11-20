"""Reader and container for Canada Output-Based Pricing System (OBPS) policy data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class OBPSCANData:
    """Container for Canada OBPS policy tables loaded from CSV files.

    Attributes:
        df_rates: Carbon price trajectory. Rows are years; columns are
            'Date' plus one column per province/jurisdiction code.
        df_policy: Per-industry emission standards. Must have columns
            'Industry', 'reduction_factor', and 'tightening_rate'.
        df_policy_elec: Optional electricity-specific tightening rates.
            Same structure as df_policy but for electricity sectors only.
    """

    df_rates: pd.DataFrame
    df_policy: pd.DataFrame
    df_policy_elec: Optional[pd.DataFrame] = None


@dataclass
class OBPSCANReader:
    """Reader for Canada OBPS CSV data files.

    CSV formats
    -----------
    rates_path:
        Date, CAN, CAN_AB, CAN_BC, ...
        2014, 15.0, 10.0, 12.0, ...
        2015, 20.0, 15.0, 15.0, ...

    policy_path:
        Industry, reduction_factor, tightening_rate
        B05a, 0.80, 0.02
        C19,  0.75, 0.02

    policy_elec_path (optional):
        same structure as policy_path but for electricity sectors.
    """

    obps_data: Optional[OBPSCANData] = None

    @classmethod
    def read_from_raw_data(
        cls,
        rates_path: Path | str,
        policy_path: Path | str,
        policy_elec_path: Optional[Path | str] = None,
    ) -> OBPSCANReader:
        """Load OBPS data from CSV files.

        Args:
            rates_path: Path to carbon price rates CSV.
            policy_path: Path to per-industry policy values CSV.
            policy_elec_path: Optional path to electricity-specific CSV.

        Returns:
            OBPSCANReader with loaded OBPSCANData, or obps_data=None if any
            required file is absent.
        """
        rates_path = Path(rates_path)
        policy_path = Path(policy_path)

        if not rates_path.exists() or not policy_path.exists():
            return cls(obps_data=None)

        df_rates = pd.read_csv(rates_path)
        df_policy = pd.read_csv(policy_path)

        df_policy_elec = None
        if policy_elec_path is not None:
            policy_elec_path = Path(policy_elec_path)
            if policy_elec_path.exists():
                df_policy_elec = pd.read_csv(policy_elec_path)

        return cls(obps_data=OBPSCANData(df_rates=df_rates, df_policy=df_policy, df_policy_elec=df_policy_elec))
