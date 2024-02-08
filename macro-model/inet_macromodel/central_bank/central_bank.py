import numpy as np
import pandas as pd
import h5py

from pathlib import Path

from inet_data import SyntheticCentralBank

from configurations import CentralBankConfiguration
from inet_macromodel.agents.agent import Agent
from inet_macromodel.timeseries import TimeSeries
from inet_macromodel.util.function_mapping import get_functions, functions_from_model
from inet_macromodel.central_bank.central_bank_ts import create_central_bank_timeseries

from typing import Any


class CentralBank(Agent):
    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        functions: dict[str, Any],
        ts: TimeSeries,
        states: dict[str, float | np.ndarray | list[np.ndarray]],
    ):
        super().__init__(
            country_name,
            all_country_names,
            n_industries,
            0,
            0,
            ts,
            states,
        )

        self.functions = functions

    @classmethod
    def from_pickled_agent(
        cls,
        synthetic_central_bank: SyntheticCentralBank,
        configuration: CentralBankConfiguration,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
    ) -> "CentralBank":
        # Get corresponding functions and parameters
        functions = functions_from_model(model=configuration.functions, loc="inet_macromodel.central_bank")

        data = synthetic_central_bank.central_bank_data.astype(float).rename_axis("Central Bank ID")

        # Create the corresponding time series object
        ts = create_central_bank_timeseries(data)

        # No additional states initially
        states: dict[str, float | np.ndarray | list[np.ndarray]] = {}

        return cls(
            country_name,
            all_country_names,
            n_industries,
            functions,
            ts,
            states,
        )

    def compute_rate(self) -> float:
        return self.functions["policy_rate"].compute_rate(prev_rate=self.ts.current("policy_rate")[0])

    def save_to_h5(self, group: h5py.Group):
        self.ts.write_to_h5("central_Bank", group)
