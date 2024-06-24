import h5py
import numpy as np
from macro_data import SyntheticCentralBank
from typing import Any

from macromodel.configurations import CentralBankConfiguration
from macromodel.agents.agent import Agent
from macromodel.central_bank.central_bank_ts import create_central_bank_timeseries
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import functions_from_model


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
        functions = functions_from_model(model=configuration.functions, loc="macromodel.central_bank")

        data = synthetic_central_bank.central_bank_data.astype(float).rename_axis("Central Bank ID")

        # Create the corresponding time series object
        ts = create_central_bank_timeseries(data)

        # No additional states initially
        states: dict[str, float | np.ndarray | list[np.ndarray]] = {
            "targeted_inflation_rate": data["targeted_inflation_rate"].values[0],
            "rho": data["rho"].values[0],
            "r_star": data["r_star"].values[0],
            "xi_pi": data["xi_pi"].values[0],
            "xi_gamma": data["xi_gamma"].values[0],
        }

        return cls(
            country_name,
            all_country_names,
            n_industries,
            functions,
            ts,
            states,
        )

    def reset(self, configuration: CentralBankConfiguration) -> None:
        self.gen_reset()
        self.functions = functions_from_model(model=configuration.functions, loc="macromodel.central_bank")

    def compute_rate(self, inflation: float, growth: float) -> float:
        return self.functions["policy_rate"].compute_rate(
            prev_rate=self.ts.current("policy_rate")[0],
            inflation=inflation,
            growth=growth,
            central_bank_states=self.states,
        )

    def save_to_h5(self, group: h5py.Group):
        self.ts.write_to_h5("central_Bank", group)
