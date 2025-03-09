"""Central Bank agent implementation for macroeconomic modeling.

This module implements the central bank agent, which manages:
- Monetary policy implementation
- Interest rate setting
- Price stability targeting
- Economic growth considerations

The central bank plays a crucial role in:
- Inflation control through policy rates
- Economic stabilization
- Financial system oversight
- Macroeconomic management
"""

from typing import Any

import h5py
import numpy as np

from macro_data import SyntheticCentralBank
from macromodel.agents.agent import Agent
from macromodel.agents.central_bank.central_bank_ts import (
    create_central_bank_timeseries,
)
from macromodel.configurations import CentralBankConfiguration
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import functions_from_model, update_functions


class CentralBank(Agent):
    """Central Bank agent responsible for monetary policy.

    This class implements central bank operations including:
    - Policy rate setting
    - Inflation targeting
    - Growth considerations
    - Monetary policy rules

    The agent manages monetary policy through:
    - Interest rate adjustments
    - Inflation target maintenance
    - Economic growth monitoring
    - Policy rule implementation

    Attributes:
        functions (dict[str, Any]): Mapping of function names to implementations
        states (dict[str, float | np.ndarray]): State variables including:
            - targeted_inflation_rate: Target inflation level
            - rho: Interest rate smoothing parameter
            - r_star: Natural real interest rate
            - xi_pi: Inflation gap response coefficient
            - xi_gamma: Output growth response coefficient
        ts (TimeSeries): Time series data for central bank variables
    """

    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        functions: dict[str, Any],
        ts: TimeSeries,
        states: dict[str, float | np.ndarray | list[np.ndarray]],
    ):
        """Initialize the Central Bank agent.

        Args:
            country_name (str): Name of the country this bank serves
            all_country_names (list[str]): List of all countries in the model
            n_industries (int): Number of industries in the economy
            functions (dict[str, Any]): Function implementations for bank operations
            ts (TimeSeries): Time series data for tracking variables
            states (dict[str, float | np.ndarray]): State variables including
                monetary policy parameters
        """
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
        """Create a Central Bank instance from pickled data.

        Initializes the bank with:
        - Policy parameters from synthetic data
        - Configuration settings
        - Country-specific information
        - Initial state variables

        Args:
            synthetic_central_bank (SyntheticCentralBank): Synthetic data
                containing policy parameters and initial states
            configuration (CentralBankConfiguration): Configuration parameters
                for bank operations
            country_name (str): Name of the country this bank serves
            all_country_names (list[str]): List of all countries
            n_industries (int): Number of industries

        Returns:
            CentralBank: Initialized central bank agent
        """
        # Get corresponding functions and parameters
        functions = functions_from_model(model=configuration.functions, loc="macromodel.agents.central_bank")

        data = synthetic_central_bank.central_bank_data.astype(float).rename_axis("Central Bank ID")

        # Create the corresponding time series object
        ts = create_central_bank_timeseries(data)

        # Initialize monetary policy parameters
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
        """Reset the central bank agent to initial state.

        Resets all state variables and updates function implementations
        based on the provided configuration.

        Args:
            configuration (CentralBankConfiguration): New configuration
                parameters for the reset state
        """
        self.gen_reset()
        update_functions(model=configuration.functions, loc="macromodel.agents.central_bank", functions=self.functions)

    def compute_rate(self, inflation: float, growth: float) -> float:
        """Calculate the policy interest rate.

        Determines appropriate policy rate based on:
        - Current inflation relative to target
        - Economic growth rate
        - Previous policy rate
        - Monetary policy rule parameters

        Args:
            inflation (float): Current inflation rate
            growth (float): Current economic growth rate

        Returns:
            float: New policy interest rate
        """
        return self.functions["policy_rate"].compute_rate(
            prev_rate=self.ts.current("policy_rate")[0],
            inflation=inflation,
            growth=growth,
            central_bank_states=self.states,
        )

    def save_to_h5(self, group: h5py.Group):
        """Save central bank data to HDF5 format.

        Stores all time series data in the specified HDF5 group.

        Args:
            group (h5py.Group): HDF5 group to save data in
        """
        self.ts.write_to_h5("central_Bank", group)
