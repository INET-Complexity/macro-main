"""Labour market simulation and clearing mechanism.

This module implements the core labour market functionality, managing employment
relationships between firms and individuals across industries. It handles job
matching, wage determination, and employment status tracking.

Key Features:
1. Employment Management:
   - Job matching between firms and individuals
   - Industry-specific employment tracking
   - Employment status transitions
   - Workforce allocation

2. Market Clearing:
   - Supply-demand matching
   - Wage determination
   - Employment adjustments
   - Labour cost calculation

3. Employment Dynamics:
   - New hires tracking
   - Firing mechanisms
   - Voluntary quits
   - Industry transitions

4. Market Analysis:
   - Employment statistics
   - Labour costs
   - Industry distribution
   - Market efficiency metrics
"""

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from macromodel.agents.firms import Firms
from macromodel.agents.households.households import Households
from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.agents.individuals.individuals import Individuals
from macromodel.configurations import LabourMarketConfiguration
from macromodel.markets.labour_market.labour_market_ts import (
    create_labour_market_timeseries,
)
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import (
    functions_from_model,
    get_functions,
    update_functions,
)


class LabourMarket:
    """Labour market implementation managing employment relationships.

    This class implements the core labour market functionality, handling
    job matching, employment status tracking, and market clearing across
    multiple industries.

    The market manages:
    - Employment relationships between firms and individuals
    - Industry-specific labour allocation
    - Hiring and firing decisions
    - Wage determination and labour costs
    - Market clearing and efficiency metrics
    """

    def __init__(
        self,
        country_name: str,
        n_industries: int,
        functions: dict[str, Any],
        ts: TimeSeries,
    ):
        """Initialize a labour market instance.

        Args:
            country_name: Name of the country/region
            n_industries: Number of industries in the economy
            functions: Dict of market functions (clearing, matching, etc.)
            ts: Time series object for tracking market evolution
        """
        self.country_name = country_name
        self.n_industries = n_industries
        self.functions = functions
        self.ts = ts

    @classmethod
    def from_agents(
        cls,
        individuals: Individuals,
        labour_market_configuration: LabourMarketConfiguration,
        country_name: str,
        n_industries: int,
    ):
        """Create a labour market instance from agent data.

        This method initializes a labour market using existing agent
        data, typically used when setting up a new simulation from
        a pre-existing agent population.

        Args:
            individuals: Individual agents with employment data
            labour_market_configuration: Market configuration parameters
            country_name: Name of the country/region
            n_industries: Number of industries in the economy

        Returns:
            LabourMarket: New market instance initialized with agent data
        """
        # initial_individual_activity=individuals.states["Activity Status"],
        #                 initial_individual_employment_industry=individuals.states["Employment Industry"],
        #             )

        initial_individual_activity = individuals.states["Activity Status"]
        initial_individual_employment_industry = individuals.states["Employment Industry"]

        functions = functions_from_model(labour_market_configuration.functions, loc="macromodel.markets.labour_market")
        ts = create_labour_market_timeseries(
            initial_individual_activity=initial_individual_activity,
            initial_individual_employment_industry=initial_individual_employment_industry,
            n_industries=n_industries,
        )

        return cls(
            country_name,
            n_industries,
            functions,
            ts,
        )

    @classmethod
    def from_data(
        cls,
        country_name: str,
        n_industries: int,
        initial_individual_activity: np.ndarray,
        initial_individual_employment_industry: np.ndarray,
        config: dict[str, Any],
    ) -> "LabourMarket":
        """Create a labour market instance from raw data.

        This method initializes a labour market using raw data arrays,
        providing flexibility in data sources and market setup.

        Args:
            country_name: Name of the country/region
            n_industries: Number of industries in the economy
            initial_individual_activity: Initial employment status array
            initial_individual_employment_industry: Initial industry
                assignments array
            config: Configuration dictionary including functions and
                parameters

        Returns:
            LabourMarket: New market instance initialized with the data
        """
        # Get corresponding functions
        functions = get_functions(
            config["functions"],
            loc="macromodel.markets.labour_market",
            func_dir=Path(__file__).parent / "func",
        )

        # Create the corresponding time series object
        ts = create_labour_market_timeseries(
            initial_individual_activity=initial_individual_activity,
            initial_individual_employment_industry=initial_individual_employment_industry,
            n_industries=n_industries,
        )

        return cls(
            country_name,
            n_industries,
            functions,
            ts,
        )

    def reset(self, configuration: LabourMarketConfiguration):
        """Reset the labour market to initial state.

        This method restores the market to its original configuration,
        useful for running multiple simulations or scenarios.

        Args:
            configuration: New configuration parameters to apply
        """
        update_functions(model=configuration.functions, loc="macromodel.agents.labour_market", functions=self.functions)
        self.ts.reset()

    def clear(
        self,
        firms: Firms,
        households: Households,
        individuals: Individuals,
    ) -> np.ndarray:
        """Clear the labour market by matching workers with firms.

        This method executes the market clearing algorithm to match
        individuals with job openings, process separations, and
        calculate labour costs.

        The clearing process:
        1. Track pre-clearing employment
        2. Execute market clearing algorithm
        3. Process new hires and separations
        4. Update employment statistics
        5. Calculate labour costs

        Args:
            firms: Firm agents with job openings
            households: Household agents
            individuals: Individual agents seeking employment

        Returns:
            np.ndarray: Array of labour costs by industry

        Note:
            The specific clearing logic is defined in the configured
            clearing function, allowing for different matching mechanisms.
        """
        # The number of employed individuals before labour market clearing
        self.ts.num_employed_individuals_before_clearing.append(
            [np.sum(individuals.states["Activity Status"] == ActivityStatus.EMPLOYED)]
        )

        # Clear the labour market
        (
            labour_costs,
            num_newly_joining,
            num_newly_randomly_fired,
            num_newly_randomly_quit,
            num_newly_fired,
        ) = self.functions["clearing"].clear(
            firms=firms,
            households=households,
            individuals=individuals,
        )

        # Update a few aggregates
        self.ts.num_individuals_newly_joining.append([num_newly_joining])
        self.ts.num_individuals_newly_randomly_fired.append([num_newly_randomly_fired])
        self.ts.num_individuals_newly_randomly_quit.append([num_newly_randomly_quit])
        self.ts.num_individuals_newly_fired.append([num_newly_fired])
        self.ts.num_individuals_newly_leaving.append(
            [num_newly_randomly_fired + num_newly_randomly_quit + num_newly_fired]
        )

        # Number of employed individuals by sector
        num_employed = np.zeros(self.n_industries)
        for g in range(self.n_industries):
            num_employed[g] = np.sum(
                np.logical_and(
                    individuals.states["Employment Industry"] == g,
                    individuals.states["Activity Status"] == ActivityStatus.EMPLOYED,
                )
            )
        self.ts.num_employed_individuals_by_sector.append(num_employed)

        return labour_costs

    def save_to_h5(self, group: h5py.Group):
        """Save market state to HDF5 format.

        Args:
            group: HDF5 group to save the market data into
        """
        self.ts.write_to_h5("labour_market", group)
