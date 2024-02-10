import h5py
import numpy as np
from pathlib import Path
from typing import Any

from macromodel.configurations import LabourMarketConfiguration
from macromodel.firms.firms import Firms
from macromodel.households.households import Households
from macromodel.individuals.individual_properties import ActivityStatus
from macromodel.individuals.individuals import Individuals
from macromodel.labour_market.labour_market_ts import create_labour_market_timeseries
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import get_functions, functions_from_model


class LabourMarket:
    def __init__(
        self,
        country_name: str,
        n_industries: int,
        functions: dict[str, Any],
        ts: TimeSeries,
    ):
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
        # initial_individual_activity=individuals.states["Activity Status"],
        #                 initial_individual_employment_industry=individuals.states["Employment Industry"],
        #             )

        initial_individual_activity = individuals.states["Activity Status"]
        initial_individual_employment_industry = individuals.states["Employment Industry"]

        functions = functions_from_model(labour_market_configuration.functions, loc="inet_macromodel.labour_market")
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
        # Parameters
        if "parameters" in config.keys():
            parameters = config["parameters"].copy()
        else:
            parameters = {}

        # Get corresponding functions
        functions = get_functions(
            config["functions"],
            loc="inet_macromodel.labour_market",
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

    def clear(
        self,
        firms: Firms,
        households: Households,
        individuals: Individuals,
    ) -> np.ndarray:
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
        self.ts.write_to_h5("labour_market", group)
