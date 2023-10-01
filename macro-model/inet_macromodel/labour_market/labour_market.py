import numpy as np

from pathlib import Path

from inet_macromodel.firms.firms import Firms
from inet_macromodel.timeseries import TimeSeries
from inet_macromodel.households.households import Households
from inet_macromodel.util.function_mapping import get_functions
from inet_macromodel.individuals.individuals import Individuals
from inet_macromodel.individuals.individual_properties import ActivityStatus
from inet_macromodel.labour_market.labour_market_ts import create_labour_market_timeseries

from typing import Any


class LabourMarket:
    def __init__(
        self,
        country_name: str,
        year: int,
        t_max: int,
        n_industries: int,
        functions: dict[str, Any],
        parameters: dict[str, Any],
        ts: TimeSeries,
    ):
        self.country_name = country_name
        self.year = year
        self.t_max = t_max
        self.n_industries = n_industries
        self.functions = functions
        self.parameters = parameters
        self.ts = ts

    @classmethod
    def from_data(
        cls,
        country_name: str,
        year: int,
        t_max: int,
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
            year,
            t_max,
            n_industries,
            functions,
            parameters,
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
