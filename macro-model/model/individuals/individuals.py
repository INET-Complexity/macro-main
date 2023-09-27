import warnings
import numpy as np
import pandas as pd

from pathlib import Path
from model.agents.agent import Agent
from model.timeseries import TimeSeries
from model.util.property_mapping import map_to_enum
from model.util.function_mapping import get_functions
from model.individuals.individuals_ts import create_individuals_timeseries
from model.individuals.individual_properties import (
    ActivityStatus,
    Gender,
    Education,
)

from typing import Any


class Individuals(Agent):
    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        year: int,
        t_max: int,
        n_industries: int,
        n_transactors: int,
        functions: dict[str, Any],
        parameters: dict[str, Any],
        ts: TimeSeries,
        states: dict[str, float | np.ndarray | list[np.ndarray]],
    ):
        super().__init__(
            country_name,
            all_country_names,
            year,
            t_max,
            n_industries,
            0,
            0,
            functions,
            parameters,
            ts,
            states,
        )

    @classmethod
    def from_data(
        cls,
        country_name: str,
        all_country_names: list[str],
        year: int,
        t_max: int,
        n_industries: int,
        scale: int,
        data: pd.DataFrame,
        config: dict[str, Any],
    ) -> "Individuals":
        # Get corresponding functions and parameters
        functions = get_functions(
            config["functions"],
            loc="model.individuals",
            func_dir=Path(__file__).parent / "func",
        )
        if "parameters" in config.keys():
            parameters = config["parameters"].copy()
        else:
            parameters = {}

        # Create the corresponding time series object
        ts = create_individuals_timeseries(data, scale)

        # Additional states
        states: dict[str, float | np.ndarray | list[np.ndarray]] = {}
        for state_name in [
            "Gender",
            "Age",
            "Education",
            "Activity Status",
            "Employment Industry",
            "Income",
            "Employee Income",
            "Income from Unemployment Benefits",
            "Corresponding Household ID",
            "Corresponding Firm ID",
        ]:
            if state_name not in data.columns:
                raise ValueError("Missing " + state_name + " from the data for initialising individuals.")
            states[state_name] = data[state_name].values

        # Update the activity status
        states["Activity Status"] = np.array(map_to_enum(states["Activity Status"], ActivityStatus))

        # Update gender
        states["Gender"] = np.array(map_to_enum(states["Gender"], Gender))

        # Level of education
        states["Education"] = np.array(map_to_enum(states["Education"], Education))

        # Cosmetics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            states["Corresponding Household ID"] = states["Corresponding Household ID"].astype(int)
            states["Corresponding Firm ID"] = states["Corresponding Firm ID"].astype(int)
            states["Corresponding Firm ID"][states["Corresponding Firm ID"] < 0] = -1

        return cls(
            country_name,
            all_country_names,
            year,
            t_max,
            n_industries,
            ts.current("n_individuals"),
            functions,
            parameters,
            ts,
            states,
        )

    def compute_labour_inputs(self) -> np.ndarray:
        return self.functions["labour_inputs"].update_labour_inputs(
            previous_individuals_labour_inputs=self.ts.current("labour_inputs"),
            current_individuals_activity=self.states["Activity Status"],
        )

    def compute_reservation_wages(
        self,
        unemployment_benefits_by_individual: float,
    ) -> np.ndarray:
        return (
            self.functions["reservation_wages"]
            .compute_reservation_wages(
                historic_wages=self.ts.historic("employee_income"),
                current_individuals_activity=self.states["Activity Status"],
                current_unemployment_benefits_by_individual=unemployment_benefits_by_individual,
            )
            .astype(float)
        )

    def compute_income(self) -> np.ndarray:
        return (
            self.functions["income"].compute_income(
                current_individual_activity_status=self.states["Activity Status"],
                current_wage=self.ts.current("employee_income"),
                individual_social_benefits=self.ts.current("income_from_unemployment_benefits"),
            )
        ).astype(float)

    def update_demography(self) -> None:
        self.ts.n_individuals.append(
            self.functions["demography"].update(
                self.ts.current("n_individuals"),
            )
        )
