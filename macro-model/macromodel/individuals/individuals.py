import warnings
from typing import Any

import h5py
import numpy as np
from macro_data import SyntheticPopulation

from macromodel.agents.agent import Agent
from macromodel.configurations import IndividualsConfiguration
from macromodel.individuals.individual_properties import ActivityStatus, Education, Gender
from macromodel.individuals.individuals_ts import create_individuals_timeseries
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import functions_from_model, update_functions
from macromodel.util.property_mapping import map_to_enum


class Individuals(Agent):
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
        self.functions: dict[str, Any] = functions

    @classmethod
    def from_pickled_agent(
        cls,
        synthetic_population: SyntheticPopulation,
        configuration: IndividualsConfiguration,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        scale: int,
    ) -> "Individuals":
        data = (synthetic_population.individual_data.astype(float)).rename_axis("Individual ID")

        functions = functions_from_model(model=configuration.functions, loc="macromodel.individuals")

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
            "Corresponding Invested Firm",
            "Corresponding Invested Bank",
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

        states["Started New Job"] = np.full(len(states["Activity Status"]), False)
        states["Offered Wage of Accepted Job"] = np.zeros(len(states["Activity Status"]))
        states["Dividend Payout Ratio"] = 0.0

        def fillnan(x: np.ndarray) -> np.ndarray:
            return np.where(np.isnan(x), -1, x)

        # Cosmetics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            states["Corresponding Household ID"] = fillnan(states["Corresponding Household ID"])
            states["Corresponding Household ID"] = states["Corresponding Household ID"].astype(int)
            states["Corresponding Firm ID"] = fillnan(states["Corresponding Firm ID"])
            states["Corresponding Firm ID"] = states["Corresponding Firm ID"].astype(int)
            states["Corresponding Firm ID"][states["Corresponding Firm ID"] < 0] = -1

            states["Corresponding Invested Firm"] = fillnan(states["Corresponding Invested Firm"])
            states["Corresponding Invested Firm"] = states["Corresponding Invested Firm"].astype(int)
            states["Corresponding Invested Firm"][states["Corresponding Invested Firm"] < 0] = -1

            states["Corresponding Invested Bank"] = fillnan(states["Corresponding Invested Bank"])
            states["Corresponding Invested Bank"] = states["Corresponding Invested Bank"].astype(int)
            states["Corresponding Invested Bank"][states["Corresponding Invested Bank"] < 0] = -1

        return cls(country_name, all_country_names, n_industries, functions, ts, states)

    def reset(self, configuration: IndividualsConfiguration):
        self.gen_reset()
        update_functions(functions=self.functions, model=configuration.functions, loc="macromodel.individuals")

    def compute_labour_inputs(self) -> np.ndarray:
        return self.functions["labour_inputs"].update_labour_inputs(
            previous_individuals_labour_inputs=self.ts.current("labour_inputs"),
            current_individuals_activity=self.states["Activity Status"],
        )

    def compute_reservation_wages(self, unemployment_benefits_by_individual: float) -> np.ndarray:
        return (
            self.functions["reservation_wages"]
            .compute_reservation_wages(
                historic_wages=self.ts.historic("employee_income"),
                current_individuals_activity=self.states["Activity Status"],
                current_unemployment_benefits_by_individual=unemployment_benefits_by_individual,
            )
            .astype(float)
        )

    def compute_expected_income(
        self,
        expected_firm_profits: np.ndarray,
        expected_bank_profits: np.ndarray,
        cpi: float,
        expected_inflation: float,
        income_taxes: float,
        tau_firm: float,
    ) -> np.ndarray:
        return (
            self.functions["income"].compute_expected_income(
                current_individual_activity_status=self.states["Activity Status"],
                current_wage=self.ts.current("employee_income"),
                individual_social_benefits=self.ts.current("income_from_unemployment_benefits"),
                expected_firm_profits=expected_firm_profits,
                corr_invested_firms=self.states["Corresponding Invested Firm"],
                expected_bank_profits=expected_bank_profits,
                corr_invested_banks=self.states["Corresponding Invested Bank"],
                cpi=cpi,
                expected_inflation=expected_inflation,
                dividend_payout_ratio=self.states["Dividend Payout Ratio"],
                income_taxes=income_taxes,
                tau_firm=tau_firm,
            )
        ).astype(float)

    def compute_income(
        self,
        firm_profits: np.ndarray,
        bank_profits: np.ndarray,
        cpi: float,
        income_taxes: float,
        tau_firm: float,
    ) -> np.ndarray:
        return (
            self.functions["income"].compute_income(
                current_individual_activity_status=self.states["Activity Status"],
                current_wage=self.ts.current("employee_income"),
                individual_social_benefits=self.ts.current("income_from_unemployment_benefits"),
                firm_profits=firm_profits,
                corr_invested_firms=self.states["Corresponding Invested Firm"],
                bank_profits=bank_profits,
                corr_invested_banks=self.states["Corresponding Invested Bank"],
                cpi=cpi,
                dividend_payout_ratio=self.states["Dividend Payout Ratio"],
                income_taxes=income_taxes,
                tau_firm=tau_firm,
            )
        ).astype(float)

    def update_demography(self) -> None:
        self.ts.n_individuals.append(self.functions["demography"].update(self.ts.current("n_individuals")))

    def save_to_h5(self, group: h5py.Group):
        self.ts.write_to_h5("individuals", group)

    @property
    def n_individuals(self) -> int:
        return self.states["Age"].shape[0]
