import numpy as np
import pandas as pd

from pathlib import Path
from mergedeep import merge

from model.agents.agent import Agent
from model.timeseries import TimeSeries
from model.util.function_mapping import get_functions
from model.individuals.individual_properties import ActivityStatus
from model.central_government.central_government_ts import (
    create_central_government_timeseries,
)

from typing import Any, Optional


class CentralGovernment(Agent):
    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        year: int,
        t_max: int,
        n_industries: int,
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
        data: pd.DataFrame,
        tax_data: pd.DataFrame,
        taxes_net_subsidies: np.ndarray,
        number_of_unemployed_individuals: int,
        unemployment_benefits_model: Optional[Any],
        other_benefits_model: Optional[Any],
        config: dict[str, Any],
        init_config: dict[str, Any],
    ) -> "CentralGovernment":
        # Parameters
        parameters = {}
        merge(parameters, config, init_config)

        # Get corresponding functions and parameters
        functions = get_functions(
            parameters["functions"],
            loc="model.central_government",
            func_dir=Path(__file__).parent / "func",
        )

        # Create the corresponding time series object
        ts = create_central_government_timeseries(
            data=data,
            number_of_unemployed_individuals=number_of_unemployed_individuals,
        )

        # Additional states
        states: dict[str, float | np.ndarray] = {}
        for state_name in [
            "Value-added Tax",
            "Export Tax",
            "Employer Social Insurance Tax",
            "Employee Social Insurance Tax",
            "Profit Tax",
            "Income Tax",
            "Capital Formation Tax",
        ]:
            if state_name not in tax_data.columns:
                raise ValueError("Missing " + state_name + " from the data for initialising the central government.")
            states[state_name] = tax_data[state_name].values[0]
        states["Taxes Less Subsidies Rates"] = taxes_net_subsidies
        states["unemployment_benefits_model"] = unemployment_benefits_model
        states["other_benefits_model"] = other_benefits_model

        return cls(
            country_name,
            all_country_names,
            year,
            t_max,
            n_industries,
            functions,
            parameters,
            ts,
            states,
        )

    def update_benefits(
        self,
        historic_cpi_inflation: list[np.ndarray],
        exogenous_cpi_inflation: np.ndarray,
        current_unemployment_rate: float,
    ) -> None:
        all_cpi_inflation = np.concatenate(
            (
                exogenous_cpi_inflation,
                np.array(historic_cpi_inflation).flatten(),
            )
        )

        # Unemployment benefits
        self.ts.unemployment_benefits_by_individual.append(
            [
                self.functions["social_benefits"].compute_unemployment_benefits(
                    prev_unemployment_benefits=self.ts.current("unemployment_benefits_by_individual")[0],
                    historic_cpi_inflation=all_cpi_inflation,
                    current_unemployment_rate=current_unemployment_rate,
                    model=self.states["unemployment_benefits_model"],
                )
            ]
        )

        # Regular social transfers to households
        self.ts.total_other_benefits.append(
            [
                self.functions["social_benefits"].compute_regular_transfer_to_households(
                    prev_regular_transfer_to_households=self.ts.current("total_other_benefits")[0],
                    historic_cpi_inflation=all_cpi_inflation,
                    current_unemployment_rate=current_unemployment_rate,
                    model=self.states["other_benefits_model"],
                )
            ]
        )

    def distribute_unemployment_benefits_to_individuals(
        self,
        current_individual_activity_status: np.ndarray,
    ) -> np.ndarray:
        unemployment_benefits = np.zeros_like(current_individual_activity_status)
        unemployment_benefits[current_individual_activity_status == ActivityStatus.UNEMPLOYED] = self.ts.current(
            "unemployment_benefits_by_individual"
        )
        return unemployment_benefits.astype(float)

    def compute_taxes(
        self,
        current_ind_employee_income: np.ndarray,
        current_total_rent_paid: np.ndarray,
        current_income_financial_assets: np.ndarray,
        current_ind_activity: np.ndarray,
        current_ind_realised_cons: np.ndarray,
        current_bank_profits: np.ndarray,
        current_firm_production: np.ndarray,
        current_firm_price: np.ndarray,
        current_firm_profits: np.ndarray,
        current_firm_industries: np.ndarray,
        current_household_new_real_wealth: np.ndarray,
        taxes_less_subsidies_rates: np.ndarray,
        current_total_exports: float,
    ) -> None:
        # Taxes on production
        self.ts.taxes_production.append(
            [np.sum(taxes_less_subsidies_rates[current_firm_industries] * current_firm_production * current_firm_price)]
        )

        # Value-added taxes
        self.ts.taxes_vat.append([self.states["Value-added Tax"] * np.sum(current_ind_realised_cons)])

        # Taxes on capital formation
        self.ts.taxes_cf.append(
            [self.states["Capital Formation Tax"] * np.sum(np.maximum(0.0, current_household_new_real_wealth))]
        )

        # Corporate income taxes
        self.ts.taxes_corporate_income.append(
            [
                self.states["Profit Tax"]
                * (np.sum(np.maximum(current_firm_profits, 0)) + np.sum(np.maximum(current_bank_profits, 0)))
            ]
        )

        # Taxes on exports
        self.ts.taxes_exports.append([self.states["Export Tax"] * current_total_exports])

        # Total wages of employed individuals
        tot_wages_employed_ind = np.sum([current_ind_employee_income[current_ind_activity == ActivityStatus.EMPLOYED]])

        # Taxes on income
        self.ts.taxes_income.append(
            [
                self.states["Income Tax"] * (1 - self.states["Employee Social Insurance Tax"]) * tot_wages_employed_ind
                + self.states["Income Tax"] * current_total_rent_paid.sum()
                + self.states["Income Tax"] * current_income_financial_assets.sum(),
            ]
        )
        self.ts.taxes_rental_income.append([self.states["Income Tax"] * current_total_rent_paid.sum()])

        # Taxes on employer social insurance
        self.ts.taxes_employer_si.append([self.states["Employer Social Insurance Tax"] * tot_wages_employed_ind])

        # Taxes on employee social insurance
        self.ts.taxes_employee_si.append([self.states["Employee Social Insurance Tax"] * tot_wages_employed_ind])

    def compute_taxes_on_products(self) -> float:
        return (
            self.ts.current("taxes_production")[0]
            + self.ts.current("taxes_vat")[0]
            + self.ts.current("taxes_cf")[0]
            + self.ts.current("taxes_exports")[0]
        )

    def compute_revenue(self, household_rent_paid_to_government: float) -> float:
        self.ts.total_rent_received.append([household_rent_paid_to_government])
        return (
            self.ts.current("taxes_production")[0]
            + self.ts.current("taxes_vat")[0]
            + self.ts.current("taxes_cf")[0]
            + self.ts.current("taxes_corporate_income")[0]
            + self.ts.current("taxes_exports")[0]
            + self.ts.current("taxes_income")[0]
            + self.ts.current("taxes_employee_si")[0]
            + self.ts.current("taxes_employer_si")[0]
            + household_rent_paid_to_government
        )
