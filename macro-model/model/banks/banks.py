import numpy as np
import pandas as pd

from pathlib import Path
from mergedeep import merge

from model.agents.agent import Agent
from model.timeseries import TimeSeries
from model.util.function_mapping import get_functions
from model.banks.banks_ts import create_banks_timeseries
from model.credit_market.credit_market import CreditMarket

from typing import Any


class Banks(Agent):
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
        scale: int,
        data: pd.DataFrame,
        corr_firms: pd.DataFrame,
        corr_households: pd.DataFrame,
        policy_rate_markup: float,
        long_term_ir: float,
        config: dict[str, Any],
        init_config: dict[str, Any],
    ) -> "Banks":
        parameters = {}
        merge(parameters, config["parameters"], init_config["parameters"])

        # Get corresponding functions and parameters
        functions = get_functions(
            config["functions"],
            loc="model.banks",
            func_dir=Path(__file__).parent / "func",
        )
        parameters["policy_rate_markup"] = policy_rate_markup

        # Create the corresponding time series object
        ts = create_banks_timeseries(
            bank_data=data,
            long_term_ir=long_term_ir,
            scale=scale,
        )

        # Additional states
        states: dict[str, float | np.ndarray | list[np.ndarray]] = {
            "corr_firms": [corr_firms.values[i][0] for i in range(len(corr_firms.values))],
            "corr_households": [corr_households.values[i][0] for i in range(len(corr_households.values))],
            "is_insolvent": np.full(ts.current("n_banks"), False),
        }

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

    def set_interest_rates(self, central_bank_policy_rate: float) -> None:
        # On loans
        self.states["interest_rate_on_firm_short_term_loans_function"] = self.functions[
            "interest_rates"
        ].get_interest_rate_on_firm_short_term_loans_function(
            central_bank_policy_rate=central_bank_policy_rate,
            bank_markup_interest_rate_loans=np.full(self.ts.current("n_banks"), self.parameters["policy_rate_markup"]),
        )
        self.states["interest_rate_on_firm_long_term_loans_function"] = self.functions[
            "interest_rates"
        ].get_interest_rate_on_firm_long_term_loans_function(
            central_bank_policy_rate=central_bank_policy_rate,
            bank_markup_interest_rate_loans=np.full(self.ts.current("n_banks"), self.parameters["policy_rate_markup"]),
        )
        self.states["interest_rate_on_household_payday_loans_function"] = self.functions[
            "interest_rates"
        ].get_interest_rate_on_household_payday_loans_function(
            central_bank_policy_rate=central_bank_policy_rate,
            bank_markup_interest_rate_loans=np.full(self.ts.current("n_banks"), self.parameters["policy_rate_markup"]),
        )
        self.states["interest_rate_on_household_consumption_expansion_loans_function"] = self.functions[
            "interest_rates"
        ].get_interest_rate_on_household_consumption_expansion_loans_function(
            central_bank_policy_rate=central_bank_policy_rate,
            bank_markup_interest_rate_loans=np.full(
                self.ts.current("n_banks"),
                self.parameters["initial_markup_interest_rate_household_consumption_loans"]["value"],
            ),
        )
        self.states["interest_rate_on_mortgages_function"] = self.functions[
            "interest_rates"
        ].get_interest_rate_on_mortgages_function(
            central_bank_policy_rate=central_bank_policy_rate,
            bank_markup_interest_rate_loans=np.full(
                self.ts.current("n_banks"), self.parameters["initial_markup_mortgage_interest_rate"]["value"]
            ),
        )

        # On deposits
        self.ts.interest_rate_on_firm_deposits.append(
            self.functions["interest_rates"].compute_interest_rate_on_firms_deposits(
                central_bank_policy_rate=central_bank_policy_rate,
                n_banks=self.ts.current("n_banks"),
            )
        )
        self.ts.average_interest_rate_on_firm_deposits.append(
            [self.ts.current("interest_rate_on_firm_deposits").mean()]
        )
        self.ts.overdraft_rate_on_firm_deposits.append(
            self.functions["interest_rates"].compute_overdraft_rate_on_firm_deposits(
                central_bank_policy_rate=central_bank_policy_rate,
                bank_markup_interest_rate_overdraft_firm=np.full(
                    self.ts.current("n_banks"), self.parameters["policy_rate_markup"]
                ),
            )
        )
        self.ts.average_overdraft_rate_on_firm_deposits.append(
            [self.ts.current("overdraft_rate_on_firm_deposits").mean()]
        )
        self.ts.interest_rate_on_household_deposits.append(
            self.functions["interest_rates"].compute_interest_rate_on_household_deposits(
                central_bank_policy_rate=central_bank_policy_rate,
                n_banks=self.ts.current("n_banks"),
            )
        )
        self.ts.average_interest_rate_on_household_deposits.append(
            [self.ts.current("interest_rate_on_household_deposits").mean()]
        )
        self.ts.overdraft_rate_on_household_deposits.append(
            self.functions["interest_rates"].compute_overdraft_rate_on_household_deposits(
                central_bank_policy_rate=central_bank_policy_rate,
                bank_markup_interest_rate_overdraft_household=np.full(
                    self.ts.current("n_banks"),
                    self.parameters["initial_markup_interest_rate_overdraft_households"]["value"],
                ),
            )
        )
        self.ts.average_overdraft_rate_on_household_deposits.append(
            [self.ts.current("overdraft_rate_on_household_deposits").mean()]
        )

    def compute_interest_received_on_deposits(self, central_bank_policy_rate: float) -> np.ndarray:
        return (
            central_bank_policy_rate * np.maximum(0, self.ts.current("deposits"))
            + self.ts.current("overdraft_rate_on_firm_deposits")
            * np.maximum(0, -self.ts.current("deposits_from_firms"))
            + self.ts.current("overdraft_rate_on_household_deposits")
            * np.maximum(0, -self.ts.current("deposits_from_households"))
        ) - (
            central_bank_policy_rate * np.maximum(0, -self.ts.current("deposits"))
            + self.ts.current("interest_rate_on_firm_deposits") * np.maximum(0, self.ts.current("deposits_from_firms"))
            + self.ts.current("interest_rate_on_household_deposits")
            * np.maximum(0, self.ts.current("deposits_from_households"))
        )

    def compute_profits(self) -> np.ndarray:
        return self.ts.current("interest_received_on_loans") + self.ts.current("interest_received_on_deposits")

    def update_deposits(
        self,
        current_firm_deposits: np.ndarray,
        current_household_deposits: np.ndarray,
        firm_corresponding_bank: np.ndarray,
        households_corresponding_bank: np.ndarray,
    ) -> None:
        current_deposits_from_firms = np.bincount(firm_corresponding_bank, weights=current_firm_deposits)
        current_deposits_from_households = np.bincount(
            households_corresponding_bank, weights=current_household_deposits
        )
        self.ts.deposits_from_firms.append(current_deposits_from_firms)
        self.ts.total_deposits_from_firms.append([current_deposits_from_firms.sum()])
        self.ts.deposits_from_households.append(current_deposits_from_households)
        self.ts.total_deposits_from_households.append([current_deposits_from_households.sum()])

    def update_loans(self, credit_market: CreditMarket) -> None:
        self.ts.short_term_loans_to_firms.append(
            credit_market.compute_outstanding_short_term_firm_loans_by_bank(n_banks=self.ts.current("n_banks"))
        )
        self.ts.total_short_term_loans_to_firms.append([self.ts.current("short_term_loans_to_firms").sum()])
        self.ts.long_term_loans_to_firms.append(
            credit_market.compute_outstanding_long_term_firm_loans_by_bank(n_banks=self.ts.current("n_banks"))
        )
        self.ts.total_long_term_loans_to_firms.append([self.ts.current("long_term_loans_to_firms").sum()])

        self.ts.payday_loans_to_households.append(
            credit_market.compute_outstanding_household_payday_loans_by_bank(n_banks=self.ts.current("n_banks"))
        )
        self.ts.total_payday_loans_to_households.append([self.ts.current("payday_loans_to_households").sum()])

        self.ts.consumption_expansion_loans_to_households.append(
            credit_market.compute_outstanding_household_ce_loans_by_bank(n_banks=self.ts.current("n_banks"))
        )
        self.ts.total_consumption_expansion_loans_to_households.append(
            [self.ts.current("consumption_expansion_loans_to_households").sum()]
        )
        self.ts.mortgages_to_households.append(
            credit_market.compute_outstanding_mortgages_by_bank(n_banks=self.ts.current("n_banks"))
        )
        self.ts.total_mortgages_to_households.append([self.ts.current("mortgages_to_households").sum()])
        self.ts.total_outstanding_loans.append(
            credit_market.compute_outstanding_loans_by_bank(n_banks=self.ts.current("n_banks"))
        )

    def compute_market_share(self) -> np.ndarray:
        total_amount_of_loans_and_deposits = (
            np.absolute(self.ts.current("total_outstanding_loans")).sum()
            + np.absolute(self.ts.current("deposits_from_firms")).sum()
            + np.absolute(self.ts.current("deposits_from_households")).sum()
        )
        if total_amount_of_loans_and_deposits > 0:
            return (
                np.absolute(self.ts.current("total_outstanding_loans"))
                + np.absolute(self.ts.current("deposits_from_firms"))
                + np.absolute(self.ts.current("deposits_from_households"))
            ) / total_amount_of_loans_and_deposits
        else:
            return np.full(self.ts.current("n_banks"), 1.0 / self.ts.current("n_banks"))

    def compute_equity(self, profit_taxes: float) -> np.ndarray:
        return (
            self.ts.current("equity")
            + self.ts.current("profits")
            - profit_taxes * np.maximum(0.0, self.ts.current("profits"))
        )

    def compute_liability(self) -> np.ndarray:
        return (
            self.ts.current("equity")
            + np.maximum(0, self.ts.current("deposits_from_firms"))
            + np.maximum(0, self.ts.current("deposits_from_households"))
            + np.maximum(0, -self.ts.current("deposits"))
        )

    def compute_deposits(self) -> np.ndarray:
        return (
            self.ts.current("deposits_from_firms")
            + self.ts.current("deposits_from_households")
            + self.ts.current("equity")
            - self.ts.current("total_outstanding_loans")
        )

    def handle_insolvency(self, credit_market: CreditMarket) -> float:
        for bank_id in np.where(self.states["is_insolvent"])[0]:
            credit_market.remove_loans_by_bank(bank_id)
        return self.functions["demography"].handle_bank_insolvency(
            current_bank_equity=self.ts.current("equity"),
            current_bank_deposits=self.ts.current("deposits"),
            is_insolvent=self.states["is_insolvent"],
        )

    def compute_insolvency_rate(self) -> float:
        insolvency_rate = self.states["is_insolvent"].mean()
        self.states["is_insolvent"] = np.full(self.ts.current("n_banks"), False)
        return insolvency_rate
