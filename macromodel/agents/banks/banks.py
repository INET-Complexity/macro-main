from typing import Any

import h5py
import numpy as np

from macro_data import SyntheticBanks
from macromodel.agents.agent import Agent
from macromodel.agents.banks.banks_ts import create_banks_timeseries
from macromodel.configurations import BankParameters, BanksConfiguration
from macromodel.markets.credit_market.credit_market import CreditMarket
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import functions_from_model, update_functions


class Banks(Agent):
    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        functions: dict[str, Any],
        parameters: BankParameters,
        policy_rate_markup: float,
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

        self.parameters: BankParameters = parameters
        self.functions: dict[str, Any] = functions
        self.policy_rate_markup: float = policy_rate_markup

    @classmethod
    def from_pickled_agent(
        cls,
        synthetic_banks: SyntheticBanks,
        configuration: BanksConfiguration,
        policy_rate_markup: float,
        n_industries: int,
        scale: int,
        country_name: str,
        all_country_names: list[str],
    ):
        corr_firms_id = synthetic_banks.bank_data["Corresponding Firms ID"]
        corr_households_id = synthetic_banks.bank_data["Corresponding Households ID"]
        parameters = configuration.parameters
        functions = functions_from_model(model=configuration.functions, loc="macromodel.banks")

        data = synthetic_banks.bank_data.drop(columns=["Corresponding Firms ID", "Corresponding Households ID"])
        ts = create_banks_timeseries(
            bank_data=data,
            scale=scale,
        )

        states: dict[str, float | np.ndarray | list[np.ndarray]] = {
            "corr_firms": [corr_firms_id.values[i][0] for i in range(corr_firms_id.shape[0])],
            "corr_households": [corr_households_id.values[i][0] for i in range(corr_households_id.shape[0])],
            "is_insolvent": np.full(ts.current("n_banks"), False),
            "Firm Pass Through": synthetic_banks.firm_passthrough,
            "Firm ECT": synthetic_banks.firm_ect,
            "Household Consumption Pass Through": synthetic_banks.hh_consumption_passthrough,
            "Household Consumption ECT": synthetic_banks.hh_consumption_ect,
            "Household Mortgage Pass Through": synthetic_banks.hh_mortgage_passthrough,
            "Household Mortgage ECT": synthetic_banks.hh_mortgage_ect,
        }

        return cls(
            country_name,
            all_country_names,
            n_industries,
            functions,
            parameters,
            policy_rate_markup,
            ts,
            states,
        )

    def reset(self, configuration: BanksConfiguration) -> None:
        self.gen_reset()
        self.parameters = configuration.parameters
        update_functions(model=configuration.functions, loc="macromodel.banks", functions=self.functions)

    def compute_estimated_profits(self, estimated_growth: float, estimated_inflation: float) -> np.ndarray:
        return self.functions["profit_estimator"].compute_estimated_profits(
            current_profits=self.ts.current("profits"),
            estimated_growth=estimated_growth,
            estimated_inflation=estimated_inflation,
        )

    def set_interest_rates(self, central_bank_policy_rate: float) -> None:
        # On loans
        self.ts.interest_rates_on_short_term_firm_loans.append(
            self.functions["interest_rates"].get_interest_rates_on_short_term_firm_loans(
                central_bank_policy_rate=central_bank_policy_rate,
                prev_interest_rates_on_short_term_firm_loans=self.ts.current("interest_rates_on_short_term_firm_loans"),
                firm_pt=self.states["Firm Pass Through"],
                firm_ect=self.states["Firm ECT"],
            )
        )
        self.ts.average_interest_rates_on_short_term_firm_loans.append(
            [self.ts.current("interest_rates_on_short_term_firm_loans").mean()]
        )
        self.ts.interest_rates_on_long_term_firm_loans.append(
            self.functions["interest_rates"].get_interest_rates_on_long_term_firm_loans(
                central_bank_policy_rate=central_bank_policy_rate,
                prev_interest_rates_on_long_term_firm_loans=self.ts.current("interest_rates_on_long_term_firm_loans"),
                firm_pt=self.states["Firm Pass Through"],
                firm_ect=self.states["Firm ECT"],
            )
        )
        self.ts.average_interest_rates_on_long_term_firm_loans.append(
            [self.ts.current("interest_rates_on_long_term_firm_loans").mean()]
        )
        self.ts.interest_rates_on_household_consumption_loans.append(
            self.functions["interest_rates"].get_interest_rates_on_household_consumption_loans(
                central_bank_policy_rate=central_bank_policy_rate,
                prev_interest_rate_on_hh_consumption_loans=self.ts.current(
                    "interest_rates_on_household_consumption_loans"
                ),
                hh_cons_pt=self.states["Household Consumption Pass Through"],
                hh_cons_ect=self.states["Household Consumption ECT"],
            )
        )
        self.ts.average_interest_rates_on_household_consumption_loans.append(
            [self.ts.current("interest_rates_on_household_consumption_loans").mean()]
        )
        self.ts.interest_rates_on_mortgages.append(
            self.functions["interest_rates"].get_interest_rate_on_mortgages(
                central_bank_policy_rate=central_bank_policy_rate,
                prev_interest_rate_on_mortgages=self.ts.current("interest_rates_on_mortgages"),
                hh_mortgage_pt=self.states["Household Mortgage Pass Through"],
                hh_mortgage_ect=self.states["Household Mortgage ECT"],
            )
        )
        self.ts.average_interest_rates_on_mortgages.append([self.ts.current("interest_rates_on_mortgages").mean()])

        # On deposits
        self.ts.interest_rate_on_firm_deposits.append(
            self.functions["interest_rates"].compute_interest_rate_on_firm_deposits(
                central_bank_policy_rate=central_bank_policy_rate,
                prev_interest_rate_on_firm_deposits=self.ts.current("interest_rate_on_firm_deposits"),
                firm_pt=self.states["Firm Pass Through"],
                firm_ect=self.states["Firm ECT"],
            )
        )
        self.ts.average_interest_rate_on_firm_deposits.append(
            [self.ts.current("interest_rate_on_firm_deposits").mean()]
        )
        self.ts.overdraft_rate_on_firm_deposits.append(
            self.functions["interest_rates"].compute_overdraft_rate_on_firm_deposits(
                central_bank_policy_rate=central_bank_policy_rate,
                prev_overdraft_rate_on_firm_deposits=self.ts.current("overdraft_rate_on_firm_deposits"),
                firm_pt=self.states["Firm Pass Through"],
                firm_ect=self.states["Firm ECT"],
            )
        )
        self.ts.average_overdraft_rate_on_firm_deposits.append(
            [self.ts.current("overdraft_rate_on_firm_deposits").mean()]
        )
        self.ts.interest_rate_on_household_deposits.append(
            self.functions["interest_rates"].compute_interest_rate_on_household_deposits(
                central_bank_policy_rate=central_bank_policy_rate,
                prev_interest_rate_on_hh_deposits=self.ts.current("interest_rate_on_household_deposits"),
                hh_cons_pt=self.states["Household Consumption Pass Through"],
                hh_cons_ect=self.states["Household Consumption ECT"],
            )
        )
        self.ts.average_interest_rate_on_household_deposits.append(
            [self.ts.current("interest_rate_on_household_deposits").mean()]
        )
        self.ts.overdraft_rate_on_household_deposits.append(
            self.functions["interest_rates"].compute_overdraft_rate_on_household_deposits(
                central_bank_policy_rate=central_bank_policy_rate,
                prev_overdraft_rate_on_hh_deposits=self.ts.current("overdraft_rate_on_household_deposits"),
                hh_cons_pt=self.states["Household Consumption Pass Through"],
                hh_cons_ect=self.states["Household Consumption ECT"],
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
        current_deposits_from_firms = np.bincount(
            firm_corresponding_bank,
            weights=current_firm_deposits,
            minlength=self.ts.current("n_banks"),
        )
        current_deposits_from_households = np.bincount(
            households_corresponding_bank,
            weights=current_household_deposits,
            minlength=self.ts.current("n_banks"),
        )
        self.ts.deposits_from_firms.append(current_deposits_from_firms)
        self.ts.total_deposits_from_firms.append([current_deposits_from_firms.sum()])
        self.ts.deposits_from_households.append(current_deposits_from_households)
        self.ts.total_deposits_from_households.append([current_deposits_from_households.sum()])

    def update_loans(self, credit_market: CreditMarket) -> None:
        self.ts.short_term_loans_to_firms.append(credit_market.compute_outstanding_short_term_firm_loans_by_bank())
        self.ts.total_short_term_loans_to_firms.append([self.ts.current("short_term_loans_to_firms").sum()])
        self.ts.long_term_loans_to_firms.append(credit_market.compute_outstanding_long_term_firm_loans_by_bank())
        self.ts.total_long_term_loans_to_firms.append([self.ts.current("long_term_loans_to_firms").sum()])
        self.ts.consumption_loans_to_households.append(
            credit_market.compute_outstanding_household_consumption_loans_by_bank()
        )
        self.ts.total_consumption_loans_to_households.append([self.ts.current("consumption_loans_to_households").sum()])
        self.ts.mortgages_to_households.append(credit_market.compute_outstanding_mortgages_by_bank())
        self.ts.total_mortgages_to_households.append([self.ts.current("mortgages_to_households").sum()])
        self.ts.total_outstanding_loans.append(credit_market.compute_outstanding_loans_by_bank())

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
        equity_injection, average_equity = self.functions["demography"].handle_bank_insolvency(
            current_bank_equity=self.ts.current("equity"),
            current_bank_loans=self.ts.current("total_outstanding_loans"),
            current_bank_deposits=self.ts.current("deposits"),
            is_insolvent=self.states["is_insolvent"],
        )

        # Remove loans
        for bank_id in np.where(self.states["is_insolvent"])[0]:
            credit_market.remove_loans_by_bank(bank_id)

        # Update deposits
        new_firm_deposits = self.ts.current("deposits")
        new_firm_deposits[self.states["is_insolvent"]] = 0.0
        self.ts.deposits.pop()
        self.ts.deposits.append(new_firm_deposits)

        # Update equity
        new_firm_equity = self.ts.current("equity")
        new_firm_equity[self.states["is_insolvent"]] = average_equity
        self.ts.equity.pop()
        self.ts.equity.append(new_firm_equity)

        return equity_injection

    def compute_insolvency_rate(self) -> float:
        insolvency_rate = self.states["is_insolvent"].mean()
        self.states["is_insolvent"] = np.full(self.ts.current("n_banks"), False)
        return insolvency_rate

    def save_to_h5(self, group: h5py.Group):
        self.ts.write_to_h5("banks", group)
