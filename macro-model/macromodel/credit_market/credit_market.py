from __future__ import annotations

from copy import deepcopy

import h5py
import numpy as np
from macro_data import SyntheticCreditMarket
from typing import Any, TYPE_CHECKING, Tuple

from macromodel.configurations import CreditMarketConfiguration
from macromodel.credit_market.credit_market_ts import create_credit_market_timeseries
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import functions_from_model

if TYPE_CHECKING:
    from macromodel.firms.firms import Firms
    from macromodel.banks.banks import Banks
    from macromodel.households.households import Households


class CreditMarket:
    def __init__(
        self,
        country_name: str,
        functions: dict[str, Any],
        ts: TimeSeries,
        states: dict[str, np.ndarray],
        initial_states: dict[str, np.ndarray],
    ):
        self.country_name = country_name
        self.functions = functions
        self.ts = ts
        self.states = states
        self.initial_states = initial_states

    @classmethod
    def from_pickled_market(
        cls,
        synthetic_credit_market: SyntheticCreditMarket,
        credit_market_configuration: CreditMarketConfiguration,
        country_name: str,
    ) -> "CreditMarket":
        functions = functions_from_model(
            credit_market_configuration.functions,
            loc="macromodel.credit_market",
        )

        shortterm_loans = synthetic_credit_market.shortterm_loans.stack()

        longterm_loans = synthetic_credit_market.longterm_loans.stack()

        payday_loans = synthetic_credit_market.payday_loans.stack()

        consumption_expansion_loans = synthetic_credit_market.consumption_expansion_loans.stack()

        mortgage_loans = synthetic_credit_market.mortgage_loans.stack()

        ts = create_credit_market_timeseries(
            total_consumption_expansion_loans=consumption_expansion_loans.sum(),
            total_short_term_loans=shortterm_loans.sum(),
            total_long_term_loans=longterm_loans.sum(),
            total_mortgage_loans=mortgage_loans.sum(),
        )

        states = {
            "st_loans": shortterm_loans,
            "lt_loans": longterm_loans,
            "payday_loans": payday_loans,
            "cons_loans": consumption_expansion_loans,
            "mort_loans": mortgage_loans,
        }

        initial_states = deepcopy(states)

        return cls(
            country_name,
            functions,
            ts,
            states=states,
            initial_states=initial_states,
        )

    def reset(self, configuration: CreditMarketConfiguration) -> None:
        self.states = deepcopy(self.initial_states)
        self.ts.reset()
        self.functions = functions_from_model(configuration.functions, loc="macromodel.credit_market")

    @classmethod
    def from_data(
        cls,
        country_name: str,
        st_loans: np.ndarray,
        lt_loans: np.ndarray,
        cons_loans: np.ndarray,
        mort_loans: np.ndarray,
    ) -> "CreditMarket":
        # Record the states of all loans
        states = {
            "st_loans": st_loans,
            "lt_loans": lt_loans,
            "cons_loans": cons_loans,
            "mort_loans": mort_loans,
        }

        # Create the corresponding time series object
        ts = create_credit_market_timeseries(
            total_short_term_loans=st_loans.sum(),
            total_long_term_loans=lt_loans.sum(),
            total_consumption_expansion_loans=cons_loans.sum(),
            total_mortgage_loans=mort_loans.sum(),
        )

        return cls(
            country_name=country_name,
            functions={},
            ts=ts,
            states=states,
            initial_states=deepcopy(states),
        )

    def clear(
        self,
        banks: Banks,
        firms: Firms,
        households: Households,
        current_npl_firm_loans: float,
        current_npl_hh_cons_loans: float,
        current_npl_mortgages: float,
    ) -> None:
        # Clear the credit market
        (
            new_st_loans,
            new_lt_loans,
            new_cons_loans,
            new_mort_loans,
        ) = self.functions["clearing"].clear(
            banks=banks,
            firms=firms,
            households=households,
            current_npl_firm_loans=current_npl_firm_loans,
            current_npl_hh_cons_loans=current_npl_hh_cons_loans,
            current_npl_mortgages=current_npl_mortgages,
        )

        # Record the new loans
        self.states["st_loans"] += new_st_loans
        self.states["lt_loans"] += new_lt_loans
        self.states["cons_loans"] += new_cons_loans
        self.states["mort_loans"] += new_mort_loans

        # Calculate aggregates for firms
        firms.ts.received_short_term_credit.append(new_st_loans[0].sum(axis=0))
        firms.ts.total_received_short_term_credit.append([firms.ts.current("received_short_term_credit").sum()])
        firms.ts.received_long_term_credit.append(new_lt_loans[0].sum(axis=0))
        firms.ts.total_received_long_term_credit.append([firms.ts.current("received_long_term_credit").sum()])
        firms.ts.received_credit.append(
            firms.ts.current("received_short_term_credit") + firms.ts.current("received_long_term_credit")
        )

        # Calculate aggregates for households
        households.ts.received_consumption_loans.append(new_cons_loans[0].sum(axis=0))
        households.ts.total_received_consumption_loans.append(
            [households.ts.current("received_consumption_loans").sum()]
        )
        households.ts.received_mortgages.append(new_mort_loans[0].sum(axis=0))
        households.ts.total_received_mortgages.append([households.ts.current("received_mortgages").sum()])

        # Update credit market aggregates
        self.ts.total_newly_loans_granted_firms_short_term.append(
            [firms.ts.current("received_short_term_credit").sum()]
        )
        self.ts.total_newly_loans_granted_firms_long_term.append([firms.ts.current("received_long_term_credit").sum()])
        self.ts.total_newly_loans_granted_households_consumption.append(
            [households.ts.current("received_consumption_loans").sum()]
        )
        self.ts.total_newly_loans_granted_mortgages.append([households.ts.current("received_mortgages").sum()])

        # Update fractions of types of loans granted by bank
        total_loans_by_bank = (
            self.states["st_loans"][0].sum(axis=1)
            + self.states["lt_loans"][0].sum(axis=1)
            + self.states["cons_loans"][0].sum(axis=1)
            + self.states["mort_loans"][0].sum(axis=1)
        )
        banks.ts.new_loans_fraction_firms.append(
            np.divide(
                self.states["st_loans"][0].sum(axis=1) + self.states["lt_loans"][0].sum(axis=1),
                total_loans_by_bank,
                out=np.zeros(banks.ts.current("n_banks")),
                where=total_loans_by_bank != 0.0,
            )
        )
        banks.ts.new_loans_fraction_hh_cons.append(
            np.divide(
                self.states["cons_loans"][0].sum(axis=1),
                total_loans_by_bank,
                out=np.zeros(banks.ts.current("n_banks")),
                where=total_loans_by_bank != 0.0,
            )
        )
        banks.ts.new_loans_fraction_mortgages.append(
            np.divide(
                self.states["mort_loans"][0].sum(axis=1),
                total_loans_by_bank,
                out=np.zeros(banks.ts.current("n_banks")),
                where=total_loans_by_bank != 0.0,
            )
        )

    def pay_firm_installments(self) -> np.ndarray:
        di_st = np.minimum(self.states["st_loans"][0], self.states["st_loans"][2])
        di_lt = np.minimum(self.states["lt_loans"][0], self.states["lt_loans"][2])
        self.states["st_loans"][0] -= di_st
        self.states["lt_loans"][0] -= di_lt
        return di_st.sum(axis=0) + di_lt.sum(axis=0)

    def pay_household_installments(self) -> np.ndarray:
        di_cons = np.minimum(self.states["cons_loans"][0], self.states["cons_loans"][2])
        di_mort = np.minimum(self.states["mort_loans"][0], self.states["mort_loans"][2])
        self.states["cons_loans"][0] -= di_cons
        self.states["mort_loans"][0] -= di_mort
        return di_cons.sum(axis=0) + di_mort.sum(axis=0)

    def remove_repaid_loans(self) -> None:
        for loans in [
            self.states["st_loans"],
            self.states["lt_loans"],
            self.states["cons_loans"],
            self.states["mort_loans"],
        ]:
            ind = np.isclose(loans[0], 0.0, atol=1e-2)
            loans[:, ind] = 0.0

    def compute_aggregates(self) -> None:
        self.ts.total_outstanding_loans_granted_firms_short_term.append([self.states["st_loans"][0].sum()])
        self.ts.total_outstanding_loans_granted_firms_long_term.append([self.states["lt_loans"][0].sum()])
        self.ts.total_outstanding_loans_granted_households_consumption.append([self.states["cons_loans"][0].sum()])
        self.ts.total_outstanding_loans_granted_mortgages.append([self.states["mort_loans"][0].sum()])

    def compute_outstanding_short_term_loans_by_firm(self) -> np.ndarray:
        return self.states["st_loans"][0].sum(axis=0)

    def compute_outstanding_long_term_loans_by_firm(self) -> np.ndarray:
        return self.states["lt_loans"][0].sum(axis=0)

    def compute_outstanding_consumption_loans_by_household(self) -> np.ndarray:
        return self.states["cons_loans"][0].sum(axis=0)

    def compute_outstanding_mortgages_by_household(self) -> np.ndarray:
        return self.states["mort_loans"][0].sum(axis=0)

    def compute_outstanding_loans_by_bank(self) -> np.ndarray:
        return (
            self.states["st_loans"][0].sum(axis=1)
            + self.states["lt_loans"][0].sum(axis=1)
            + self.states["cons_loans"][0].sum(axis=1)
            + self.states["mort_loans"][0].sum(axis=1)
        )

    def compute_outstanding_short_term_firm_loans_by_bank(self) -> np.ndarray:
        return self.states["st_loans"][0].sum(axis=1)

    def compute_outstanding_long_term_firm_loans_by_bank(self) -> np.ndarray:
        return self.states["lt_loans"][0].sum(axis=1)

    def compute_outstanding_household_consumption_loans_by_bank(
        self,
    ) -> np.ndarray:
        return self.states["cons_loans"][0].sum(axis=1)

    def compute_outstanding_mortgages_by_bank(self) -> np.ndarray:
        return self.states["mort_loans"][0].sum(axis=1)

    def compute_interest_paid_by_firm(self) -> np.ndarray:
        return self.states["st_loans"][1].sum(axis=0) + self.states["lt_loans"][1].sum(axis=0)

    def compute_interest_paid_by_household(self) -> np.ndarray:
        return self.states["cons_loans"][1].sum(axis=0) + self.states["mort_loans"][1].sum(axis=0)

    def compute_interest_received_by_bank(self) -> np.ndarray:
        return (
            self.states["st_loans"][1].sum(axis=1)
            + self.states["lt_loans"][1].sum(axis=1)
            + self.states["cons_loans"][1].sum(axis=1)
            + self.states["mort_loans"][1].sum(axis=1)
        )

    def remove_loans_to_firm(self, firm_id: int | np.ndarray) -> float:
        rem_loans = self.states["st_loans"][0][:, firm_id].sum() + self.states["lt_loans"][0][:, firm_id].sum()
        self.states["st_loans"][:, :, firm_id] = 0.0
        self.states["lt_loans"][:, :, firm_id] = 0.0
        return rem_loans

    def remove_loans_to_households(self, household_id: int | np.ndarray) -> Tuple[float, float]:
        rem_cons_loans = self.states["cons_loans"][0][:, household_id].sum()
        rem_mort_loans = self.states["mort_loans"][0][:, household_id].sum()
        self.states["cons_loans"][:, :, household_id] = 0.0
        self.states["mort_loans"][:, :, household_id] = 0.0
        return rem_cons_loans, rem_mort_loans

    def remove_loans_by_bank(self, bank_id: int | np.ndarray) -> None:
        self.states["st_loans"][:, bank_id] = 0.0
        self.states["lt_loans"][:, bank_id] = 0.0
        self.states["cons_loans"][:, bank_id] = 0.0
        self.states["mort_loans"][:, bank_id] = 0.0

    def save_to_h5(self, group: h5py.Group):
        self.ts.write_to_h5("credit_market", group)
