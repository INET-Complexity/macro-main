"""Credit market implementation for macroeconomic agent-based model.

This module implements a sophisticated credit market system that manages lending relationships
between banks, firms, and households. It handles multiple types of loans including:

1. Firm Loans:
   - Short-term loans: Working capital, operational expenses
   - Long-term loans: Capital investment, expansion

2. Household Loans:
   - Consumption loans: Personal loans, credit lines
   - Mortgage loans: Home purchases, refinancing
   - Payday loans: Short-term emergency credit

The market clearing mechanism considers:
- Bank lending capacity and risk appetite
- Borrower creditworthiness and collateral
- Interest rate determination
- Non-performing loan (NPL) dynamics
- Regulatory constraints

Key Features:
- Multi-agent lending relationships
- Dynamic interest rate adjustment
- Risk-based credit allocation
- Loan lifecycle management
- Default handling
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Tuple

import h5py
import numpy as np

from macro_data import SyntheticCreditMarket
from macromodel.configurations import CreditMarketConfiguration
from macromodel.markets.credit_market.credit_market_ts import (
    create_credit_market_timeseries,
)
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import functions_from_model, update_functions

if TYPE_CHECKING:
    from macromodel.agents.banks.banks import Banks
    from macromodel.agents.firms import Firms
    from macromodel.agents.households.households import Households


class CreditMarket:
    """Credit market implementation managing lending relationships and loan lifecycles.

    This class implements the core credit market functionality, managing the interactions
    between financial institutions (banks) and borrowers (firms and households). It handles
    loan origination, servicing, repayment, and default processes.

    The market maintains state information about all outstanding loans including:
    - Principal amounts
    - Interest rates
    - Payment schedules
    - Default status

    Loan types are tracked in separate arrays with dimensions [3, n_banks, n_borrowers]:
    - Index 0: Outstanding principal
    - Index 1: Interest rate
    - Index 2: Monthly payment amount

    Attributes:
        country_name (str): Name of the country this market operates in
        functions (dict[str, Any]): Market functions (clearing, pricing, etc.)
        ts (TimeSeries): Time series tracking market metrics
        states (dict[str, np.ndarray]): Current state of all loans
        initial_states (dict[str, np.ndarray]): Initial state snapshot for resets

    Example:
        >>> market = CreditMarket.from_data(
        ...     country_name="USA",
        ...     st_loans=short_term_loan_data,
        ...     lt_loans=long_term_loan_data,
        ...     cons_loans=consumer_loan_data,
        ...     mort_loans=mortgage_loan_data
        ... )
        >>> market.clear(banks, firms, households, npl_firm=0.02, npl_cons=0.03, npl_mort=0.01)
    """

    def __init__(
        self,
        country_name: str,
        functions: dict[str, Any],
        ts: TimeSeries,
        states: dict[str, np.ndarray],
        initial_states: dict[str, np.ndarray],
    ):
        """Initialize a new credit market instance.

        Args:
            country_name (str): Name of the country this market operates in
            functions (dict[str, Any]): Dictionary of market functions (clearing, etc.)
            ts (TimeSeries): Time series object for tracking market metrics
            states (dict[str, np.ndarray]): Current state of all loans
            initial_states (dict[str, np.ndarray]): Initial state snapshot for resets
        """
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
        """Create a credit market instance from a pickled synthetic market.

        This factory method initializes a credit market from preprocessed synthetic data,
        which includes historical loan data and market configuration parameters.

        Args:
            synthetic_credit_market (SyntheticCreditMarket): Preprocessed market data
            credit_market_configuration (CreditMarketConfiguration): Market parameters
            country_name (str): Name of the country this market operates in

        Returns:
            CreditMarket: Initialized credit market instance with historical data

        Note:
            The synthetic market data includes:
            - Historical loan volumes by type
            - Default rates and loss history
            - Interest rate patterns
            - Bank-borrower relationships
        """
        functions = functions_from_model(
            credit_market_configuration.functions,
            loc="macromodel.markets.credit_market",
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
        """Reset the credit market to its initial state.

        Restores all loan states to their initial values and updates market functions
        with the new configuration. This is useful for running multiple simulations
        or testing different scenarios.

        Args:
            configuration (CreditMarketConfiguration): New market configuration to use
        """
        self.states = deepcopy(self.initial_states)
        self.ts.reset()
        update_functions(model=configuration.functions, loc="macromodel.agents.credit_market", functions=self.functions)

    @classmethod
    def from_data(
        cls,
        country_name: str,
        st_loans: np.ndarray,
        lt_loans: np.ndarray,
        cons_loans: np.ndarray,
        mort_loans: np.ndarray,
    ) -> "CreditMarket":
        """Create a credit market instance directly from loan data arrays.

        This factory method provides a simpler way to initialize a credit market
        when you have direct access to loan data arrays rather than a synthetic market.

        Args:
            country_name (str): Name of the country this market operates in
            st_loans (np.ndarray): Short-term loan data [3, n_banks, n_firms]
            lt_loans (np.ndarray): Long-term loan data [3, n_banks, n_firms]
            cons_loans (np.ndarray): Consumer loan data [3, n_banks, n_households]
            mort_loans (np.ndarray): Mortgage loan data [3, n_banks, n_households]

        Returns:
            CreditMarket: Initialized credit market instance

        Note:
            Each loan array has shape [3, n_banks, n_borrowers] where:
            - Index 0: Outstanding principal
            - Index 1: Interest rate
            - Index 2: Monthly payment amount
        """
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
        """Clear the credit market by matching loan supply with demand.

        This is the core market clearing function that:
        1. Evaluates new loan applications
        2. Determines credit allocation
        3. Updates loan states
        4. Records lending activity

        The clearing process considers:
        - Bank lending capacity and risk appetite
        - Borrower creditworthiness
        - Current NPL rates
        - Market conditions

        Args:
            banks (Banks): Banking sector agent
            firms (Firms): Corporate sector agents
            households (Households): Household sector agents
            current_npl_firm_loans (float): Current NPL rate for firm loans
            current_npl_hh_cons_loans (float): Current NPL rate for consumer loans
            current_npl_mortgages (float): Current NPL rate for mortgages

        Note:
            The function updates various time series metrics including:
            - New loan originations by type
            - Outstanding loan balances
            - Bank portfolio composition
        """
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
        """Process firm loan payments for the current period.

        Processes monthly payments for both short-term and long-term firm loans.
        The payment amount is the minimum of the scheduled payment and outstanding balance.

        Returns:
            np.ndarray: Array of total payments made by each firm
        """
        di_st = np.minimum(self.states["st_loans"][0], self.states["st_loans"][2])
        di_lt = np.minimum(self.states["lt_loans"][0], self.states["lt_loans"][2])
        self.states["st_loans"][0] -= di_st
        self.states["lt_loans"][0] -= di_lt
        return di_st.sum(axis=0) + di_lt.sum(axis=0)

    def pay_household_installments(self) -> np.ndarray:
        """Process household loan payments for the current period.

        Processes monthly payments for both consumer loans and mortgages.
        The payment amount is the minimum of the scheduled payment and outstanding balance.

        Returns:
            np.ndarray: Array of total payments made by each household
        """
        di_cons = np.minimum(self.states["cons_loans"][0], self.states["cons_loans"][2])
        di_mort = np.minimum(self.states["mort_loans"][0], self.states["mort_loans"][2])
        self.states["cons_loans"][0] -= di_cons
        self.states["mort_loans"][0] -= di_mort
        return di_cons.sum(axis=0) + di_mort.sum(axis=0)

    def remove_repaid_loans(self) -> None:
        """Clean up fully repaid loans from the market state.

        Identifies loans with near-zero balances (accounting for numerical precision)
        and removes them from the market state by zeroing out all their attributes.
        """
        for loans in [
            self.states["st_loans"],
            self.states["lt_loans"],
            self.states["cons_loans"],
            self.states["mort_loans"],
        ]:
            ind = np.isclose(loans[0], 0.0, atol=1e-2)
            loans[:, ind] = 0.0

    def compute_aggregates(self) -> None:
        """Update aggregate loan statistics.

        Calculates and records total outstanding loan amounts by type:
        - Short-term firm loans
        - Long-term firm loans
        - Consumer loans
        - Mortgages
        """
        self.ts.total_outstanding_loans_granted_firms_short_term.append([self.states["st_loans"][0].sum()])
        self.ts.total_outstanding_loans_granted_firms_long_term.append([self.states["lt_loans"][0].sum()])
        self.ts.total_outstanding_loans_granted_households_consumption.append([self.states["cons_loans"][0].sum()])
        self.ts.total_outstanding_loans_granted_mortgages.append([self.states["mort_loans"][0].sum()])

    def compute_outstanding_short_term_loans_by_firm(self) -> np.ndarray:
        """Calculate total short-term loans for each firm.

        Returns:
            np.ndarray: Array of total short-term loan balances by firm
        """
        return self.states["st_loans"][0].sum(axis=0)

    def compute_outstanding_long_term_loans_by_firm(self) -> np.ndarray:
        """Calculate total long-term loans for each firm.

        Returns:
            np.ndarray: Array of total long-term loan balances by firm
        """
        return self.states["lt_loans"][0].sum(axis=0)

    def compute_outstanding_consumption_loans_by_household(self) -> np.ndarray:
        """Calculate total consumer loans for each household.

        Returns:
            np.ndarray: Array of total consumer loan balances by household
        """
        return self.states["cons_loans"][0].sum(axis=0)

    def compute_outstanding_mortgages_by_household(self) -> np.ndarray:
        """Calculate total mortgage loans for each household.

        Returns:
            np.ndarray: Array of total mortgage balances by household
        """
        return self.states["mort_loans"][0].sum(axis=0)

    def compute_outstanding_loans_by_bank(self) -> np.ndarray:
        """Calculate total loans outstanding for each bank.

        Returns:
            np.ndarray: Array of total loan balances by bank across all loan types
        """
        return (
            self.states["st_loans"][0].sum(axis=1)
            + self.states["lt_loans"][0].sum(axis=1)
            + self.states["cons_loans"][0].sum(axis=1)
            + self.states["mort_loans"][0].sum(axis=1)
        )

    def compute_outstanding_short_term_firm_loans_by_bank(self) -> np.ndarray:
        """Calculate total short-term firm loans for each bank.

        Returns:
            np.ndarray: Array of short-term firm loan balances by bank
        """
        return self.states["st_loans"][0].sum(axis=1)

    def compute_outstanding_long_term_firm_loans_by_bank(self) -> np.ndarray:
        """Calculate total long-term firm loans for each bank.

        Returns:
            np.ndarray: Array of long-term firm loan balances by bank
        """
        return self.states["lt_loans"][0].sum(axis=1)

    def compute_outstanding_household_consumption_loans_by_bank(self) -> np.ndarray:
        """Calculate total consumer loans for each bank.

        Returns:
            np.ndarray: Array of consumer loan balances by bank
        """
        return self.states["cons_loans"][0].sum(axis=1)

    def compute_outstanding_mortgages_by_bank(self) -> np.ndarray:
        """Calculate total mortgage loans for each bank.

        Returns:
            np.ndarray: Array of mortgage balances by bank
        """
        return self.states["mort_loans"][0].sum(axis=1)

    def compute_interest_paid_by_firm(self) -> np.ndarray:
        """Calculate total interest paid by each firm.

        Returns:
            np.ndarray: Array of interest payments by firm across all loan types
        """
        return self.states["st_loans"][1].sum(axis=0) + self.states["lt_loans"][1].sum(axis=0)

    def compute_interest_paid_by_household(self) -> np.ndarray:
        """Calculate total interest paid by each household.

        Returns:
            np.ndarray: Array of interest payments by household across all loan types
        """
        return self.states["cons_loans"][1].sum(axis=0) + self.states["mort_loans"][1].sum(axis=0)

    def compute_interest_received_by_bank(self) -> np.ndarray:
        """Calculate total interest received by each bank.

        Returns:
            np.ndarray: Array of interest income by bank across all loan types
        """
        return (
            self.states["st_loans"][1].sum(axis=1)
            + self.states["lt_loans"][1].sum(axis=1)
            + self.states["cons_loans"][1].sum(axis=1)
            + self.states["mort_loans"][1].sum(axis=1)
        )

    def remove_loans_to_firm(self, firm_id: int | np.ndarray) -> float:
        """Remove all loans associated with specified firm(s).

        Used when firms default or exit the market. Returns the total amount written off.

        Args:
            firm_id (int | np.ndarray): ID(s) of firm(s) to remove loans for

        Returns:
            float: Total amount of loans written off
        """
        total_amount = self.states["st_loans"][0][:, firm_id].sum() + self.states["lt_loans"][0][:, firm_id].sum()
        self.states["st_loans"][:, :, firm_id] = 0.0
        self.states["lt_loans"][:, :, firm_id] = 0.0
        return total_amount

    def remove_loans_to_households(self, household_id: int | np.ndarray) -> Tuple[float, float]:
        """Remove all loans associated with specified household(s).

        Used when households default. Returns the total amounts written off by loan type.

        Args:
            household_id (int | np.ndarray): ID(s) of household(s) to remove loans for

        Returns:
            Tuple[float, float]: Total consumer loans and mortgages written off
        """
        cons_amount = self.states["cons_loans"][0][:, household_id].sum()
        mort_amount = self.states["mort_loans"][0][:, household_id].sum()
        self.states["cons_loans"][:, :, household_id] = 0.0
        self.states["mort_loans"][:, :, household_id] = 0.0
        return cons_amount, mort_amount

    def remove_loans_by_bank(self, bank_id: int | np.ndarray) -> None:
        """Remove all loans associated with specified bank(s).

        Used when banks fail or exit the market.

        Args:
            bank_id (int | np.ndarray): ID(s) of bank(s) to remove loans for
        """
        self.states["st_loans"][:, bank_id] = 0.0
        self.states["lt_loans"][:, bank_id] = 0.0
        self.states["cons_loans"][:, bank_id] = 0.0
        self.states["mort_loans"][:, bank_id] = 0.0

    def save_to_h5(self, group: h5py.Group):
        """Save credit market state to HDF5 file.

        Args:
            group (h5py.Group): HDF5 group to save data to
        """
        self.ts.write_to_h5("CM", group)
