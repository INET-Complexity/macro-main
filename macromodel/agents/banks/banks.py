"""Banking system implementation for macroeconomic modeling.

This module implements the banking sector, which serves as a financial
intermediary in the economy through:
- Deposit taking from firms and households
- Loan provision to firms and households
- Interest rate setting and adjustment
- Credit market participation
- Financial stability monitoring

The banks operate by:
- Managing deposits and loans
- Setting interest rates based on policy rates
- Computing profits and equity
- Handling insolvency cases
- Tracking market shares
"""

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
    """Banking system that intermediates between savers and borrowers.

    This class represents the banking sector that facilitates financial
    intermediation through:
    - Deposit management (firms and households)
    - Loan provision (short/long-term, consumption, mortgages)
    - Interest rate setting (loans and deposits)
    - Profit generation and equity accumulation
    - Market share tracking
    - Insolvency handling

    The banks operate with:
    - Policy rate-based interest rate adjustment
    - Differentiated rates for different products
    - Balance sheet management
    - Profit computation and distribution
    - Financial stability monitoring

    Attributes:
        parameters (BankParameters): Banking sector parameters
        functions (dict[str, Any]): Function implementations
        policy_rate_markup (float): Markup over policy rate
        ts (TimeSeries): Time series tracking bank variables
        states (dict): State variables including:
            - corr_firms: Corresponding firm IDs
            - corr_households: Corresponding household IDs
            - is_insolvent: Bank solvency status
            - Pass Through/ECT parameters for different products
    """

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
        """Initialize banking system.

        Args:
            country_name (str): Name of the country
            all_country_names (list[str]): List of all countries
            n_industries (int): Number of industries
            functions (dict[str, Any]): Function implementations
            parameters (BankParameters): Banking sector parameters
            policy_rate_markup (float): Markup over policy rate
            ts (TimeSeries): Time series for tracking variables
            states (dict): State variables and parameters
        """
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
        """Create banking system from pickled data.

        Initializes banks with:
        - Configuration parameters
        - Synthetic bank data
        - Interest rate parameters
        - Corresponding firm/household mappings

        Args:
            synthetic_banks (SyntheticBanks): Synthetic bank data
            configuration (BanksConfiguration): Bank configuration
            policy_rate_markup (float): Markup over policy rate
            n_industries (int): Number of industries
            scale (int): Scale factor for histograms
            country_name (str): Name of the country
            all_country_names (list[str]): List of all countries

        Returns:
            Banks: Initialized banking system
        """
        corr_firms_id = synthetic_banks.bank_data["Corresponding Firms ID"]
        corr_households_id = synthetic_banks.bank_data["Corresponding Households ID"]
        parameters = configuration.parameters
        functions = functions_from_model(model=configuration.functions, loc="macromodel.agents.banks")

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
        """Reset banking system to initial state.

        Resets all state variables and updates function implementations
        based on the provided configuration.

        Args:
            configuration (BanksConfiguration): New configuration
        """
        self.gen_reset()
        self.parameters = configuration.parameters
        update_functions(model=configuration.functions, loc="macromodel.agents.banks", functions=self.functions)

    def compute_estimated_profits(self, estimated_growth: float, estimated_inflation: float) -> np.ndarray:
        """Calculate estimated future profits.

        Estimates profits based on:
        - Current profit levels
        - Expected economic growth
        - Expected inflation

        Args:
            estimated_growth (float): Expected economic growth rate
            estimated_inflation (float): Expected inflation rate

        Returns:
            np.ndarray: Estimated profits by bank
        """
        return self.functions["profit_estimator"].compute_estimated_profits(
            current_profits=self.ts.current("profits"),
            estimated_growth=estimated_growth,
            estimated_inflation=estimated_inflation,
        )

    def set_interest_rates(self, central_bank_policy_rate: float) -> None:
        """Set interest rates for all banking products.

        Updates rates for:
        - Short-term firm loans
        - Long-term firm loans
        - Household consumption loans
        - Mortgages
        - Firm deposits
        - Household deposits
        - Overdraft facilities

        Each rate considers:
        - Central bank policy rate
        - Previous rate levels
        - Pass-through parameters
        - Error correction terms

        Args:
            central_bank_policy_rate (float): Current policy rate
        """
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
        """Calculate net interest received on deposits.

        Computes interest considering:
        - Positive and negative deposit balances
        - Different rates for firms and households
        - Overdraft rates for negative balances
        - Regular deposit rates for positive balances

        Args:
            central_bank_policy_rate (float): Current policy rate

        Returns:
            np.ndarray: Net interest received by bank
        """
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
        """Calculate total bank profits.

        Combines:
        - Interest received on loans
        - Net interest received on deposits

        Returns:
            np.ndarray: Total profits by bank
        """
        return self.ts.current("interest_received_on_loans") + self.ts.current("interest_received_on_deposits")

    def update_deposits(
        self,
        current_firm_deposits: np.ndarray,
        current_household_deposits: np.ndarray,
        firm_corresponding_bank: np.ndarray,
        households_corresponding_bank: np.ndarray,
    ) -> None:
        """Update deposit balances.

        Records:
        - Firm deposits by bank
        - Household deposits by bank
        - Total deposits from firms
        - Total deposits from households

        Args:
            current_firm_deposits (np.ndarray): Current firm deposits
            current_household_deposits (np.ndarray): Current household deposits
            firm_corresponding_bank (np.ndarray): Bank IDs for firms
            households_corresponding_bank (np.ndarray): Bank IDs for households
        """
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
        """Update loan balances.

        Records:
        - Short-term firm loans
        - Long-term firm loans
        - Household consumption loans
        - Mortgages
        - Total outstanding loans

        Args:
            credit_market (CreditMarket): Credit market instance
        """
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
        """Calculate market share of each bank.

        Based on:
        - Total outstanding loans
        - Total deposits from firms
        - Total deposits from households

        Returns:
            np.ndarray: Market share by bank
        """
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
        """Calculate bank equity.

        Considers:
        - Current equity levels
        - After-tax profits
        - Insolvency status

        Args:
            profit_taxes (float): Tax rate on profits

        Returns:
            np.ndarray: Equity by bank
        """
        return (
            self.ts.current("equity")
            + self.ts.current("profits")
            - profit_taxes * np.maximum(0.0, self.ts.current("profits"))
        )

    def compute_liability(self) -> np.ndarray:
        """Calculate total bank liabilities.

        Sums:
        - Deposits from firms
        - Deposits from households

        Returns:
            np.ndarray: Total liabilities by bank
        """
        return (
            self.ts.current("equity")
            + np.maximum(0, self.ts.current("deposits_from_firms"))
            + np.maximum(0, self.ts.current("deposits_from_households"))
            + np.maximum(0, -self.ts.current("deposits"))
        )

    def compute_deposits(self) -> np.ndarray:
        """Calculate total deposits.

        Sums:
        - Deposits from firms
        - Deposits from households

        Returns:
            np.ndarray: Total deposits by bank
        """
        return (
            self.ts.current("deposits_from_firms")
            + self.ts.current("deposits_from_households")
            + self.ts.current("equity")
            - self.ts.current("total_outstanding_loans")
        )

    def handle_insolvency(self, credit_market: CreditMarket) -> float:
        """Handle insolvent banks.

        Processes:
        - Identification of insolvent banks
        - Marking of insolvent status
        - Credit market adjustments
        - Loss computation

        Args:
            credit_market (CreditMarket): Credit market instance

        Returns:
            float: Total losses from insolvency
        """
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
        """Calculate bank insolvency rate.

        Returns:
            float: Fraction of banks that are insolvent
        """
        insolvency_rate = self.states["is_insolvent"].mean()
        self.states["is_insolvent"] = np.full(self.ts.current("n_banks"), False)
        return insolvency_rate

    def save_to_h5(self, group: h5py.Group):
        """Save bank data to HDF5.

        Stores all time series data in the specified HDF5 group.

        Args:
            group (h5py.Group): HDF5 group to save data in
        """
        self.ts.write_to_h5("banks", group)
