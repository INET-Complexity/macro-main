import numpy as np

from abc import abstractmethod, ABC


class InterestRatesSetter(ABC):
    @abstractmethod
    def get_interest_rates_on_short_term_firm_loans(
        self,
        central_bank_policy_rate: float,
        prev_interest_rates_on_short_term_firm_loans: np.ndarray,
        firm_pt: float,
        firm_ect: float,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def get_interest_rates_on_long_term_firm_loans(
        self,
        central_bank_policy_rate: float,
        prev_interest_rates_on_long_term_firm_loans: np.ndarray,
        firm_pt: float,
        firm_ect: float,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def get_interest_rates_on_household_consumption_loans(
        self,
        central_bank_policy_rate: float,
        prev_interest_rate_on_hh_consumption_loans: np.ndarray,
        hh_cons_pt: float,
        hh_cons_ect: float,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def get_interest_rate_on_mortgages(
        self,
        central_bank_policy_rate: float,
        prev_interest_rate_on_mortgages: np.ndarray,
        hh_mortgage_pt: float,
        hh_mortgage_ect: float,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_interest_rate_on_firm_deposits(
        self,
        central_bank_policy_rate: float,
        prev_interest_rates_on_short_term_firm_loans: np.ndarray,
        firm_pt: float,
        firm_ect: float,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_overdraft_rate_on_firm_deposits(
        self,
        central_bank_policy_rate: float,
        prev_interest_rates_on_short_term_firm_loans: np.ndarray,
        firm_pt: float,
        firm_ect: float,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_interest_rate_on_household_deposits(
        self,
        central_bank_policy_rate: float,
        prev_interest_rate_on_hh_ce_loans: np.ndarray,
        hh_cons_pt: float,
        hh_cons_ect: float,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_overdraft_rate_on_household_deposits(
        self,
        central_bank_policy_rate: float,
        prev_interest_rate_on_hh_ce_loans: np.ndarray,
        hh_cons_pt: float,
        hh_cons_ect: float,
    ) -> np.ndarray:
        pass


class DefaultInterestRatesSetter(InterestRatesSetter):
    def get_interest_rates_on_short_term_firm_loans(
        self,
        central_bank_policy_rate: float,
        prev_interest_rates_on_short_term_firm_loans: np.ndarray,
        firm_pt: float,
        firm_ect: float,
    ) -> np.ndarray:
        return prev_interest_rates_on_short_term_firm_loans + firm_ect * (
            prev_interest_rates_on_short_term_firm_loans - firm_pt * central_bank_policy_rate
        )

    def get_interest_rates_on_long_term_firm_loans(
        self,
        central_bank_policy_rate: float,
        prev_interest_rates_on_long_term_firm_loans: np.ndarray,
        firm_pt: float,
        firm_ect: float,
    ) -> np.ndarray:
        return prev_interest_rates_on_long_term_firm_loans + firm_ect * (
            prev_interest_rates_on_long_term_firm_loans - firm_pt * central_bank_policy_rate
        )

    def get_interest_rates_on_household_consumption_loans(
        self,
        central_bank_policy_rate: float,
        prev_interest_rate_on_hh_consumption_loans: np.ndarray,
        hh_cons_pt: float,
        hh_cons_ect: float,
    ) -> np.ndarray:
        return prev_interest_rate_on_hh_consumption_loans + hh_cons_ect * (
            prev_interest_rate_on_hh_consumption_loans - hh_cons_pt * central_bank_policy_rate
        )

    def get_interest_rate_on_mortgages(
        self,
        central_bank_policy_rate: float,
        prev_interest_rate_on_mortgages: np.ndarray,
        hh_mortgage_pt: float,
        hh_mortgage_ect: float,
    ) -> np.ndarray:
        return prev_interest_rate_on_mortgages + hh_mortgage_ect * (
            prev_interest_rate_on_mortgages - hh_mortgage_pt * central_bank_policy_rate
        )

    def compute_interest_rate_on_firm_deposits(
        self,
        central_bank_policy_rate: float,
        prev_interest_rate_on_firm_deposits: np.ndarray,
        firm_pt: float,
        firm_ect: float,
    ) -> np.ndarray:
        return np.full(prev_interest_rate_on_firm_deposits.shape, central_bank_policy_rate)

    def compute_overdraft_rate_on_firm_deposits(
        self,
        central_bank_policy_rate: float,
        prev_overdraft_rate_on_firm_deposits: np.ndarray,
        firm_pt: float,
        firm_ect: float,
    ) -> np.ndarray:
        return prev_overdraft_rate_on_firm_deposits + firm_ect * (
            prev_overdraft_rate_on_firm_deposits - firm_pt * central_bank_policy_rate
        )

    def compute_interest_rate_on_household_deposits(
        self,
        central_bank_policy_rate: float,
        prev_interest_rate_on_hh_deposits: np.ndarray,
        hh_cons_pt: float,
        hh_cons_ect: float,
    ) -> np.ndarray:
        return np.full(prev_interest_rate_on_hh_deposits.shape, central_bank_policy_rate)

    def compute_overdraft_rate_on_household_deposits(
        self,
        central_bank_policy_rate: float,
        prev_overdraft_rate_on_hh_deposits: np.ndarray,
        hh_cons_pt: float,
        hh_cons_ect: float,
    ) -> np.ndarray:
        return prev_overdraft_rate_on_hh_deposits + hh_cons_ect * (
            prev_overdraft_rate_on_hh_deposits - hh_cons_pt * central_bank_policy_rate
        )
