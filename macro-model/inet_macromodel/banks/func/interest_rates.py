import numpy as np

from abc import abstractmethod, ABC

from typing import Callable


class InterestRatesSetter(ABC):
    @abstractmethod
    def get_interest_rate_on_firm_short_term_loans_function(
        self,
        central_bank_policy_rate: float,
        bank_markup_interest_rate_loans: np.ndarray,
    ) -> Callable[[int], float]:
        pass

    @abstractmethod
    def get_interest_rate_on_firm_long_term_loans_function(
        self,
        central_bank_policy_rate: float,
        bank_markup_interest_rate_loans: np.ndarray,
    ) -> Callable[[int], float]:
        pass

    @abstractmethod
    def get_interest_rate_on_household_payday_loans_function(
        self,
        central_bank_policy_rate: float,
        bank_markup_interest_rate_loans: np.ndarray,
    ) -> Callable[[int], float]:
        pass

    @abstractmethod
    def get_interest_rate_on_household_consumption_expansion_loans_function(
        self,
        central_bank_policy_rate: float,
        bank_markup_interest_rate_loans: np.ndarray,
    ) -> Callable[[int], float]:
        pass

    @abstractmethod
    def get_interest_rate_on_mortgages_function(
        self,
        central_bank_policy_rate: float,
        bank_markup_interest_rate_loans: np.ndarray,
    ) -> Callable[[int], float]:
        pass

    @abstractmethod
    def compute_interest_rate_on_firms_deposits(
        self,
        central_bank_policy_rate: float,
        n_banks: int,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_overdraft_rate_on_firm_deposits(
        self,
        central_bank_policy_rate: float,
        bank_markup_interest_rate_overdraft_firm: np.ndarray,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_interest_rate_on_household_deposits(
        self,
        central_bank_policy_rate: float,
        n_banks: int,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_overdraft_rate_on_household_deposits(
        self,
        central_bank_policy_rate: float,
        bank_markup_interest_rate_overdraft_household: np.ndarray,
    ) -> np.ndarray:
        pass

    """
    @staticmethod
    @abstractmethod
    def compute_interest_rate_on_government_debt(central_bank_policy_rate: float) -> float:
        pass
    """


class DefaultInterestRatesSetter(InterestRatesSetter):
    def __init__(self, interest_noise_std: float):
        self.interest_noise_std = interest_noise_std

    def get_interest_rate_on_firm_short_term_loans_function(
        self,
        central_bank_policy_rate: float,
        bank_markup_interest_rate_loans: np.ndarray,
    ) -> Callable[[int], float]:
        def f(bank_id: int) -> float:
            return (1 + np.random.normal(0, self.interest_noise_std)) * (
                central_bank_policy_rate + bank_markup_interest_rate_loans[bank_id]
            )

        return f

    def get_interest_rate_on_firm_long_term_loans_function(
        self,
        central_bank_policy_rate: float,
        bank_markup_interest_rate_loans: np.ndarray,
    ) -> Callable[[int], float]:
        def f(bank_id: int) -> float:
            noise = np.random.uniform(0, self.interest_noise_std) - self.interest_noise_std / 2.0
            return (1 + noise) * (central_bank_policy_rate + bank_markup_interest_rate_loans[bank_id])

        return f

    def get_interest_rate_on_household_payday_loans_function(
        self,
        central_bank_policy_rate: float,
        bank_markup_interest_rate_loans: np.ndarray,
    ) -> Callable[[int], float]:
        def f(bank_id: int) -> float:
            noise = np.random.uniform(0, self.interest_noise_std) - self.interest_noise_std / 2.0
            return (1 + noise) * (central_bank_policy_rate + bank_markup_interest_rate_loans[bank_id])

        return f

    def get_interest_rate_on_household_consumption_expansion_loans_function(
        self,
        central_bank_policy_rate: float,
        bank_markup_interest_rate_loans: np.ndarray,
    ) -> Callable[[int], float]:
        def f(bank_id: int) -> float:
            noise = np.random.uniform(0, self.interest_noise_std) - self.interest_noise_std / 2.0
            return (1 + noise) * (central_bank_policy_rate + bank_markup_interest_rate_loans[bank_id])

        return f

    def get_interest_rate_on_mortgages_function(
        self,
        central_bank_policy_rate: float,
        bank_markup_interest_rate_loans: np.ndarray,
    ) -> Callable[[int], float]:
        def f(bank_id: int) -> float:
            noise = np.random.uniform(0, self.interest_noise_std) - self.interest_noise_std / 2.0
            return (1 + noise) * (central_bank_policy_rate + bank_markup_interest_rate_loans[bank_id])

        return f

    def compute_interest_rate_on_firms_deposits(
        self,
        central_bank_policy_rate: float,
        n_banks: int,
    ) -> np.ndarray:
        noise = np.random.uniform(0, self.interest_noise_std, n_banks) - self.interest_noise_std / 2.0
        return (1 + noise) * central_bank_policy_rate

    def compute_overdraft_rate_on_firm_deposits(
        self,
        central_bank_policy_rate: float,
        bank_markup_interest_rate_overdraft_firm: np.ndarray,
    ) -> np.ndarray:
        noise = (
            np.random.uniform(
                0,
                self.interest_noise_std,
                bank_markup_interest_rate_overdraft_firm.shape,
            )
            - self.interest_noise_std / 2.0
        )
        return (1 + noise) * (central_bank_policy_rate + bank_markup_interest_rate_overdraft_firm)

    def compute_interest_rate_on_household_deposits(
        self,
        central_bank_policy_rate: float,
        n_banks: int,
    ) -> np.ndarray:
        noise = np.random.uniform(0, self.interest_noise_std, n_banks) - self.interest_noise_std / 2.0
        return (1 + noise) * central_bank_policy_rate

    def compute_overdraft_rate_on_household_deposits(
        self,
        central_bank_policy_rate: float,
        bank_markup_interest_rate_overdraft_household: np.ndarray,
    ) -> np.ndarray:
        noise = (
            np.random.uniform(
                0,
                self.interest_noise_std,
                bank_markup_interest_rate_overdraft_household.shape,
            )
            - self.interest_noise_std / 2.0
        )
        return (1 + noise) * (central_bank_policy_rate + bank_markup_interest_rate_overdraft_household)

    """
    @staticmethod
    def compute_interest_rate_on_government_debt(central_bank_policy_rate: float) -> float:
        return central_bank_policy_rate
    """
