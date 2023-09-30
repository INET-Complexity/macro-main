import numpy as np
import pandas as pd

from abc import abstractmethod, ABC


class SyntheticCreditMarket(ABC):
    @abstractmethod
    def __init__(
        self,
        country_name: str,
        year: int,
    ):
        self.country_name = country_name
        self.year = year

        # Credit market inet_data
        self.credit_market_data = pd.DataFrame()

    @abstractmethod
    def create(
        self,
        bank_data: pd.DataFrame,
        initial_firm_debt: np.ndarray,
        initial_household_other_debt: np.ndarray,
        initial_household_mortgage_debt: np.ndarray,
        firms_corresponding_bank: np.ndarray,
        households_corresponding_bank: np.ndarray,
        initial_firm_loan_maturity: int,
        household_consumption_loan_maturity: int,
        mortgage_maturity: int,
        assume_zero_initial_firm_debt: bool,
    ) -> None:
        pass

    @abstractmethod
    def set_initial_loans(
        self,
        bank_data: pd.DataFrame,
        initial_firm_debt: np.ndarray,
        initial_household_other_debt: np.ndarray,
        initial_household_mortgage_debt: np.ndarray,
        firms_corresponding_bank: np.ndarray,
        households_corresponding_bank: np.ndarray,
        initial_firm_loan_maturity: int,
        household_consumption_loan_maturity: int,
        mortgage_maturity: int,
        assume_zero_initial_firm_debt: bool,
    ) -> None:
        pass
