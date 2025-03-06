"""Interest rate determination for banking products.

This module implements strategies for setting interest rates on various
banking products through:
- Policy rate transmission
- Product-specific adjustments
- Pass-through mechanisms
- Error correction terms

The rates cover:
- Firm loans (short/long-term)
- Household loans (consumption/mortgages)
- Deposits (firm/household)
- Overdraft facilities
"""

from abc import ABC, abstractmethod

import numpy as np


class InterestRatesSetter(ABC):
    """Abstract base class for interest rate determination.

    This class defines strategies for setting interest rates on all
    banking products through:
    - Policy rate transmission
    - Pass-through mechanisms
    - Error correction adjustments
    - Product differentiation

    The rates include:
    - Firm lending rates
    - Household lending rates
    - Deposit rates
    - Overdraft rates
    """

    @abstractmethod
    def get_interest_rates_on_short_term_firm_loans(
        self,
        central_bank_policy_rate: float,
        prev_interest_rates_on_short_term_firm_loans: np.ndarray,
        firm_pt: float,
        firm_ect: float,
    ) -> np.ndarray:
        """Calculate short-term firm loan rates.

        Args:
            central_bank_policy_rate (float): Current policy rate
            prev_interest_rates_on_short_term_firm_loans (np.ndarray):
                Previous period's rates
            firm_pt (float): Firm pass-through parameter
            firm_ect (float): Firm error correction parameter

        Returns:
            np.ndarray: New short-term firm loan rates by bank
        """
        pass

    @abstractmethod
    def get_interest_rates_on_long_term_firm_loans(
        self,
        central_bank_policy_rate: float,
        prev_interest_rates_on_long_term_firm_loans: np.ndarray,
        firm_pt: float,
        firm_ect: float,
    ) -> np.ndarray:
        """Calculate long-term firm loan rates.

        Args:
            central_bank_policy_rate (float): Current policy rate
            prev_interest_rates_on_long_term_firm_loans (np.ndarray):
                Previous period's rates
            firm_pt (float): Firm pass-through parameter
            firm_ect (float): Firm error correction parameter

        Returns:
            np.ndarray: New long-term firm loan rates by bank
        """
        pass

    @abstractmethod
    def get_interest_rates_on_household_consumption_loans(
        self,
        central_bank_policy_rate: float,
        prev_interest_rate_on_hh_consumption_loans: np.ndarray,
        hh_cons_pt: float,
        hh_cons_ect: float,
    ) -> np.ndarray:
        """Calculate household consumption loan rates.

        Args:
            central_bank_policy_rate (float): Current policy rate
            prev_interest_rate_on_hh_consumption_loans (np.ndarray):
                Previous period's rates
            hh_cons_pt (float): Household consumption pass-through
            hh_cons_ect (float): Household consumption error correction

        Returns:
            np.ndarray: New household consumption loan rates by bank
        """
        pass

    @abstractmethod
    def get_interest_rate_on_mortgages(
        self,
        central_bank_policy_rate: float,
        prev_interest_rate_on_mortgages: np.ndarray,
        hh_mortgage_pt: float,
        hh_mortgage_ect: float,
    ) -> np.ndarray:
        """Calculate mortgage rates.

        Args:
            central_bank_policy_rate (float): Current policy rate
            prev_interest_rate_on_mortgages (np.ndarray):
                Previous period's rates
            hh_mortgage_pt (float): Mortgage pass-through parameter
            hh_mortgage_ect (float): Mortgage error correction parameter

        Returns:
            np.ndarray: New mortgage rates by bank
        """
        pass

    @abstractmethod
    def compute_interest_rate_on_firm_deposits(
        self,
        central_bank_policy_rate: float,
        prev_interest_rate_on_firm_deposits: np.ndarray,
        firm_pt: float,
        firm_ect: float,
    ) -> np.ndarray:
        """Calculate firm deposit rates.

        Args:
            central_bank_policy_rate (float): Current policy rate
            prev_interest_rate_on_firm_deposits (np.ndarray):
                Previous period's rates
            firm_pt (float): Firm pass-through parameter
            firm_ect (float): Firm error correction parameter

        Returns:
            np.ndarray: New firm deposit rates by bank
        """
        pass

    @abstractmethod
    def compute_overdraft_rate_on_firm_deposits(
        self,
        central_bank_policy_rate: float,
        prev_overdraft_rate_on_firm_deposits: np.ndarray,
        firm_pt: float,
        firm_ect: float,
    ) -> np.ndarray:
        """Calculate firm overdraft rates.

        Args:
            central_bank_policy_rate (float): Current policy rate
            prev_overdraft_rate_on_firm_deposits (np.ndarray):
                Previous period's rates
            firm_pt (float): Firm pass-through parameter
            firm_ect (float): Firm error correction parameter

        Returns:
            np.ndarray: New firm overdraft rates by bank
        """
        pass

    @abstractmethod
    def compute_interest_rate_on_household_deposits(
        self,
        central_bank_policy_rate: float,
        prev_interest_rate_on_hh_deposits: np.ndarray,
        hh_cons_pt: float,
        hh_cons_ect: float,
    ) -> np.ndarray:
        """Calculate household deposit rates.

        Args:
            central_bank_policy_rate (float): Current policy rate
            prev_interest_rate_on_hh_deposits (np.ndarray):
                Previous period's rates
            hh_cons_pt (float): Household pass-through parameter
            hh_cons_ect (float): Household error correction parameter

        Returns:
            np.ndarray: New household deposit rates by bank
        """
        pass

    @abstractmethod
    def compute_overdraft_rate_on_household_deposits(
        self,
        central_bank_policy_rate: float,
        prev_overdraft_rate_on_hh_deposits: np.ndarray,
        hh_cons_pt: float,
        hh_cons_ect: float,
    ) -> np.ndarray:
        """Calculate household overdraft rates.

        Args:
            central_bank_policy_rate (float): Current policy rate
            prev_overdraft_rate_on_hh_deposits (np.ndarray):
                Previous period's rates
            hh_cons_pt (float): Household pass-through parameter
            hh_cons_ect (float): Household error correction parameter

        Returns:
            np.ndarray: New household overdraft rates by bank
        """
        pass


class DefaultInterestRatesSetter(InterestRatesSetter):
    """Default implementation of interest rate setting.

    This class implements interest rate determination through:
    - Policy rate transmission
    - Pass-through mechanisms
    - Error correction adjustments
    - Product-specific spreads

    The approach:
    - Uses pass-through parameters
    - Applies error correction terms
    - Maintains rate differentials
    - Ensures rate consistency
    """

    def get_interest_rates_on_short_term_firm_loans(
        self,
        central_bank_policy_rate: float,
        prev_interest_rates_on_short_term_firm_loans: np.ndarray,
        firm_pt: float,
        firm_ect: float,
    ) -> np.ndarray:
        """Calculate short-term firm loan rates.

        Adjusts rates based on:
        - Previous period's rates
        - Policy rate changes
        - Pass-through effects
        - Error correction

        Args:
            central_bank_policy_rate (float): Current policy rate
            prev_interest_rates_on_short_term_firm_loans (np.ndarray):
                Previous period's rates
            firm_pt (float): Firm pass-through parameter
            firm_ect (float): Firm error correction parameter

        Returns:
            np.ndarray: New short-term firm loan rates by bank
        """
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
        """Calculate long-term firm loan rates.

        Adjusts rates based on:
        - Previous period's rates
        - Policy rate changes
        - Pass-through effects
        - Error correction

        Args:
            central_bank_policy_rate (float): Current policy rate
            prev_interest_rates_on_long_term_firm_loans (np.ndarray):
                Previous period's rates
            firm_pt (float): Firm pass-through parameter
            firm_ect (float): Firm error correction parameter

        Returns:
            np.ndarray: New long-term firm loan rates by bank
        """
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
        """Calculate household consumption loan rates.

        Adjusts rates based on:
        - Previous period's rates
        - Policy rate changes
        - Pass-through effects
        - Error correction

        Args:
            central_bank_policy_rate (float): Current policy rate
            prev_interest_rate_on_hh_consumption_loans (np.ndarray):
                Previous period's rates
            hh_cons_pt (float): Household consumption pass-through
            hh_cons_ect (float): Household consumption error correction

        Returns:
            np.ndarray: New household consumption loan rates by bank
        """
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
        """Calculate mortgage rates.

        Adjusts rates based on:
        - Previous period's rates
        - Policy rate changes
        - Pass-through effects
        - Error correction

        Args:
            central_bank_policy_rate (float): Current policy rate
            prev_interest_rate_on_mortgages (np.ndarray):
                Previous period's rates
            hh_mortgage_pt (float): Mortgage pass-through parameter
            hh_mortgage_ect (float): Mortgage error correction parameter

        Returns:
            np.ndarray: New mortgage rates by bank
        """
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
        """Calculate firm deposit rates.

        Sets rates equal to policy rate for all banks.

        Args:
            central_bank_policy_rate (float): Current policy rate
            prev_interest_rate_on_firm_deposits (np.ndarray):
                Previous period's rates
            firm_pt (float): Firm pass-through parameter
            firm_ect (float): Firm error correction parameter

        Returns:
            np.ndarray: New firm deposit rates by bank
        """
        return np.full(prev_interest_rate_on_firm_deposits.shape, central_bank_policy_rate)

    def compute_overdraft_rate_on_firm_deposits(
        self,
        central_bank_policy_rate: float,
        prev_overdraft_rate_on_firm_deposits: np.ndarray,
        firm_pt: float,
        firm_ect: float,
    ) -> np.ndarray:
        """Calculate firm overdraft rates.

        Adjusts rates based on:
        - Previous period's rates
        - Policy rate changes
        - Pass-through effects
        - Error correction

        Args:
            central_bank_policy_rate (float): Current policy rate
            prev_overdraft_rate_on_firm_deposits (np.ndarray):
                Previous period's rates
            firm_pt (float): Firm pass-through parameter
            firm_ect (float): Firm error correction parameter

        Returns:
            np.ndarray: New firm overdraft rates by bank
        """
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
        """Calculate household deposit rates.

        Sets rates equal to policy rate for all banks.

        Args:
            central_bank_policy_rate (float): Current policy rate
            prev_interest_rate_on_hh_deposits (np.ndarray):
                Previous period's rates
            hh_cons_pt (float): Household pass-through parameter
            hh_cons_ect (float): Household error correction parameter

        Returns:
            np.ndarray: New household deposit rates by bank
        """
        return np.full(prev_interest_rate_on_hh_deposits.shape, central_bank_policy_rate)

    def compute_overdraft_rate_on_household_deposits(
        self,
        central_bank_policy_rate: float,
        prev_overdraft_rate_on_hh_deposits: np.ndarray,
        hh_cons_pt: float,
        hh_cons_ect: float,
    ) -> np.ndarray:
        """Calculate household overdraft rates.

        Adjusts rates based on:
        - Previous period's rates
        - Policy rate changes
        - Pass-through effects
        - Error correction

        Args:
            central_bank_policy_rate (float): Current policy rate
            prev_overdraft_rate_on_hh_deposits (np.ndarray):
                Previous period's rates
            hh_cons_pt (float): Household pass-through parameter
            hh_cons_ect (float): Household error correction parameter

        Returns:
            np.ndarray: New household overdraft rates by bank
        """
        return prev_overdraft_rate_on_hh_deposits + hh_cons_ect * (
            prev_overdraft_rate_on_hh_deposits - hh_cons_pt * central_bank_policy_rate
        )
