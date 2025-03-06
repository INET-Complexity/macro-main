"""Household rental price determination implementation.

This module implements rental price management through:
- Current rent adjustment
- New property rent setting
- Existing property rent updates
- Inflation indexation

The implementation handles:
- Rent level calculation
- Property value consideration
- Inflation adjustment
- Markup application
"""

from abc import ABC, abstractmethod

import numpy as np


class RentSetter(ABC):
    """Abstract base class for rental price management.

    Defines interface for determining rental prices through:
    - Current rent updates
    - New property pricing
    - Existing property adjustments
    - Inflation consideration

    Attributes:
        partial_rent_inflation_indexation (float): Inflation pass-through rate
        new_property_rent_markup (float): Initial rent markup
        offered_rent_decrease (float): Rent reduction rate
    """

    def __init__(
        self,
        partial_rent_inflation_indexation: float,
        new_property_rent_markup: float,
        offered_rent_decrease: float,
    ):
        """Initialize rental price management.

        Args:
            partial_rent_inflation_indexation (float): Inflation pass-through rate
            new_property_rent_markup (float): Initial rent markup
            offered_rent_decrease (float): Rent reduction rate
        """
        self.partial_rent_inflation_indexation = partial_rent_inflation_indexation
        self.new_property_rent_markup = new_property_rent_markup
        self.offered_rent_decrease = offered_rent_decrease

    @abstractmethod
    def compute_rent(
        self,
        current_rent: np.ndarray,
        historic_inflation: np.ndarray,
    ) -> np.ndarray:
        """Calculate updated rental prices.

        Args:
            current_rent (np.ndarray): Current rental prices
            historic_inflation (np.ndarray): Past inflation rates

        Returns:
            np.ndarray: Updated rental prices
        """
        pass

    @abstractmethod
    def compute_offered_rent_for_new_properties(
        self,
        property_value: np.ndarray,
        observed_fraction_rent_value: np.ndarray,
    ) -> np.ndarray:
        """Calculate initial rental offers for new properties.

        Args:
            property_value (np.ndarray): Property values
            observed_fraction_rent_value (np.ndarray): Rent/value ratios

        Returns:
            np.ndarray: Initial rental prices
        """
        pass

    @abstractmethod
    def compute_offered_rent_for_existing_properties(self, current_offered_rent: np.ndarray) -> np.ndarray:
        """Update rental offers for existing properties.

        Args:
            current_offered_rent (np.ndarray): Current rental offers

        Returns:
            np.ndarray: Updated rental prices
        """
        pass


class ConstantRentSetter(RentSetter):
    """Simple rental price implementation using constant rates.

    Maintains unchanged rental prices for:
    - Current rents
    - New property offers
    - Existing property offers
    """

    def compute_rent(self, current_rent: np.ndarray, historic_inflation: np.ndarray) -> np.ndarray:
        """Return unchanged rental prices.

        Args:
            current_rent (np.ndarray): Current rental prices
            historic_inflation (np.ndarray): Past inflation rates

        Returns:
            np.ndarray: Same rental prices
        """
        return current_rent

    def compute_offered_rent_for_new_properties(
        self, property_value: np.ndarray, observed_fraction_rent_value: np.ndarray
    ) -> np.ndarray:
        """Set new property rents equal to values.

        Args:
            property_value (np.ndarray): Property values
            observed_fraction_rent_value (np.ndarray): Rent/value ratios

        Returns:
            np.ndarray: Property values as rents
        """
        return property_value

    def compute_offered_rent_for_existing_properties(self, current_offered_rent: np.ndarray) -> np.ndarray:
        """Return unchanged rental offers.

        Args:
            current_offered_rent (np.ndarray): Current rental offers

        Returns:
            np.ndarray: Same rental prices
        """
        return current_offered_rent


class DefaultRentSetter(RentSetter):
    """Default implementation of rental price management.

    Implements rent determination through:
    - Inflation indexation
    - Value-based pricing
    - Regular adjustments
    """

    def compute_rent(
        self,
        current_rent: np.ndarray,
        historic_inflation: np.ndarray,
    ) -> np.ndarray:
        """Calculate rents using default behavior.

        Updates prices through:
        - Inflation indexation
        - Minimum zero change
        - Current rent basis

        Args:
            current_rent (np.ndarray): Current rental prices
            historic_inflation (np.ndarray): Past inflation rates

        Returns:
            np.ndarray: Updated rental prices
        """
        return (
            1
            + np.maximum(
                0.0,
                self.partial_rent_inflation_indexation * historic_inflation[-1],
            )
        ) * current_rent

    def compute_offered_rent_for_new_properties(
        self,
        property_value: np.ndarray,
        observed_fraction_rent_value: np.ndarray,
    ) -> np.ndarray:
        """Calculate new property rents using default behavior.

        Determines prices through:
        - Value-based calculation
        - Initial markup
        - Market ratio consideration

        Args:
            property_value (np.ndarray): Property values
            observed_fraction_rent_value (np.ndarray): Rent/value ratios

        Returns:
            np.ndarray: Initial rental prices
        """
        return (1 + self.new_property_rent_markup) * (
            observed_fraction_rent_value[0] * property_value + observed_fraction_rent_value[1]
        )

    def compute_offered_rent_for_existing_properties(self, current_offered_rent: np.ndarray) -> np.ndarray:
        """Update existing property rents using default behavior.

        Adjusts prices through:
        - Standard decrease rate
        - Current offer basis

        Args:
            current_offered_rent (np.ndarray): Current rental offers

        Returns:
            np.ndarray: Updated rental prices
        """
        return (1 - self.offered_rent_decrease) * current_offered_rent
