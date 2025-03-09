"""Social housing management and rent calculation.

This module implements strategies for managing social housing,
particularly focusing on:
- Rent determination for social housing units
- Affordability considerations
- Household size adjustments
- Links to unemployment benefits

The social housing system aims to:
- Provide affordable housing options
- Scale rents with household ability to pay
- Maintain sustainable housing provision
- Support vulnerable households
"""

from abc import ABC, abstractmethod

import numpy as np


class SocialHousing(ABC):
    """Abstract base class for social housing management.

    This class defines strategies for determining social housing rents
    based on:
    - Unemployment benefit levels
    - Household size and composition
    - Affordability targets
    - Social policy objectives

    Attributes:
        rent_as_fraction_of_unemployment_rate (float): Target rent as a
            fraction of unemployment benefits, ensuring affordability
    """

    def __init__(self, rent_as_fraction_of_unemployment_rate: float):
        """Initialize social housing manager.

        Args:
            rent_as_fraction_of_unemployment_rate (float): Target rent as
                proportion of unemployment benefits (between 0 and 1)
        """
        self.rent_as_fraction_of_unemployment_rate = rent_as_fraction_of_unemployment_rate

    @abstractmethod
    def compute_social_housing_rent(
        self,
        current_unemployment_benefits_by_individual: float,
        current_household_size: np.ndarray,
    ) -> np.ndarray:
        """Calculate social housing rents for households.

        Determines appropriate rent levels considering:
        - Current unemployment benefit levels
        - Household size and composition
        - Affordability targets
        - Social policy objectives

        Args:
            current_unemployment_benefits_by_individual (float): Current
                unemployment benefit level
            current_household_size (np.ndarray): Size of each household

        Returns:
            np.ndarray: Calculated rent for each household
        """
        pass


class DefaultSocialHousing(SocialHousing):
    """Default implementation of social housing management.

    This class implements a simple rent calculation that:
    - Scales with unemployment benefits
    - Adjusts for household size
    - Maintains consistent affordability
    - Provides predictable housing costs

    The approach ensures:
    - Affordable rents for all households
    - Fair scaling with household size
    - Sustainable revenue generation
    - Clear and transparent pricing
    """

    def compute_social_housing_rent(
        self,
        current_unemployment_benefits_by_individual: float,
        current_household_size: np.ndarray,
    ) -> np.ndarray:
        """Calculate social housing rents using default strategy.

        Computes rents as a fixed fraction of unemployment benefits,
        scaled by household size to account for larger units.

        Args:
            current_unemployment_benefits_by_individual (float): Current
                unemployment benefit level
            current_household_size (np.ndarray): Size of each household

        Returns:
            np.ndarray: Calculated rent for each household, proportional
                to both benefits and household size
        """
        return (
            self.rent_as_fraction_of_unemployment_rate
            * current_unemployment_benefits_by_individual
            * current_household_size
        )
