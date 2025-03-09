"""Social benefits calculation and management for central government.

This module implements various strategies for determining and updating
social benefit levels, including:
- Unemployment benefits
- Regular household transfers
- Benefit adjustments based on economic conditions

The benefit calculations consider:
- Historical benefit levels
- Inflation rates
- Economic growth
- Unemployment rates
- Statistical models for benefit prediction
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class SocialBenefitsSetter(ABC):
    """Abstract base class for determining social benefit levels.

    This class defines strategies for calculating two main types of benefits:
    1. Unemployment benefits for jobless individuals
    2. Regular transfers to households

    The benefit calculation process considers:
    - Previous benefit levels
    - Price level changes
    - Economic growth rates
    - Labor market conditions
    - Statistical models for prediction
    """

    @abstractmethod
    def compute_unemployment_benefits(
        self,
        prev_unemployment_benefits: float,
        historic_ppi_inflation: np.ndarray,
        current_estimated_growth: float,
        current_unemployment_rate: float,
        model: Optional[Any],
    ) -> float:
        """Calculate unemployment benefit levels.

        Determines appropriate unemployment benefits considering:
        - Previous benefit levels
        - Historical inflation rates
        - Economic growth prospects
        - Current unemployment situation
        - Model-based predictions if available

        Args:
            prev_unemployment_benefits (float): Previous period's benefit level
            historic_ppi_inflation (np.ndarray): Historical inflation rates
            current_estimated_growth (float): Expected economic growth rate
            current_unemployment_rate (float): Current unemployment rate
            model (Optional[Any]): Statistical model for benefit prediction

        Returns:
            float: New unemployment benefit level
        """
        pass

    @abstractmethod
    def compute_regular_transfer_to_households(
        self,
        prev_regular_transfer_to_households: float,
        historic_ppi_inflation: np.ndarray,
        current_estimated_growth: float,
        current_unemployment_rate: float,
        model: Optional[Any],
    ) -> float:
        """Calculate regular household transfer levels.

        Determines appropriate regular transfer amounts considering:
        - Previous transfer levels
        - Historical inflation rates
        - Economic growth prospects
        - Current unemployment situation
        - Model-based predictions if available

        Args:
            prev_regular_transfer_to_households (float): Previous transfer level
            historic_ppi_inflation (np.ndarray): Historical inflation rates
            current_estimated_growth (float): Expected economic growth rate
            current_unemployment_rate (float): Current unemployment rate
            model (Optional[Any]): Statistical model for transfer prediction

        Returns:
            float: New regular transfer amount
        """
        pass


class ConstantSocialBenefitsSetter(SocialBenefitsSetter):
    """Implementation of constant social benefits.

    This class maintains fixed benefit levels by:
    - Keeping unemployment benefits constant
    - Maintaining same regular transfer amounts
    - Ignoring economic conditions

    This approach is useful for:
    - Model testing and validation
    - Scenarios with fixed social policies
    - Isolating effects of benefit changes
    """

    def compute_unemployment_benefits(
        self,
        prev_unemployment_benefits: float,
        historic_ppi_inflation: np.ndarray,
        current_estimated_growth: float,
        current_unemployment_rate: float,
        model: Optional[Any],
    ) -> float:
        """Keep unemployment benefits constant.

        Returns the same benefit level regardless of economic conditions.

        Args:
            [same as parent class]

        Returns:
            float: Previous unemployment benefit level (unchanged)
        """
        return prev_unemployment_benefits

    def compute_regular_transfer_to_households(
        self,
        prev_regular_transfer_to_households: float,
        historic_ppi_inflation: np.ndarray,
        current_estimated_growth: float,
        current_unemployment_rate: float,
        model: Optional[Any],
    ) -> float:
        """Keep regular transfers constant.

        Returns the same transfer amount regardless of economic conditions.

        Args:
            [same as parent class]

        Returns:
            float: Previous transfer amount (unchanged)
        """
        return prev_regular_transfer_to_households


class GrowthSocialBenefitsSetter(SocialBenefitsSetter):
    """Implementation of growth-adjusted social benefits.

    This class adjusts benefits based on economic growth:
    - Unemployment benefits increase in downturns
    - Regular transfers grow with the economy
    - Counter-cyclical unemployment support
    - Pro-cyclical regular transfers

    This approach provides:
    - Automatic stabilization
    - Growth sharing through transfers
    - Counter-cyclical social protection
    """

    def compute_unemployment_benefits(
        self,
        prev_unemployment_benefits: float,
        historic_ppi_inflation: np.ndarray,
        current_estimated_growth: float,
        current_unemployment_rate: float,
        model: Optional[Any],
    ) -> float:
        """Calculate growth-adjusted unemployment benefits.

        Increases benefits when growth is negative and maintains
        them when growth is positive, providing counter-cyclical support.

        Args:
            [same as parent class]

        Returns:
            float: Growth-adjusted unemployment benefit level
        """
        return max(1.0, 1 / (1 + current_estimated_growth)) * prev_unemployment_benefits

    def compute_regular_transfer_to_households(
        self,
        prev_regular_transfer_to_households: float,
        historic_ppi_inflation: np.ndarray,
        current_estimated_growth: float,
        current_unemployment_rate: float,
        model: Optional[Any],
    ) -> float:
        """Calculate growth-adjusted regular transfers.

        Increases transfers in proportion to economic growth,
        sharing prosperity through social transfers.

        Args:
            [same as parent class]

        Returns:
            float: Growth-adjusted transfer amount
        """
        return (1 + current_estimated_growth) * prev_regular_transfer_to_households


class DefaultSocialBenefitsSetter(SocialBenefitsSetter):
    """Default implementation using statistical models.

    This class determines benefits using:
    - Statistical prediction models
    - Historical benefit levels as fallback
    - Current economic conditions
    - Inflation and unemployment data

    The approach provides:
    - Data-driven benefit determination
    - Robust fallback mechanisms
    - Economic condition consideration
    """

    def compute_unemployment_benefits(
        self,
        prev_unemployment_benefits: float,
        historic_ppi_inflation: np.ndarray,
        current_estimated_growth: float,
        current_unemployment_rate: float,
        model: Optional[Any],
    ) -> float:
        """Calculate model-based unemployment benefits.

        Uses statistical model predictions based on inflation and
        unemployment rates, with fallback to previous levels.

        Args:
            [same as parent class]

        Returns:
            float: Model-predicted or previous unemployment benefit level
        """
        if model is None:
            return prev_unemployment_benefits
        pred = model.predict(np.array([[historic_ppi_inflation[-1], current_unemployment_rate]]))[0]
        return pred

    def compute_regular_transfer_to_households(
        self,
        prev_regular_transfer_to_households: float,
        historic_ppi_inflation: np.ndarray,
        current_estimated_growth: float,
        current_unemployment_rate: float,
        model: Optional[Any],
    ) -> float:
        """Calculate model-based regular transfers.

        Uses statistical model predictions based on inflation and
        unemployment rates, with fallback to previous levels.

        Args:
            [same as parent class]

        Returns:
            float: Model-predicted or previous transfer amount
        """
        if model is None:
            return prev_regular_transfer_to_households
        pred = model.predict(np.array([[historic_ppi_inflation[-1], current_unemployment_rate]]))[0]
        return pred
