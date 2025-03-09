from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class OfferedWageSetter(ABC):
    """Abstract base class for determining wage offers to potential employees.

    This class defines strategies for calculating wage offers based on:
    - Current employment and labor inputs
    - Historical wage levels
    - Labor market conditions and tightness
    - Unemployment benefits (reservation wages)

    The wage offer process aims to:
    - Attract necessary labor inputs
    - Remain competitive in the labor market
    - Account for market conditions
    - Respect minimum wage floors
    """

    @abstractmethod
    def get_offered_wage_given_labour_inputs_function(
        self,
        firm_employments: list[np.ndarray],
        current_individual_labour_inputs: np.ndarray,
        previous_employee_income: np.ndarray,
        historic_desired_labour_inputs: list[np.ndarray],
        historic_realised_labour_inputs: list[np.ndarray],
        unemployment_benefits_by_individual: float,
    ) -> Callable[[int, float | np.ndarray], float | np.ndarray]:
        """Create a function that calculates wage offers for given labor inputs.

        Returns a callable that firms can use to determine appropriate wage
        offers for different levels of labor input, considering:
        - Current employment patterns
        - Recent wage history
        - Market tightness
        - Minimum wage requirements

        Args:
            firm_employments (list[np.ndarray]): Current employee assignments
                by firm
            current_individual_labour_inputs (np.ndarray): Current labor
                input levels by individual
            previous_employee_income (np.ndarray): Previous period wages
            historic_desired_labour_inputs (list[np.ndarray]): Time series
                of desired labor inputs by firm
            historic_realised_labour_inputs (list[np.ndarray]): Time series
                of achieved labor inputs by firm
            unemployment_benefits_by_individual (float): Minimum wage floor
                set by unemployment benefits

        Returns:
            Callable[[int, float | np.ndarray], float | np.ndarray]: Function
                that takes firm ID and labor inputs and returns wage offers
        """
        pass


class DefaultOfferedWageSetter(OfferedWageSetter):
    """Default implementation of wage offer calculation.

    This class implements a strategy that:
    1. Calculates market tightness markup from hiring history
    2. Determines base wage rates from current wages
    3. Adjusts offers based on market conditions
    4. Ensures offers exceed unemployment benefits

    The approach aims to:
    - Reflect labor market conditions in wage offers
    - Maintain wage competitiveness
    - Account for firm-specific hiring difficulties
    - Provide fair compensation

    Attributes:
        labour_market_tightness_markup_scale (float): Scale factor for
            wage adjustments based on hiring difficulty
        markup_time_span (int): Number of periods to consider when
            calculating labor market tightness markup
    """

    def __init__(
        self,
        labour_market_tightness_markup_scale: float,
        markup_time_span: int,
    ):
        """Initialize the wage offer setter with markup parameters.

        Args:
            labour_market_tightness_markup_scale (float): Scale factor for
                wage adjustments based on hiring difficulty
            markup_time_span (int): Number of periods to consider when
                calculating labor market tightness markup
        """
        self.labour_market_tightness_markup_scale = labour_market_tightness_markup_scale
        self.markup_time_span = markup_time_span

    def get_offered_wage_given_labour_inputs_function(
        self,
        firm_employments: list[np.ndarray],
        current_individual_labour_inputs: np.ndarray,
        previous_employee_income: np.ndarray,
        historic_desired_labour_inputs: list[np.ndarray],
        historic_realised_labour_inputs: list[np.ndarray],
        unemployment_benefits_by_individual: float,
    ) -> Callable[[int, float | np.ndarray], float | np.ndarray]:
        """Create a function for calculating wage offers based on market conditions.

        The method:
        1. Calculates market tightness markup for each firm
        2. Determines base wage rates from current wages
        3. Creates a function that applies these calculations

        The markup calculation considers:
        - Historical hiring success rates
        - Recent labor market conditions
        - Firm-specific hiring difficulties

        Args:
            firm_employments (list[np.ndarray]): Employee assignments by firm
            current_individual_labour_inputs (np.ndarray): Current labor inputs
            previous_employee_income (np.ndarray): Previous period wages
            historic_desired_labour_inputs (list[np.ndarray]): Desired labor
                inputs history
            historic_realised_labour_inputs (list[np.ndarray]): Achieved labor
                inputs history
            unemployment_benefits_by_individual (float): Minimum wage floor

        Returns:
            Callable[[int, float | np.ndarray], float | np.ndarray]: Function
                that calculates appropriate wage offers
        """
        new_individual_wages = np.zeros_like(current_individual_labour_inputs)
        for firm_ind in range(len(firm_employments)):
            # Labour market tightness mark-up
            markup = (
                self.labour_market_tightness_markup_scale
                * 1.0
                / self.markup_time_span
                * sum(
                    [
                        (
                            max(
                                0,
                                (
                                    historic_desired_labour_inputs[-t][firm_ind]
                                    - historic_realised_labour_inputs[-t][firm_ind]
                                )
                                / historic_desired_labour_inputs[-t][firm_ind],
                            )
                            if historic_desired_labour_inputs[-t][firm_ind] > 0
                            else 0.0
                        )
                        for t in range(
                            1,
                            min(
                                len(historic_desired_labour_inputs),
                                self.markup_time_span + 1,
                            ),
                        )
                    ]
                )
            )

            # Wage by unit of productivity
            new_individual_wages[firm_employments[firm_ind]] = (
                (1 + markup)
                * previous_employee_income[firm_employments[firm_ind]].sum()
                / current_individual_labour_inputs[firm_employments[firm_ind]].sum()
            )

        def f(firm_id: int, labour_inputs: float | np.ndarray) -> float | np.ndarray:
            """Calculate wage offer for given firm and labor inputs.

            Applies the pre-calculated wage rates and ensures offers exceed
            unemployment benefits.

            Args:
                firm_id (int): ID of the firm making the offer
                labour_inputs (float | np.ndarray): Proposed labor input level(s)

            Returns:
                float | np.ndarray: Wage offer(s) that exceed unemployment benefits
            """
            return np.maximum(
                unemployment_benefits_by_individual,
                labour_inputs * new_individual_wages[firm_id],
            )

        return f
