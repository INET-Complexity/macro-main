"""Individual reservation wage determination.

This module implements strategies for calculating individual reservation
wages through:
- Employment status consideration
- Historical wage analysis
- Unemployment benefit incorporation
- Time-based adjustments

The implementation handles:
- Minimum acceptable wages
- Employment transitions
- Benefit effects
- Wage history impact
"""

from abc import ABC, abstractmethod

import numpy as np

from macromodel.agents.individuals.individual_properties import ActivityStatus


class ReservationWageSetter(ABC):
    """Abstract base class for reservation wage calculation.

    This class defines strategies for determining minimum acceptable
    wages based on:
    - Current activity status
    - Previous wage history
    - Unemployment benefits
    - Time preferences

    The strategies consider:
    - Employment status
    - Historical earnings
    - Social benefits
    - Duration effects

    Attributes:
        unemployed_reservation_wage_timespan (int): Number of periods to
            consider for unemployed individuals' wage history
    """

    def __init__(
        self,
        unemployed_reservation_wage_timespan: int,
    ):
        """Initialize reservation wage setter.

        Args:
            unemployed_reservation_wage_timespan (int): Number of periods
                to consider for unemployed individuals' wage history
        """
        self.unemployed_reservation_wage_timespan = int(unemployed_reservation_wage_timespan)

    @abstractmethod
    def compute_reservation_wages(
        self,
        historic_wages: np.ndarray,
        current_individuals_activity: np.ndarray,
        current_unemployment_benefits_by_individual: float,
    ) -> np.ndarray:
        """Calculate reservation wages for individuals.

        Args:
            historic_wages (np.ndarray): Past wages by individual
            current_individuals_activity (np.ndarray): Activity status
            current_unemployment_benefits_by_individual (float):
                Per person unemployment benefit

        Returns:
            np.ndarray: Reservation wages by individual
        """
        pass


class DefaultReservationWageSetter(ReservationWageSetter):
    """Default implementation of reservation wage calculation.

    This class implements wage determination through:
    - Status-based differentiation
    - Historical wage consideration
    - Benefit level incorporation
    - Time-based averaging

    The approach:
    - Uses current wages for employed
    - Considers history for unemployed
    - Ensures minimum at benefit level
    - Applies time window for history
    """

    def compute_reservation_wages(
        self,
        historic_wages: np.ndarray,
        current_individuals_activity: np.ndarray,
        current_unemployment_benefits_by_individual: float,
    ) -> np.ndarray:
        """Calculate reservation wages for individuals.

        Determines wages based on:
        - Employment status (current/last wage)
        - Unemployment duration (wage history)
        - Benefit levels (minimum threshold)

        The calculation:
        - Uses current wage for employed
        - Takes max of benefits and history for unemployed
        - Sets zero for non-participants

        Args:
            historic_wages (np.ndarray): Past wages by individual
            current_individuals_activity (np.ndarray): Activity status
            current_unemployment_benefits_by_individual (float):
                Per person unemployment benefit

        Returns:
            np.ndarray: Reservation wages by individual
        """
        reservation_wages = np.zeros_like(current_individuals_activity)

        # For employed individuals
        reservation_wages[current_individuals_activity == ActivityStatus.EMPLOYED] = historic_wages[-1][
            current_individuals_activity == ActivityStatus.EMPLOYED
        ]

        # For unemployed individuals
        if self.unemployed_reservation_wage_timespan == 0:
            reservation_wages[current_individuals_activity == ActivityStatus.UNEMPLOYED] = (
                current_unemployment_benefits_by_individual
            )
        else:
            if np.sum(current_individuals_activity == ActivityStatus.UNEMPLOYED) > 0:
                reservation_wages[current_individuals_activity == ActivityStatus.UNEMPLOYED] = np.maximum(
                    current_unemployment_benefits_by_individual,
                    np.array(historic_wages[-self.unemployed_reservation_wage_timespan :])[
                        :,
                        current_individuals_activity == ActivityStatus.UNEMPLOYED,
                    ].mean(axis=0),
                )

        return reservation_wages
